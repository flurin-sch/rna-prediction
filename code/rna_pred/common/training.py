"""Deterministic optimization, prediction, checkpoint IO, and stopping utilities."""
from pathlib import Path
import os
import random
import time
import numpy as np
import torch

def seed_everything(seed, cpu_threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cpu_threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def resolve_device(requested):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(requested)


def atomic_torch_save(path, value):
    path = Path(path)
    temp = path.with_suffix(".tmp.pt")
    torch.save(value, temp)
    os.replace(temp, path)


def rng_state(generator):
    numpy = np.random.get_state()
    return {"python": random.getstate(), "numpy": [numpy[0], numpy[1].tolist(), *numpy[2:]],
            "torch": torch.get_rng_state(), "loader": generator.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state, generator):
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(state["torch"].cpu())
    generator.set_state(state["loader"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])


def initial_stopping_state():
    return {"best_val_loss": None, "meaningful_best_train_loss": None,
            "train_plateau_epochs": 0, "val_rise_epochs": 0}


def update_stopping(state, train_loss, val_loss, settings):
    """Keep strict validation minima independently of plateau and deterioration counters."""
    if not np.isfinite([train_loss, val_loss]).all() or min(train_loss, val_loss) < 0:
        raise ValueError("Stopping requires finite nonnegative losses")
    state = dict(state)
    reference = state["meaningful_best_train_loss"]
    meaningful = reference is None or reference - train_loss > settings.get("train_min_relative_improvement", 0.001) * max(abs(reference), 1e-12)
    if meaningful:
        state["meaningful_best_train_loss"], state["train_plateau_epochs"] = train_loss, 0
    else:
        state["train_plateau_epochs"] += 1
    best = state["best_val_loss"]
    improved = best is None or val_loss < best
    if improved:
        state["best_val_loss"] = val_loss
    best = state["best_val_loss"]
    rising = val_loss - best > settings.get("val_min_relative_rise", 0.001) * max(abs(best), 1e-12)
    state["val_rise_epochs"] = state["val_rise_epochs"] + 1 if rising else 0
    stopped = min(state["train_plateau_epochs"], state["val_rise_epochs"]) >= settings["patience"]
    return state, improved, stopped



def sequence_forward(model, batch, device):
    return model(batch["sequence"].to(device), batch["features"].to(device), batch["valid"].to(device))


def run_predictions(model, loader, device, forward=sequence_forward):
    model.eval()
    predictions = np.empty(len(loader.dataset), dtype=np.float64)
    with torch.inference_mode():
        for batch in loader:
            output = forward(model, batch, device)
            if not torch.isfinite(output).all():
                raise ValueError("Model produced nonfinite predictions")
            predictions[batch["index"].numpy()] = output.cpu().numpy()
    return predictions


def neural_epochs(model, train_loader, train_eval_loader, val_loader, optimizer, device,
                  loss_fn, *, start, epochs, stopped, forward=sequence_forward, predict_fn=None):
    """One optimization/evaluation loop; callers retain their checkpoint/selection policy.

    Prediction and loss adapters support different model inputs without changing the
    ordering of optimizer updates, RNG calls, or complete-dataset evaluations.
    """
    if predict_fn is None:
        def predict_fn(model, loader, device):
            return run_predictions(model, loader, device, forward)
    for epoch in range(start, epochs):
        if stopped():
            break
        began = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            output = forward(model, batch, device)
            loss = loss_fn(batch, output)
            if not torch.isfinite(loss):
                raise ValueError("Training loss became nonfinite")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(output)
        train_predictions = predict_fn(model, train_eval_loader, device)
        val_predictions = predict_fn(model, val_loader, device)
        yield epoch, total_loss, train_predictions, val_predictions, began
