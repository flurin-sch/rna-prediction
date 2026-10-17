from pathlib import Path
import json
import hashlib
import os
import tempfile

import yaml

PATH_KEYS = ("data_dir", "reference_fasta", "prepared_dir", "cache_dir", "run_dir")


def load_config(path):
    path = Path(path).resolve()
    with path.open() as handle:
        config = yaml.safe_load(handle)
    root = (path.parent / config.get("project_root", ".")).resolve()
    config["project_root"] = str(root)
    for key in PATH_KEYS:
        config[key] = str((root / config[key]).resolve())
    checkpoint = config["extraction"].get("checkpoint_path")
    if checkpoint:
        config["extraction"]["checkpoint_path"] = str((root / checkpoint).resolve())
    r = config["regions"]
    if r["max_candidates"] < 0 or r["radius"] <= 0:
        raise ValueError("Invalid candidate count or radius")
    if any(r[k] <= 0 or r[k] % 2 for k in ("promoter_bp", "candidate_bp")):
        raise ValueError("Region widths must be positive even integers")
    if r["window_bp"] > 1048576 or r["window_bp"] % 2048 or r["window_bp"] < 2 * r["radius"] + max(r["promoter_bp"], r["candidate_bp"]):
        raise ValueError("AlphaGenome window must be a multiple of 2048 and cover all regions")
    if len(set(config["tracks"])) != len(config["tracks"]) or "DNase" not in config["tracks"]:
        raise ValueError("Tracks must be unique and include DNase")
    if config["model"]["embedding_dim"] != 3072:
        raise ValueError("Contextual AlphaGenome extraction requires embedding_dim=3072")
    if config["extraction"]["batch_size"] < 1 or config["extraction"]["shard_windows"] < 1:
        raise ValueError("Extraction batch/shard sizes must be positive")
    if config["train"]["batch_size"] < 1 or config["train"]["cpu_threads"] < 1 or config["train"]["num_workers"] < 0:
        raise ValueError("Invalid training batch size, thread count, or worker count")
    return config


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def read_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def atomic_parquet(frame, path):
    path = Path(path)
    temp = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temp, index=False)
    os.replace(temp, path)

