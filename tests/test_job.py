import copy
from pathlib import Path
import random

import numpy as np
import pandas as pd
import pytest
import torch

from rna_pred.common import job
from rna_pred.common.data.cache import build_cache
from rna_pred.common.data.dataset import PreparedData
from rna_pred.common.data.prepare import prepare
from rna_pred.baseline.workflow import predict, train


def test_staging_symlinks_identity_predictions_and_persistent_outputs(synthetic_project, fake_alphagenome, tmp_path):
    config = synthetic_project
    prepare(config)
    build_cache(config, device="cpu")
    best = train(config)
    original = predict(config, best)
    prepared_id = PreparedData(config).manifest["identity"]
    for key in ("prepared_dir", "cache_dir"):
        source = Path(config[key])
        moved = source.with_name(source.name + "-physical")
        source.rename(moved)
        source.symlink_to(moved, target_is_directory=True)
    temporary = tmp_path / "local"
    temporary.mkdir()
    staged = job.stage_inputs(config, temporary)
    assert staged["run_dir"] == config["run_dir"]
    assert staged["data_dir"] == config["data_dir"]
    assert staged["reference_fasta"] == config["reference_fasta"]
    prepared = PreparedData(staged)
    assert prepared.manifest["identity"] == prepared_id
    assert staged["runtime"]["prepared_identity"] == prepared_id
    assert not Path(staged["prepared_dir"]).is_symlink()
    output = Path(config["run_dir"]) / "staged-predictions.csv"
    actual = predict(staged, best, output, prepared_data=prepared)
    pd.testing.assert_frame_equal(original, actual, check_exact=True)
    assert output.is_file() and not output.is_relative_to(temporary)
    assert staged["runtime"]["persistent_inputs"]["cache_dir"].endswith("cache-physical")
    batch_report = job.check_batch(staged, prepared)
    assert batch_report["effective_batch_size"] == config["train"]["batch_size"]
    assert batch_report["attempts"][0]["result"] == "passed"


def test_staging_rejects_corrupt_copy_and_capacity(synthetic_project, fake_alphagenome, tmp_path, monkeypatch):
    config = synthetic_project
    prepare(config)
    build_cache(config, device="cpu")
    temporary = tmp_path / "local"
    temporary.mkdir()
    real_run = job.subprocess.run
    def corrupt_copy(args, **kwargs):
        result = real_run(args, **kwargs)
        target = Path(args[-1])
        if target.name == "cache_dir":
            shard = next(target.glob("shard-*.npy"))
            with shard.open("r+b") as handle:
                handle.seek(-1, 2)
                handle.write(b"x")
        return result
    monkeypatch.setattr(job.subprocess, "run", corrupt_copy)
    with pytest.raises(ValueError, match="corrupted"):
        job.stage_inputs(config, temporary)
    monkeypatch.setattr(job.shutil, "disk_usage", lambda _: type("Disk", (), {"free": 1})())
    with pytest.raises(ValueError, match="Insufficient"):
        job.stage_inputs(config, temporary)


def test_batch_fallback_and_rng_restoration(synthetic_project, monkeypatch):
    config = copy.deepcopy(synthetic_project)
    config["train"]["batch_size"] = 256
    attempted = []
    random.seed(13)
    np.random.seed(13)
    torch.manual_seed(13)
    before_python = random.getstate()
    before_numpy = np.random.get_state()
    before_torch = torch.get_rng_state().clone()
    def probe(config, prepared, size, device):
        attempted.append(size)
        random.random()
        np.random.rand()
        torch.rand(4)
        if size > 64:
            raise torch.cuda.OutOfMemoryError("synthetic test")
        return {"batch_size": size, "examples_checked": size}
    monkeypatch.setattr(job, "_probe_batch", probe)
    result = job.check_batch(config, None)
    assert attempted == [256, 128, 64]
    assert result["effective_batch_size"] == 64
    assert random.getstate() == before_python
    assert np.random.get_state()[0] == before_numpy[0]
    np.testing.assert_array_equal(np.random.get_state()[1], before_numpy[1])
    assert np.random.get_state()[2:] == before_numpy[2:]
    torch.testing.assert_close(torch.get_rng_state(), before_torch, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="No candidate"):
        job.check_batch(config, None, allow_fallback=False)
    def other_failure(*args):
        raise ValueError("real configuration failure")
    monkeypatch.setattr(job, "_probe_batch", other_failure)
    with pytest.raises(ValueError, match="real configuration"):
        job.check_batch(config, None)


def test_failed_preflight_does_not_replace_results(synthetic_project, monkeypatch, tmp_path):
    config = synthetic_project
    root = tmp_path / "results"
    root.mkdir()
    marker = root / "old-results"
    marker.write_text("preserve")
    monkeypatch.setenv("SLURM_JOB_ID", "test")
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(job, "load_config", lambda _: config)
    def failed_stage(*args):
        raise ValueError("bad staged inputs")
    monkeypatch.setattr(job, "stage_inputs", failed_stage)
    with pytest.raises(ValueError, match="bad staged"):
        job.main(["--config", "unused", "--output-dir", str(root), "--overwrite"])
    assert marker.read_text() == "preserve"
    assert not list(tmp_path.glob("results.previous-*"))


def test_job_entrypoint_archives_only_after_checks_and_resumes_new_local_paths(synthetic_project, fake_alphagenome, monkeypatch, tmp_path):
    config = synthetic_project
    config["project_root"] = str(tmp_path)
    prepare(config)
    build_cache(config, device="cpu")
    root = tmp_path / "results"
    root.mkdir()
    (root / "old-results").write_text("previous run")
    monkeypatch.setattr(job, "load_config", lambda _: copy.deepcopy(config))
    monkeypatch.setenv("SLURM_JOB_ID", "synthetic-first")
    temporary = tmp_path / "job-one"
    temporary.mkdir()
    monkeypatch.setenv("TMPDIR", str(temporary))
    job.main(["--config", "unused", "--output-dir", str(root), "--overwrite", "--no-plots"])
    archives = list(tmp_path.glob("results.previous-*"))
    assert len(archives) == 1 and (archives[0] / "old-results").is_file()
    from rna_pred.common.config import read_json
    runtime = read_json(root / "runtime.json")
    assert runtime["batch_preflight"]["effective_batch_size"] == 2
    assert Path(runtime["staged_inputs"]["cache_dir"]).is_relative_to(temporary)
    assert (tmp_path / "logs" / "experiment-preflight-synthetic-first.json").is_file()
    metrics = pd.read_csv(root / "metrics.csv")
    monkeypatch.setenv("SLURM_JOB_ID", "synthetic-second")
    second_temporary = tmp_path / "job-two"
    second_temporary.mkdir()
    monkeypatch.setenv("TMPDIR", str(second_temporary))
    job.main(["--config", "unused", "--output-dir", str(root), "--resume", "--no-plots"])
    pd.testing.assert_frame_equal(metrics, pd.read_csv(root / "metrics.csv"), check_exact=True)
    assert len(list(tmp_path.glob("results.previous-*"))) == 1
    runtime = read_json(root / "runtime.json")
    assert Path(runtime["staged_inputs"]["cache_dir"]).is_relative_to(second_temporary)
    for name in ("a_to_b", "x1_to_x2", "x2_to_x1"):
        assert (root / name / "last.pt").is_file()
