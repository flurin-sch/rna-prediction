"""Reports read saved results and produce one PDF, without extraction or model loading."""
import re
from pathlib import Path

import pandas as pd
import pytest

from rna_pred.cli import main
from rna_pred.common.config import load_config
from rna_pred.experiments.low_expression_losses.run import summarize
from rna_pred.experiments.low_expression_losses.plots import plot_all as loss_report
from rna_pred.experiments.chromatin_representation.plots import plot_all as chromatin_report
from rna_pred.experiments.low_expression_features.plots import plot_all as feature_report


def assert_pdf(path, pages):
    data = path.read_bytes()
    assert data.startswith(b"%PDF-")
    assert len(re.findall(rb"/Type /Page\b", data)) == pages
    assert not list(path.parent.rglob("*_legend.md"))


def test_moved_configs_resolve_the_same_project_root():
    project = Path(__file__).parents[1]
    configs = sorted((project / "code/rna_pred").rglob("*.yaml"))
    assert len(configs) == 7
    for path in configs:
        config = load_config(path)
        assert Path(config["project_root"]) == project
        assert Path(config["data_dir"]) == (project / "data/ml-4-g-project-1").resolve()


def test_loss_summary_keeps_undefined_metrics_and_writes_one_pdf(tmp_path):
    low = {"X1": {"low_positive": {"n": 3, "spearman": None, "mae_log1p": .5}, "zero_positive_auroc": .8}}
    for variant, score in [("baseline", .8), ("weighted", .79), ("ranking", .81)]:
        folder = tmp_path / variant
        folder.mkdir()
        pd.DataFrame([{"role": "validation", "experiment": "a_to_b", "spearman": score,
                       "low_expression": str(low)}]).to_csv(folder / "metrics.csv", index=False)
    frame = summarize(tmp_path)
    assert frame.low_positive_spearman.isna().all()
    assert_pdf(loss_report(tmp_path), 4)
    # Repeating replaces the report instead of appending duplicate pages.
    assert_pdf(loss_report(tmp_path), 4)


def test_chromatin_report_from_results_only(tmp_path):
    from rna_pred.common.config import atomic_json
    low = {"X1": {"low_positive": {"spearman": None}, "zero_low_positive_auroc": .7}}
    rows = [{"role": "validation", "design": "a_to_b", "variant": name, "seed": 42,
             "spearman_per_cell": {"X1": .8}, "low_expression": low}
            for name in ("baseline", "dual_branch")]
    atomic_json(tmp_path / "results.json", rows)
    assert_pdf(chromatin_report(tmp_path), 4)


def test_feature_report_supports_negative_screens_without_neural_runs(tmp_path):
    screens = tmp_path / "screens"
    screens.mkdir()
    pd.DataFrame({"condition": ["baseline", "body"], "composite": [0., -.002],
                  "promising": [False, False]}).to_csv(screens / "ranking.csv", index=False)
    assert_pdf(feature_report(tmp_path), 1)


def test_cli_reports_bypass_training_config_and_return_success(tmp_path, monkeypatch):
    from rna_pred.baseline import plots
    calls = []
    monkeypatch.setattr(plots, "plot_all", lambda root: calls.append(root) or root / "summary.pdf")
    assert main(["--config", "missing.yaml", "report", "--output-dir", str(tmp_path)]) is None
    assert calls == [tmp_path]
    from rna_pred.experiments.low_expression_features import run
    monkeypatch.setattr(run, "main", lambda arguments: calls.append(arguments))
    main(["experiment", "low_expression_features", "report", "--root", str(tmp_path)])
    assert calls[-1] == ["report", "--root", str(tmp_path)]


def test_negative_feature_selection_summarizes_without_training_data(tmp_path):
    from rna_pred.common.config import atomic_json, read_json
    from rna_pred.experiments.low_expression_features.results import summarize as summarize_features
    atomic_json(tmp_path / "selection.json", {"selected": []})
    summarize_features({}, tmp_path)
    assert read_json(tmp_path / "complete.json") == {"fits": 0, "selected": [], "stage": "negative_screens"}
    assert "No bundle passed" in (tmp_path / "results_report.md").read_text()
