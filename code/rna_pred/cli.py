"""Baseline commands and explicitly selected experiment topics."""
import argparse
import importlib
from pathlib import Path

TOPICS = ("low_expression_losses", "chromatin_representation", "low_expression_features")


def _dispatch(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=Path(__file__).parent / "baseline/configs/default.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare", help="Build region and measured-track tables")
    prepare_parser.add_argument("--limit", type=int)
    prepare_parser.add_argument("--force", action="store_true")
    cache_parser = sub.add_parser("cache", help="Extract frozen AlphaGenome embeddings")
    cache_parser.add_argument("--device", default="cuda")
    cache_parser.add_argument("--limit-windows", type=int)
    cache_parser.add_argument("--verify-repeat", action="store_true")
    validate_parser = sub.add_parser("validate-cache", help="Verify coverage, checksums, and finite values")
    validate_parser.add_argument("--allow-limited", action="store_true")
    train_parser = sub.add_parser("train", help="Train the baseline model")
    train_parser.add_argument("--resume", action="store_true")
    for role in ("train", "val"):
        train_parser.add_argument(f"--{role}-cells", nargs="+", choices=["X1", "X2"], default=["X1", "X2"])
        train_parser.add_argument(f"--{role}-splits", nargs="+", choices=["train", "val"], default=[role])
    evaluation = sub.add_parser("evaluate", help="Fit the baseline under all three evaluation designs")
    evaluation.add_argument("--output-dir", type=Path)
    mode = evaluation.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    evaluation.add_argument("--no-plots", action="store_true")
    evaluation.add_argument("--allow-limited", action="store_true")
    report = sub.add_parser("report", help="Rebuild one baseline summary PDF from saved results")
    report.add_argument("--output-dir", type=Path, default=Path("outputs/baseline-evaluation"))
    prediction = sub.add_parser("predict", help="Export baseline X3 predictions")
    prediction.add_argument("--checkpoint", type=Path)
    prediction.add_argument("--output", type=Path)
    experiment = sub.add_parser("experiment", help="Run, summarize, or report an experiment topic")
    experiment.add_argument("topic", choices=TOPICS)
    experiment.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command == "experiment":
        module = importlib.import_module(f"rna_pred.experiments.{args.topic}.run")
        return module.main(args.arguments)
    if args.command == "report":
        from .baseline.plots import plot_all
        print(plot_all(args.output_dir), flush=True)
        return
    from .common.config import load_config
    config = load_config(args.config)
    if args.command == "prepare":
        from .common.data.prepare import prepare
        return prepare(config, args.limit, args.force)
    if args.command == "cache":
        from .common.data.cache import build_cache
        return build_cache(config, args.device, args.limit_windows, args.verify_repeat)
    if args.command == "validate-cache":
        from .common.data.cache import validate_cache
        return validate_cache(config, require_full=not args.allow_limited)
    if args.command == "train":
        from .baseline.workflow import train
        return train(config, args.resume, tuple(args.train_cells), tuple(args.val_cells),
                     tuple(args.train_splits), tuple(args.val_splits))
    if args.command == "evaluate":
        from .baseline.evaluation import run_experiments
        return run_experiments(config, args.output_dir, args.resume, not args.no_plots, args.allow_limited, args.overwrite)
    from .baseline.workflow import predict
    return predict(config, args.checkpoint, args.output)


def main(argv=None):
    _dispatch(argv)


if __name__ == "__main__":
    main()
