"""Run and summarize the matched Smooth L1, weighted, and ranking comparison."""
import argparse
import ast
from pathlib import Path
import pandas as pd

from rna_pred.common.config import load_config

VARIANTS = ("baseline", "weighted", "ranking")


def summarize(root):
    root = Path(root)
    rows = []
    for variant in VARIANTS:
        folder = root / variant
        metrics = pd.read_csv(folder / "metrics.csv")
        for row in metrics[metrics.role.eq("validation")].itertuples():
            low = ast.literal_eval(row.low_expression) if isinstance(row.low_expression, str) else row.low_expression
            for cell, sub in low.items():
                rows.append({"design": row.experiment, "variant": variant, "cell": cell,
                             "combined_spearman": row.spearman,
                             "low_positive_spearman": sub["low_positive"]["spearman"],
                             "low_positive_mae_log1p": sub["low_positive"]["mae_log1p"],
                             "zero_positive_auroc": sub["zero_positive_auroc"],
                             "n_low_positive": sub["low_positive"]["n"],
                             "source": str(folder)})
    root.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "results.csv", index=False)
    return frame


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "summarize", "report"])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments/low_expression_losses"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-limited", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "run":
        from rna_pred.baseline.evaluation import run_experiments
        for variant in VARIANTS:
            config = load_config(Path(__file__).parent / "configs" / f"{variant}.yaml")
            run_experiments(config, args.output_dir / variant, resume=args.resume, overwrite=args.overwrite,
                            allow_limited=args.allow_limited, plots=False)
    summarize(args.output_dir)
    from .plots import plot_all
    print(plot_all(args.output_dir), flush=True)


if __name__ == "__main__":
    main()
