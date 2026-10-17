"""Additional features: exploration, source screens, neural diagnostic, and summary."""
import argparse
from pathlib import Path

from rna_pred.common.config import load_config, read_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["extract", "explore", "screen", "activity-screen", "bundles", "train", "summarize", "report"])
    parser.add_argument("--config", default=Path(__file__).parent / "configs/default.yaml")
    parser.add_argument("--root", type=Path, default=Path("outputs/experiments/low_expression_features"))
    parser.add_argument("--chromatin-dir", type=Path, default=Path("outputs/experiments/chromatin_representation"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "report":
        from .plots import plot_all
        print(plot_all(root), flush=True)
        return
    config = load_config(args.config)
    previous = args.chromatin_dir.resolve()
    if args.command in {"screen", "activity-screen"} and (root / "training_submission.json").exists():
        raise ValueError("Training selection is frozen; use a new investigation root for new screens")
    if args.command == "extract":
        import fcntl
        from .features import prepare_extra
        root.mkdir(parents=True, exist_ok=True)
        with (root / ".extract.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            prepare_extra(config, root, previous / "features")
    elif args.command == "explore":
        from .exploration import explore
        explore(config, root, previous / "features", previous)
    elif args.command in {"screen", "activity-screen"}:
        import fcntl
        from .screens import run_screens
        from .activity_screen import run
        with (root / ".screen.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            (run_screens if args.command == "screen" else run)(config, root, previous / "features")
    elif args.command == "bundles":
        from .training import materialize_bundles
        materialize_bundles(config, root, previous / "features")
    elif args.command == "train":
        frozen = Path(args.config).resolve().parent / "selection.json"
        if frozen.exists() and read_json(frozen) != read_json(root / "selection.json"):
            raise ValueError("Feature selection changed after training source freeze")
        from .training import train_selected
        train_selected(config, root, args.seed, args.stage)
    elif args.command == "summarize":
        from .results import summarize
        summarize(config, root)
    if args.command in {"explore", "screen", "activity-screen", "summarize"}:
        from .plots import plot_all
        print(plot_all(root), flush=True)


if __name__ == "__main__":
    main()
