import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.config import PATHS
from src.evaluation.evaluate import evaluate
from src.lstm.pipeline import LSTMTrainingPipeline


LABEL_COLUMN = "Spam/Ham"


def stratified_cumulative_splits(train_csv: Path, num_portions: int) -> List[Tuple[int, pd.DataFrame]]:
    df = pd.read_csv(train_csv)
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Missing '{LABEL_COLUMN}' column in {train_csv}")

    groups = {label: g.copy() for label, g in df.groupby(LABEL_COLUMN, sort=False)}
    splits = []

    for i in range(1, num_portions + 1):
        parts = []
        for _, g in groups.items():
            end = int(len(g) * i / num_portions)
            parts.append(g.iloc[:end])
        split_df = pd.concat(parts).sort_index()
        splits.append((len(split_df), split_df))

    return splits


def plot_curve(curve: List[Tuple[int, float]], out_path: Path) -> None:
    sizes = [x for x, _ in curve]
    f1s = [y for _, y in curve]

    plt.figure(figsize=(7, 4))
    plt.plot(sizes, f1s, marker="o")
    plt.title("Learning Curve (F1 vs Corpus Size)")
    plt.xlabel("Corpus Size")
    plt.ylabel("F1")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Learning curve (F1 only)")
    parser.add_argument("--num-portions", type=int, default=10)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    if args.num_portions < 1:
        raise ValueError("--num-portions must be >= 1")
    if not args.train and not args.eval:
        raise ValueError("Use --train and/or --eval")

    split_dir = PATHS.corpora_processed / "learning_curve"
    model_root = PATHS.models / "learning_curve"
    split_dir.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)

    splits = stratified_cumulative_splits(PATHS.train_processed, args.num_portions)

    # Always derive run dirs deterministically so train/eval can be run separately.
    run_info = []
    for i, (size, split_df) in enumerate(splits, start=1):
        split_csv = split_dir / f"train_portion_{i:02d}_of_{args.num_portions}.csv"
        run_dir = model_root / f"portion_{i:02d}_of_{args.num_portions}"
        run_dir.mkdir(parents=True, exist_ok=True)

        if args.train:
            split_df.to_csv(split_csv, index=False)
            print(f"[train] portion {i}/{args.num_portions}, size={size}")

            p = LSTMTrainingPipeline(verbose=True)
            p.train_csv = split_csv
            p.val_csv = PATHS.validation_processed
            p.save_dir = run_dir
            p.run()

        run_info.append((size, run_dir / "best_model.pt"))

    if args.eval:
        curve = []
        print("Learning curve (F1 only):")
        for size, ckpt in run_info:
            if not ckpt.exists():
                print(f"  size={size}: missing checkpoint ({ckpt})")
                continue

            metrics = evaluate(
                checkpoint_path=ckpt,
                tokenizer_path=PATHS.lstm_tokenizer,
                eval_csv_path=PATHS.test_processed,
                batch_size=64,
                threshold=0.5,
            )
            f1 = float(metrics["f1"])
            curve.append((size, f1))
            print(f"  size={size}, f1={f1:.4f}")

        if curve:
            plot_path = model_root / "learning_curve_f1.png"
            plot_curve(curve, plot_path)
            print(f"[eval] saved plot: {plot_path}")


if __name__ == "__main__":
    main()