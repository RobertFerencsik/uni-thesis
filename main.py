import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.config import PATHS
from src.evaluation.evaluate import evaluate
from src.lstm.pipeline import LSTMTrainingPipeline
from src.lstm.tune import RandomSearchTuner


LABEL_COLUMN = "Spam/Ham"
DEFAULT_SEARCH_CONFIG = REPO_ROOT / "src" / "config" / "hyperparameter_search_space.json"
DEFAULT_BEST_HPARAMS = REPO_ROOT / "src" / "config" / "best_hyperparameters.json"


def _load_best_hyperparameters(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Best hyperparameters file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "best_hyperparameters" not in data:
        raise ValueError(f"Missing 'best_hyperparameters' in {path}")
    loaded = data["best_hyperparameters"]
    required_keys = {
        "max_length",
        "batch_size",
        "embedding_dim",
        "hidden_size",
        "num_layers",
        "dropout_rate",
        "dense_hidden",
        "learning_rate",
        "max_grad_norm",
        "num_epochs",
    }
    missing = sorted(required_keys - set(loaded.keys()))
    if missing:
        raise ValueError(
            f"Missing required hyperparameter keys in {path}: {', '.join(missing)}"
        )
    return loaded


class LearningCurveRunner:
    def __init__(
        self,
        num_portions: int,
        do_train: bool,
        do_eval: bool,
        pipeline_hparams: Dict[str, Any],
    ):
        self.num_portions = num_portions
        self.do_train = do_train
        self.do_eval = do_eval
        self.pipeline_hparams = pipeline_hparams
        self.split_dir = PATHS.corpora_processed / "learning_curve"
        self.model_root = PATHS.models / "learning_curve"

    def _validate_args(self) -> None:
        if self.num_portions < 1:
            raise ValueError("--num-portions must be >= 1")
        if not self.do_train and not self.do_eval:
            raise ValueError("Use --train and/or --eval")

    def _prepare_dirs(self) -> None:
        self.split_dir.mkdir(parents=True, exist_ok=True)
        self.model_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _stratified_cumulative_splits(
        train_csv: Path, num_portions: int
    ) -> List[Tuple[int, pd.DataFrame]]:
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

    @staticmethod
    def _plot_curve(curve: List[Tuple[int, float]], out_path: Path) -> None:
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

    def _train_portion(self, i: int, size: int, split_df: pd.DataFrame) -> None:
        split_csv = self.split_dir / f"train_portion_{i:02d}_of_{self.num_portions}.csv"
        run_dir = self.model_root / f"portion_{i:02d}_of_{self.num_portions}"

        split_df.to_csv(split_csv, index=False)
        print(f"[train] portion {i}/{self.num_portions}, size={size}")

        pipeline = LSTMTrainingPipeline(**self.pipeline_hparams)
        pipeline.train_csv = split_csv
        pipeline.val_csv = PATHS.validation_processed
        pipeline.save_dir = run_dir
        pipeline.run()

    def _build_run_info(self) -> List[Tuple[int, Path]]:
        splits = self._stratified_cumulative_splits(PATHS.train_processed, self.num_portions)

        run_info = []
        for i, (size, split_df) in enumerate(splits, start=1):
            run_dir = self.model_root / f"portion_{i:02d}_of_{self.num_portions}"
            run_dir.mkdir(parents=True, exist_ok=True)

            if self.do_train:
                self._train_portion(i, size, split_df)

            run_info.append((size, run_dir / "best_model.pt"))
        return run_info

    def _evaluate(self, run_info: List[Tuple[int, Path]]) -> None:
        curve: List[Tuple[int, float]] = []
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
            plot_path = self.model_root / "learning_curve_f1.png"
            self._plot_curve(curve, plot_path)
            print(f"[eval] saved plot: {plot_path}")

    def run(self) -> None:
        self._validate_args()
        self._prepare_dirs()
        run_info = self._build_run_info()
        if self.do_eval:
            self._evaluate(run_info)


def main():
    parser = argparse.ArgumentParser(description="Learning curve (F1 only)")
    parser.add_argument("--num-portions", type=int, default=10)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--num-trials", type=int, default=10)
    args = parser.parse_args()

    if args.tune:
        tuner = RandomSearchTuner(
            config_path=DEFAULT_SEARCH_CONFIG,
            output_path=DEFAULT_BEST_HPARAMS,
        )
        result = tuner.run(num_trials=args.num_trials)
        print(
            f"[tune] best {result['metric']}={result['best_metrics'][result['metric']]:.4f}, "
            f"saved: {DEFAULT_BEST_HPARAMS}"
        )
        return

    pipeline_hparams = _load_best_hyperparameters(DEFAULT_BEST_HPARAMS)

    runner = LearningCurveRunner(
        num_portions=args.num_portions,
        do_train=args.train,
        do_eval=args.eval,
        pipeline_hparams=pipeline_hparams,
    )
    runner.run()


if __name__ == "__main__":
    main()