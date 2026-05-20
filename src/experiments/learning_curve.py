from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt
import pandas as pd

from src.evaluation.evaluate import evaluate
from src.evaluation.reporting import (
    print_full_experiment_report,
    print_section,
    save_confusion_matrix_figure,
    save_learning_curve_metrics_figure,
    save_training_history_figure,
)
from src.infrastructure.artifact_manager import ArtifactManager
from src.infrastructure.paths import PATHS
from src.training.pipeline import LSTMTrainingPipeline

LABEL_COLUMN = "Spam/Ham"
SPLIT_SEED = 42
EVAL_BATCH_SIZE = 64
EVAL_THRESHOLD = 0.5


def stratified_cumulative_splits(corpus, target_col, n_splits=10, random_state=42):
    X = corpus.drop(columns=[target_col])
    y = corpus[target_col]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    splits = [corpus.iloc[test_idx] for _, test_idx in skf.split(X, y)]

    return [
        pd.concat(splits[:i]).reset_index(drop=True)
        for i in range(1, n_splits + 1)
    ]


def plot_f1_curve(curve, out_path, artifact_manager: ArtifactManager):
    sizes = [x for x, _ in curve]
    f1s = [y for _, y in curve]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sizes, f1s, marker="o")
    ax.set_title("Learning Curve (F1 vs Corpus Size)")
    ax.set_xlabel("Corpus Size")
    ax.set_ylabel("F1")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    artifact_manager.save_figure(fig, out_path, dpi=300)
    plt.close(fig)


class LearningCurveRunner:
    def __init__(
        self,
        num_portions,
        pipeline_hparams,
        *,
        seed=SPLIT_SEED,
        eval_threshold=EVAL_THRESHOLD,
        artifact_manager: ArtifactManager | None = None,
    ):
        self._artifacts = artifact_manager or ArtifactManager()
        self.num_portions = num_portions
        self.pipeline_hparams = pipeline_hparams
        self.seed = int(seed)
        self.eval_threshold = float(eval_threshold)
        self.split_dir = PATHS.corpora_processed / "learning_curve"
        self.model_root = PATHS.models / "learning_curve"

        self._artifacts.make_dir(self.split_dir)
        self._artifacts.make_dir(self.model_root)

    def _train_portion(self, portion_index, size, split_df):
        split_csv = self.split_dir / f"train_portion_{portion_index:02d}_of_{self.num_portions}.csv"
        run_dir = self.model_root / f"portion_{portion_index:02d}_of_{self.num_portions}"

        self._artifacts.save_csv(split_df, split_csv, index=False)
        class_counts = split_df[LABEL_COLUMN].value_counts().to_dict()
        print(
            f"[train] portion {portion_index}/{self.num_portions}, "
            f"size={size}, class_counts={class_counts}"
        )

        pipeline = LSTMTrainingPipeline(
            **self.pipeline_hparams,
            artifact_manager=self._artifacts,
        )
        pipeline.train_csv = split_csv
        pipeline.val_csv = PATHS.validation_processed
        pipeline.save_dir = run_dir
        history = pipeline.run()
        save_training_history_figure(
            history,
            run_dir / "training_curves.png",
            self._artifacts,
            title=f"Portion {portion_index}/{self.num_portions}",
        )

    def _build_run_info(self):
        corpus = self._artifacts.load_csv(PATHS.train_processed)
        splits = stratified_cumulative_splits(
            corpus,
            LABEL_COLUMN,
            n_splits=self.num_portions,
            random_state=self.seed,
        )
        run_info = []
        for i, split_df in enumerate(splits, start=1):
            size = len(split_df)
            run_dir = self.model_root / f"portion_{i:02d}_of_{self.num_portions}"
            self._artifacts.make_dir(run_dir)
            self._train_portion(i, size, split_df)
            run_info.append((size, run_dir / "best_model.pt"))
        return run_info

    def _evaluate(self, run_info):
        curve_f1 = []
        curve_prf = []
        print(f"Learning curve (metrics, threshold={self.eval_threshold}):")
        for size, ckpt in run_info:
            if not ckpt.exists():
                print(f"  size={size}: missing checkpoint ({ckpt})")
                continue

            metrics = evaluate(
                checkpoint_path=ckpt,
                tokenizer_path=PATHS.lstm_tokenizer,
                eval_csv_path=PATHS.test_processed,
                batch_size=EVAL_BATCH_SIZE,
                threshold=self.eval_threshold,
                artifact_manager=self._artifacts,
                include_model_info=True,
            )
            print_section(f"Learning curve | train corpus size = {size}")
            print_full_experiment_report(metrics, metrics.get("model_info"))

            out = ckpt.parent / "eval_test_metrics.json"
            self._artifacts.save_json({"corpus_size": size, **metrics}, out)

            save_confusion_matrix_figure(
                metrics,
                ckpt.parent / "test_confusion_matrix.png",
                self._artifacts,
                title=f"Test set - corpus size {size}",
            )

            f1 = float(metrics["f1"])
            curve_f1.append((size, f1))
            curve_prf.append(
                (
                    size,
                    float(metrics["precision"]),
                    float(metrics["recall"]),
                    f1,
                )
            )
            print(
                f"  [summary] size={size}  P={metrics['precision']:.4f}  "
                f"R={metrics['recall']:.4f}  F1={f1:.4f}"
            )

        if curve_f1:
            plot_path_f1 = self.model_root / "learning_curve_f1.png"
            plot_f1_curve(curve_f1, plot_path_f1, self._artifacts)
            print(f"[eval] saved plot: {plot_path_f1}")
        if curve_prf:
            plot_path_prf = self.model_root / "learning_curve_prf.png"
            save_learning_curve_metrics_figure(curve_prf, plot_path_prf, self._artifacts)
            print(f"[eval] saved plot: {plot_path_prf}")

    def run(self):
        run_info = self._build_run_info()
        self._evaluate(run_info)
