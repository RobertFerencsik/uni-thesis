from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt
import pandas as pd

from src.config.config import PATHS
from src.evaluation.evaluate import evaluate
from src.lstm.pipeline import LSTMTrainingPipeline

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


def plot_f1_curve(curve, out_path):
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


class LearningCurveRunner:
    def __init__(
        self,
        num_portions,
        do_train,
        do_eval,
        pipeline_hparams,
        *,
        seed = SPLIT_SEED,
        eval_threshold = EVAL_THRESHOLD,
    ):
        self.num_portions = num_portions
        self.do_train = do_train
        self.do_eval = do_eval
        self.pipeline_hparams = pipeline_hparams
        self.seed = int(seed)
        self.eval_threshold = float(eval_threshold)

        self.split_dir = PATHS.corpora_processed / "learning_curve"
        self.model_root = PATHS.models / "learning_curve"

    def _validate(self):
        if self.num_portions < 1:
            raise ValueError("--num-portions must be >= 1")
        if not self.do_train and not self.do_eval:
            raise ValueError("Use --train and/or --eval")

    def _ensure_dirs(self):
        self.split_dir.mkdir(parents=True, exist_ok=True)
        self.model_root.mkdir(parents=True, exist_ok=True)

    def _train_portion(self, portion_index, size, split_df):
        split_csv = self.split_dir / f"train_portion_{portion_index:02d}_of_{self.num_portions}.csv"
        run_dir = self.model_root / f"portion_{portion_index:02d}_of_{self.num_portions}"

        split_df.to_csv(split_csv, index=False)
        class_counts = split_df[LABEL_COLUMN].value_counts().to_dict()
        print(
            f"[train] portion {portion_index}/{self.num_portions}, "
            f"size={size}, class_counts={class_counts}"
        )

        pipeline = LSTMTrainingPipeline(**self.pipeline_hparams)
        pipeline.train_csv = split_csv
        pipeline.val_csv = PATHS.validation_processed
        pipeline.save_dir = run_dir
        pipeline.run()

    def _build_run_info(self):
        corpus = pd.read_csv(PATHS.train_processed)
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
            run_dir.mkdir(parents=True, exist_ok=True)
            if self.do_train:
                self._train_portion(i, size, split_df)
            run_info.append((size, run_dir / "best_model.pt"))
        return run_info

    def _evaluate(self, run_info):
        curve = []
        print(f"Learning curve (F1, threshold={self.eval_threshold}):")
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
            )
            f1 = float(metrics["f1"])
            curve.append((size, f1))
            print(f"  size={size}, f1={f1:.4f}")

        if curve:
            plot_path = self.model_root / "learning_curve_f1.png"
            plot_f1_curve(curve, plot_path)
            print(f"[eval] saved plot: {plot_path}")

    def run(self):
        self._validate()
        self._ensure_dirs()
        run_info = self._build_run_info()
        if self.do_eval:
            self._evaluate(run_info)
