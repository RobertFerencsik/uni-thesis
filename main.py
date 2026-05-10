import argparse
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.learning_curve import LearningCurveRunner
from src.experiments.tuning import RandomSearchTuner
from src.infrastructure.artifact_manager import ArtifactManager

DEFAULT_SEARCH_CONFIG = REPO_ROOT / "src" / "config" / "hyperparameter_search_space.json"
DEFAULT_BEST_HPARAMS = REPO_ROOT / "src" / "config" / "best_hyperparameters.json"


def load_best_hyperparameters(
    path: Path, artifacts: Optional[ArtifactManager] = None
):
    am = artifacts or ArtifactManager()
    data = am.load_json(path)
    return data["best_hyperparameters"]


def run_tuning(num_trials: int):
    artifacts = ArtifactManager()
    tuner = RandomSearchTuner(
        config_path=DEFAULT_SEARCH_CONFIG,
        output_path=DEFAULT_BEST_HPARAMS,
        num_trials=num_trials,
        artifact_manager=artifacts,
    )
    result = tuner.run()
    metric = result["metric"]
    score = result["best_metrics"][metric]
    print(f"[tune] best {metric}={score:.4f}, saved: {DEFAULT_BEST_HPARAMS}")


def run_learning_curve(num_portions: int):
    artifacts = ArtifactManager()
    hparams = load_best_hyperparameters(DEFAULT_BEST_HPARAMS, artifacts)
    LearningCurveRunner(
        num_portions=num_portions,
        pipeline_hparams=hparams,
        artifact_manager=artifacts,
    ).run()


def main():
    parser = argparse.ArgumentParser(
        description="Tune LSTM hyperparameters or run learning curve"
    )
    parser.add_argument("mode", choices=["tune", "learning-curve"], help="Mode to run")
    parser.add_argument(
        "--num-trials",
        type=int,
        default=20,
        help="Number of random-search trials (mode: tune)",
    )
    parser.add_argument(
        "--num-portions",
        type=int,
        default=10,
        help="Number of cumulative stratified train portions (mode: learning-curve)",
    )
    args = parser.parse_args()

    if args.mode == "tune":
        run_tuning(args.num_trials)
    elif args.mode == "learning-curve":
        run_learning_curve(args.num_portions)


if __name__ == "__main__":
    main()
