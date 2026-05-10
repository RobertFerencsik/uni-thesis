import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.lstm.learning_curve_runner import LearningCurveRunner
from src.lstm.tune import RandomSearchTuner

DEFAULT_SEARCH_CONFIG = REPO_ROOT / "src" / "config" / "hyperparameter_search_space.json"
DEFAULT_BEST_HPARAMS = REPO_ROOT / "src" / "config" / "best_hyperparameters.json"

def load_best_hyperparameters(path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    loaded = data["best_hyperparameters"]
    return loaded


def run_tuning(num_trials):
    tuner = RandomSearchTuner(
        config_path=DEFAULT_SEARCH_CONFIG,
        output_path=DEFAULT_BEST_HPARAMS,
        num_trials=num_trials
    )
    result = tuner.run()
    metric = result["metric"]
    score = result["best_metrics"][metric]
    print(f"[tune] best {metric}={score:.4f}, saved: {DEFAULT_BEST_HPARAMS}")


def run_learning_curve(num_portions):
    hparams = load_best_hyperparameters(DEFAULT_BEST_HPARAMS)
    LearningCurveRunner(
        num_portions=num_portions,
        pipeline_hparams=hparams
    ).run()


def main():
    args = {}
    parser = argparse.ArgumentParser(description="Tune LSTM hyperparameters or run learning curve")
    parser.add_argument("mode", choices=["tune", "learning-curve"], help="Mode to run")
    args.update(vars(parser.parse_args()))

    if args["mode"] == "learning-curve":
        parser.add_argument(
            "--num-portions",
            type=int,
            default=10,
            help="Number of cumulative stratified train portions",
        )   
    elif args["mode"] == "tune":
        parser.add_argument(
            "--num-trials",
            type=int,
            default=20,
            help="Trials when using --tune",
        )
    else:
        parser.error("Invalid mode. Please use 'tune' or 'learning-curve'.")
    args.update(vars(parser.parse_args()))

    if args["mode"] == "tune":
        run_tuning(args["num_trials"])
    elif args["mode"] == "learning-curve":
        run_learning_curve(args["num_portions"])

if __name__ == "__main__":
    main()