import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Tuple

from src.config.config import PATHS
from src.evaluation.evaluate import evaluate
from src.lstm.pipeline import LSTMTrainingPipeline

REQUIRED_HYPERPARAMETERS = {
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


class RandomSearchTuner:
    def __init__(self, config_path: Path, output_path: Path):
        self.config_path = Path(config_path)
        self.output_path = Path(output_path)
        self.config = self._load_json(self.config_path)
        self.seed = int(self.config.get("random_seed", 42))
        self.metric_name = str(self.config.get("metric", "f1"))
        self.default_trials = int(self.config.get("num_trials", 10))
        self.search_space = self.config["search_space"]
        random.seed(self.seed)

    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _sample_param(self, spec: Dict[str, Any]) -> Any:
        param_type = spec["type"]
        if param_type == "choice":
            return random.choice(spec["values"])
        if param_type == "int":
            return random.randint(int(spec["min"]), int(spec["max"]))
        if param_type == "float":
            return random.uniform(float(spec["min"]), float(spec["max"]))
        if param_type == "log_float":
            low = math.log(float(spec["min"]))
            high = math.log(float(spec["max"]))
            return math.exp(random.uniform(low, high))
        raise ValueError(f"Unsupported parameter type: {param_type}")

    def _sample_hyperparameters(self) -> Dict[str, Any]:
        sampled: Dict[str, Any] = {}
        for name, spec in self.search_space.items():
            sampled[name] = self._sample_param(spec)
        missing = sorted(REQUIRED_HYPERPARAMETERS - set(sampled.keys()))
        if missing:
            raise ValueError(
                "Missing required hyperparameters in tuning config: "
                + ", ".join(missing)
            )
        return sampled

    def _run_trial(self, trial_idx: int, hyperparams: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        trial_dir = PATHS.models / "tuning" / f"trial_{trial_idx:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        pipeline = LSTMTrainingPipeline(**hyperparams)
        pipeline.train_csv = PATHS.train_processed
        pipeline.val_csv = PATHS.validation_processed
        pipeline.save_dir = trial_dir
        history = pipeline.run()

        checkpoint_path = trial_dir / "best_model.pt"
        metrics = evaluate(
            checkpoint_path=checkpoint_path,
            tokenizer_path=PATHS.lstm_tokenizer,
            eval_csv_path=PATHS.test_processed,
            batch_size=64,
            threshold=0.5,
        )
        score = float(metrics[self.metric_name])
        details = {
            "trial_index": trial_idx,
            "trial_dir": str(trial_dir),
            "hyperparameters": hyperparams,
            "training_history": history,
            "metrics": metrics,
            "score": score,
            "score_metric": self.metric_name,
        }
        return score, details

    def run(self, num_trials: int | None = None) -> Dict[str, Any]:
        trials = self.default_trials if num_trials is None else int(num_trials)
        all_results = []
        best_result = None
        best_score = float("-inf")

        for trial_idx in range(1, trials + 1):
            hyperparams = self._sample_hyperparameters()
            score, details = self._run_trial(trial_idx, hyperparams)
            all_results.append(details)
            if score > best_score:
                best_score = score
                best_result = details
            params_str = ", ".join(f"{k}={v}" for k, v in hyperparams.items())
            print(
                f"[tune] trial={trial_idx}/{trials}, "
                f"{self.metric_name}={score:.4f}, params: {params_str}"
            )

        if best_result is None:
            raise RuntimeError("No trials were executed.")

        output = {
            "config_path": str(self.config_path),
            "metric": self.metric_name,
            "num_trials": trials,
            "best_trial_index": best_result["trial_index"],
            "best_model_dir": best_result["trial_dir"],
            "best_model_checkpoint": str(Path(best_result["trial_dir"]) / "best_model.pt"),
            "best_hyperparameters": best_result["hyperparameters"],
            "best_metrics": best_result["metrics"],
            "all_trials": all_results,
        }

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

        return output
