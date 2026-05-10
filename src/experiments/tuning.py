import math
import random
from pathlib import Path
from typing import Any, Dict, Tuple

from src.evaluation.evaluate import evaluate
from src.evaluation.reporting import (
    print_full_experiment_report,
    print_section,
    save_confusion_matrix_figure,
    save_training_history_figure,
)
from src.infrastructure.artifact_manager import ArtifactManager
from src.infrastructure.paths import PATHS
from src.training.pipeline import LSTMTrainingPipeline


class RandomSearchTuner:
    def __init__(
        self,
        config_path: Path,
        output_path: Path,
        num_trials: int,
        *,
        artifact_manager: ArtifactManager | None = None,
    ):
        self._artifacts = artifact_manager or ArtifactManager()
        self.config_path = Path(config_path)
        self.output_path = Path(output_path)
        self.config = self._artifacts.load_json(self.config_path)
        self.seed = 42
        self.metric_name = str(self.config.get("metric", "f1"))
        self.num_trials = num_trials
        self.search_space = self.config["search_space"]
        random.seed(self.seed)

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
        return sampled

    def _run_trial(self, trial_idx: int, hyperparams: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        trial_dir = PATHS.models / "tuning" / f"trial_{trial_idx:03d}"
        self._artifacts.make_dir(trial_dir)

        pipeline = LSTMTrainingPipeline(**hyperparams, artifact_manager=self._artifacts)
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
            artifact_manager=self._artifacts,
            include_model_info=True,
        )
        score = float(metrics[self.metric_name])

        print_section(f"Tuning trial {trial_idx:03d} / {self.num_trials} ({self.metric_name}={score:.4f})")
        print_full_experiment_report(metrics, metrics.get("model_info"))
        save_training_history_figure(
            history,
            trial_dir / "training_curves.png",
            self._artifacts,
            title=f"Trial {trial_idx:03d}",
        )
        save_confusion_matrix_figure(
            metrics,
            trial_dir / "test_confusion_matrix.png",
            self._artifacts,
            title=f"Trial {trial_idx:03d} - test set",
        )
        trial_result = {
            "trial_index": trial_idx,
            "trial_dir": str(trial_dir),
            "hyperparameters": hyperparams,
            "metrics": metrics,
            "score": score,
            "score_metric": self.metric_name,
        }
        self._artifacts.save_json(
            trial_result,
            trial_dir / "eval_test_metrics.json",
        )
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

    def run(self) -> Dict[str, Any]:
        all_results = []
        best_result = None
        best_score = float("-inf")

        for trial_idx in range(1, self.num_trials + 1):
            hyperparams = self._sample_hyperparameters()
            score, details = self._run_trial(trial_idx, hyperparams)
            all_results.append(details)
            if score > best_score:
                best_score = score
                best_result = details
            params_str = ", ".join(f"{k}={v}" for k, v in hyperparams.items())
            print(
                f"[tune] trial={trial_idx}/{self.num_trials}, "
                f"{self.metric_name}={score:.4f}, params: {params_str}"
            )

        if best_result is None:
            raise RuntimeError("No trials were executed.")

        output = {
            "config_path": str(self.config_path),
            "metric": self.metric_name,
            "num_trials": self.num_trials,
            "best_trial_index": best_result["trial_index"],
            "best_model_dir": best_result["trial_dir"],
            "best_model_checkpoint": str(Path(best_result["trial_dir"]) / "best_model.pt"),
            "best_hyperparameters": best_result["hyperparameters"],
            "best_metrics": best_result["metrics"],
            "all_trials": all_results,
        }

        self._artifacts.save_json(output, self.output_path)

        print_section("Best tuning trial (test-set summary)")
        bm = best_result["metrics"]
        print_full_experiment_report(bm, bm.get("model_info"))
        print(
            f"  trial directory: {best_result['trial_dir']}\n"
            f"  checkpoint: {output['best_model_checkpoint']}\n"
            f"  JSON summary: {self.output_path}"
        )

        return output
