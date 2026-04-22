"""
training/mlflow_tracker.py — MLflow experiment tracking integration.

Wraps SFT and DPO training runs with full MLflow tracking:
  - Hyperparameters (LoRA rank, alpha, lr, batch size, epochs)
  - Metrics (train loss, eval loss, quality gate scores)
  - Artifacts (model checkpoints, dataset cards, evaluation reports)
  - Dataset lineage (which batch_id → which model version)

Usage:
    from training.mlflow_tracker import FoundryMLflowTracker

    tracker = FoundryMLflowTracker(experiment_name="foundry-sft")
    with tracker.start_run(run_name="csrd-sft-v3", batch_id="batch_001"):
        tracker.log_hyperparams(lora_rank=16, lr=2e-4, epochs=3)
        # ... training loop ...
        tracker.log_metric("train_loss", loss, step=step)
        tracker.log_quality_gate(gate_result)
        tracker.log_model_artifact(output_dir)
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

logger = logging.getLogger("foundry.mlflow")

_MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
_MLFLOW_EXPERIMENT_BASE = os.getenv("MLFLOW_EXPERIMENT", "foundry")


class FoundryMLflowTracker:
    """
    MLflow tracking wrapper with graceful degradation when MLflow is unavailable.
    Logs all training metadata to enable reproducibility and experiment comparison.
    """

    def __init__(self, experiment_name: str = _MLFLOW_EXPERIMENT_BASE) -> None:
        self.experiment_name = experiment_name
        self._mlflow = None
        self._run = None
        self._available = self._init_mlflow()

    def _init_mlflow(self) -> bool:
        try:
            import mlflow  # type: ignore
            mlflow.set_tracking_uri(_MLFLOW_TRACKING_URI)
            mlflow.set_experiment(self.experiment_name)
            self._mlflow = mlflow
            logger.info("MLflow connected: %s experiment=%s", _MLFLOW_TRACKING_URI, self.experiment_name)
            return True
        except ImportError:
            logger.warning("mlflow not installed — tracking disabled. pip install mlflow")
            return False
        except Exception as exc:
            logger.warning("MLflow unavailable (%s) — tracking disabled.", exc)
            return False

    @contextmanager
    def start_run(
        self,
        run_name: str,
        batch_id: str,
        tags: Optional[Dict[str, str]] = None,
    ) -> Generator[None, None, None]:
        """Context manager for a single training run."""
        if not self._available or self._mlflow is None:
            yield
            return

        run_tags = {
            "batch_id": batch_id,
            "foundry_version": "2.0.0",
            **(tags or {}),
        }
        try:
            with self._mlflow.start_run(run_name=run_name, tags=run_tags) as run:
                self._run = run
                logger.info("MLflow run started: %s (id=%s)", run_name, run.info.run_id)
                yield
                self._run = None
        except Exception as exc:
            logger.error("MLflow run error: %s", exc)
            yield

    def log_hyperparams(self, **params: Any) -> None:
        """Log hyperparameters (LoRA config, optimizer, dataset stats)."""
        if not self._available or self._mlflow is None:
            return
        try:
            self._mlflow.log_params(params)
        except Exception as exc:
            logger.debug("MLflow log_hyperparams failed: %s", exc)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a scalar metric."""
        if not self._available or self._mlflow is None:
            return
        try:
            self._mlflow.log_metric(key, value, step=step)
        except Exception as exc:
            logger.debug("MLflow log_metric failed: %s", exc)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple metrics in one call."""
        if not self._available or self._mlflow is None:
            return
        try:
            self._mlflow.log_metrics(metrics, step=step)
        except Exception as exc:
            logger.debug("MLflow log_metrics failed: %s", exc)

    def log_quality_gate(self, gate_result: dict) -> None:
        """Log quality gate results as metrics + artifact."""
        metrics = {
            "quality_gate_passed": 1.0 if gate_result.get("passed") else 0.0,
            "avg_quality_score": gate_result.get("avg_quality_score", 0.0),
            "hallucination_rate": gate_result.get("hallucination_rate", 0.0),
            "total_samples": float(gate_result.get("total_samples", 0)),
            "rejected_samples": float(gate_result.get("rejected_samples", 0)),
        }
        self.log_metrics(metrics)
        self.log_dict(gate_result, "quality_gate_report.json")

    def log_dataset_stats(self, batch_id: str, stats: dict) -> None:
        """Log dataset statistics for lineage tracking."""
        metrics = {
            "dataset_size": float(stats.get("total_samples", 0)),
            "dpo_pairs": float(stats.get("dpo_pairs", 0)),
            "unique_perspectives": float(stats.get("unique_perspectives", 0)),
            "avg_quality_score": float(stats.get("avg_quality_score", 0.0)),
        }
        self.log_metrics(metrics)
        self.log_dict({"batch_id": batch_id, **stats}, "dataset_stats.json")

    def log_model_artifact(self, model_dir: str, artifact_path: str = "model") -> None:
        """Log model checkpoint directory as MLflow artifact."""
        if not self._available or self._mlflow is None:
            return
        try:
            self._mlflow.log_artifacts(model_dir, artifact_path=artifact_path)
            logger.info("MLflow: model artifacts logged from %s → %s", model_dir, artifact_path)
        except Exception as exc:
            logger.warning("MLflow log_artifacts failed: %s", exc)

    def log_dict(self, data: dict, filename: str) -> None:
        """Log a dict as a JSON artifact."""
        if not self._available or self._mlflow is None:
            return
        try:
            self._mlflow.log_dict(data, filename)
        except Exception as exc:
            logger.debug("MLflow log_dict failed: %s", exc)

    def log_sft_config(self, model_name: str, lora_config: dict, training_args: dict) -> None:
        """Log full SFT training configuration."""
        self.log_hyperparams(
            base_model=model_name,
            **{f"lora_{k}": v for k, v in lora_config.items()},
            **{f"train_{k}": v for k, v in training_args.items()},
        )

    def register_model(self, model_name: str, model_version_tags: Optional[Dict] = None) -> None:
        """Register trained model in MLflow Model Registry."""
        if not self._available or self._mlflow is None or self._run is None:
            return
        try:
            run_id = self._run.info.run_id
            model_uri = f"runs:/{run_id}/model"
            self._mlflow.register_model(model_uri, model_name, tags=model_version_tags)
            logger.info("MLflow: model '%s' registered from run %s", model_name, run_id)
        except Exception as exc:
            logger.warning("MLflow register_model failed: %s", exc)


# Convenience singleton
_tracker: Optional[FoundryMLflowTracker] = None


def get_tracker(experiment_name: str = _MLFLOW_EXPERIMENT_BASE) -> FoundryMLflowTracker:
    global _tracker
    if _tracker is None or _tracker.experiment_name != experiment_name:
        _tracker = FoundryMLflowTracker(experiment_name=experiment_name)
    return _tracker
