"""MLflow tracking helpers for AutoML pipeline components.

Uses RHOAI/KFP-injected environment variables when MLflow integration is enabled
at the project level. All logging is explicit (no AutoGluon autologging).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TRACKING_ARTIFACT_FILENAME = "mlflow_tracking.json"

OPTIONAL_METRIC_ARTIFACTS = (
    "feature_importance.json",
    "confusion_matrix.json",
    "curves.json",
    "back_testing.json",
)


def is_mlflow_enabled() -> bool:
    """Return True when KFP has injected a MLflow tracking URI."""
    return bool(os.getenv("MLFLOW_TRACKING_URI", "").strip())


def read_mlflow_env() -> dict[str, str]:
    """Collect MLflow-related environment variables from the pod."""
    return {
        "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "").strip(),
        "mlflow_experiment_id": os.getenv("MLFLOW_EXPERIMENT_ID", "").strip(),
        "mlflow_run_id": os.getenv("MLFLOW_RUN_ID", "").strip(),
        "mlflow_workspace": os.getenv("MLFLOW_WORKSPACE", "").strip(),
        "mlflow_tracking_auth": os.getenv("MLFLOW_TRACKING_AUTH", "").strip(),
    }


def build_mlflow_run_url(tracking_uri: str, experiment_id: str, run_id: str) -> str:
    """Build a deep-link URL to the MLflow UI parent run view."""
    base = tracking_uri.rstrip("/")
    if not base or not experiment_id or not run_id:
        return ""
    return f"{base}/#/experiments/{experiment_id}/runs/{run_id}"


def build_tracking_artifact_payload(
    *,
    pipeline_name: str,
    kfp_run_id: str,
    kfp_run_name: str = "",
) -> dict[str, Any]:
    """Build the JSON payload for the KFP mlflow_tracking artifact."""
    payload: dict[str, Any] = {
        "tracking_enabled": is_mlflow_enabled(),
        "kfp_run_id": kfp_run_id,
        "pipeline_name": pipeline_name,
    }
    if kfp_run_name:
        payload["kfp_run_name"] = kfp_run_name

    if not payload["tracking_enabled"]:
        return payload

    env = read_mlflow_env()
    payload.update(
        {
            "mlflow_tracking_uri": env["mlflow_tracking_uri"],
            "mlflow_experiment_id": env["mlflow_experiment_id"],
            "mlflow_run_id": env["mlflow_run_id"],
            "mlflow_workspace": env["mlflow_workspace"],
            "mlflow_run_url": build_mlflow_run_url(
                env["mlflow_tracking_uri"],
                env["mlflow_experiment_id"],
                env["mlflow_run_id"],
            ),
        }
    )
    return payload


def write_tracking_artifact(
    artifact_path: str | Path,
    *,
    pipeline_name: str,
    kfp_run_id: str,
    kfp_run_name: str = "",
) -> Path:
    """Write mlflow_tracking.json under the KFP artifact directory."""
    output_dir = Path(artifact_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / TRACKING_ARTIFACT_FILENAME
    payload = build_tracking_artifact_payload(
        pipeline_name=pipeline_name,
        kfp_run_id=kfp_run_id,
        kfp_run_name=kfp_run_name,
    )
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return output_file


def parse_model_name(model_name: str) -> tuple[str, int]:
    """Extract model family and stack level from an AutoGluon model name."""
    model_type = model_name.split("_")[0] if "_" in model_name else model_name
    stack_level = 1
    if "_L" in model_name:
        suffix = model_name.rsplit("_L", maxsplit=1)[-1]
        level_part = suffix.split("_", maxsplit=1)[0]
        if level_part.isdigit():
            stack_level = int(level_part)
    return model_type, stack_level


def _load_model_names(models_artifact: Any) -> list[str]:
    model_names_raw = models_artifact.metadata.get("model_names", "[]")
    if isinstance(model_names_raw, str):
        return json.loads(model_names_raw)
    return list(model_names_raw)


def _load_metrics_json(models_artifact_path: Path, model_name: str) -> dict[str, Any]:
    metrics_path = models_artifact_path / model_name / "metrics" / "metrics.json"
    if not metrics_path.is_file():
        return {}
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _scalar_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    logged: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            logged[key] = float(value)
        elif hasattr(value, "item"):
            logged[key] = float(value)
    return logged


def _aggregate_scores(model_metrics: dict[str, dict[str, Any]], eval_metric: str) -> list[float]:
    scores: list[float] = []
    for metrics in model_metrics.values():
        if eval_metric in metrics and isinstance(metrics[eval_metric], (int, float)):
            scores.append(float(metrics[eval_metric]))
    return scores


def _log_optional_metric_artifacts(mlflow: Any, metrics_dir: Path) -> None:
    metrics_json = metrics_dir / "metrics.json"
    if metrics_json.is_file():
        mlflow.log_artifact(str(metrics_json), artifact_path="metrics")

    for filename in OPTIONAL_METRIC_ARTIFACTS:
        artifact_file = metrics_dir / filename
        if artifact_file.is_file():
            mlflow.log_artifact(str(artifact_file), artifact_path="metrics")


def log_automl_results(
    *,
    models_artifact: Any,
    html_artifact_path: str | Path,
    eval_metric: str,
    pipeline_name: str,
    kfp_run_id: str,
    kfp_run_name: str = "",
    task_type: str = "",
    preset: str = "",
    top_n: int = 0,
) -> bool:
    """Log AutoML experiment results to MLflow under the KFP parent run.

    Returns True when logging ran, False when MLflow is disabled or parent run is missing.
    """
    if not is_mlflow_enabled():
        logger.info("MLflow not enabled (MLFLOW_TRACKING_URI unset); skipping logging.")
        return False

    env = read_mlflow_env()
    parent_run_id = env["mlflow_run_id"]
    if not parent_run_id:
        logger.warning("MLFLOW_TRACKING_URI is set but MLFLOW_RUN_ID is missing; skipping logging.")
        return False

    import mlflow

    models_path = Path(models_artifact.path)
    model_names = _load_model_names(models_artifact)
    if not model_names:
        logger.warning("No model_names in models artifact metadata; skipping MLflow logging.")
        return False

    model_metrics = {name: _load_metrics_json(models_path, name) for name in model_names}
    valid_metrics = {name: metrics for name, metrics in model_metrics.items() if metrics}
    if not valid_metrics:
        logger.warning("No metrics.json files found for any model; skipping MLflow logging.")
        return False

    context_raw = models_artifact.metadata.get("context", {})
    if isinstance(context_raw, str):
        context = json.loads(context_raw)
    else:
        context = dict(context_raw) if context_raw else {}

    resolved_task_type = task_type or str(context.get("task_type", ""))
    autogluon_version = "unknown"
    try:
        import autogluon

        autogluon_version = autogluon.__version__
    except Exception:
        logger.debug("Could not resolve autogluon version", exc_info=True)

    scores = _aggregate_scores(valid_metrics, eval_metric)
    if not scores:
        logger.warning("No scores found for eval_metric=%r; parent aggregate metrics may be empty.", eval_metric)

    best_model_name = max(
        valid_metrics,
        key=lambda name: float(valid_metrics[name].get(eval_metric, float("-inf"))),
    )

    with mlflow.start_run(run_id=parent_run_id):
        mlflow.set_tags(
            {
                "pipeline_name": pipeline_name,
                "kfp_run_id": kfp_run_id,
                "kfp_run_name": kfp_run_name,
                "task_type": resolved_task_type,
                "autogluon_version": autogluon_version,
            }
        )

        parent_params: dict[str, Any] = {
            "eval_metric": eval_metric,
            "best_model_name": best_model_name,
            "autogluon_version": autogluon_version,
        }
        if preset:
            parent_params["preset"] = preset
        if top_n:
            parent_params["top_n"] = top_n

        data_config = context.get("data_config", {})
        if data_config:
            parent_params["data_config"] = json.dumps(data_config, sort_keys=True)

        mlflow.log_params(parent_params)

        if scores:
            mlflow.log_metric("best_score", max(scores))
            mlflow.log_metric("worst_score", min(scores))
            mlflow.log_metric("mean_score", sum(scores) / len(scores))
        mlflow.log_metric("num_models_trained", len(valid_metrics))

        html_path = Path(html_artifact_path)
        if html_path.is_file():
            mlflow.log_artifact(str(html_path), artifact_path="reports")

        base_uri = str(models_artifact.uri).rstrip("/")
        for model_name in model_names:
            metrics = valid_metrics.get(model_name)
            if not metrics:
                logger.warning("Skipping MLflow child run for %s: no metrics.", model_name)
                continue

            model_type, stack_level = parse_model_name(model_name)
            model_uri = f"{base_uri}/{model_name}"

            with mlflow.start_run(run_name=model_name, nested=True):
                mlflow.log_params(
                    {
                        "model_type": model_type,
                        "stack_level": stack_level,
                        "metrics_path": f"{model_uri}/metrics",
                        "predictor_path": f"{model_uri}/predictor",
                        "notebook_path": f"{model_uri}/notebooks/automl_predictor_notebook.ipynb",
                    }
                )
                mlflow.log_metrics(_scalar_metrics(metrics))

                metrics_dir = models_path / model_name / "metrics"
                _log_optional_metric_artifacts(mlflow, metrics_dir)

                notebook_path = models_path / model_name / "notebooks" / "automl_predictor_notebook.ipynb"
                if notebook_path.is_file():
                    mlflow.log_artifact(str(notebook_path), artifact_path="notebooks")

    logger.info(
        "Logged AutoML results to MLflow parent run %s (%d child runs).",
        parent_run_id,
        len(valid_metrics),
    )
    return True
