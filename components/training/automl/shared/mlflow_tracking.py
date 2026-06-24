"""MLflow tracking helpers for AutoML pipeline components.

Resolution order:
1. KFP/RHOAI platform integration when ``MLFLOW_TRACKING_URI`` and ``MLFLOW_RUN_ID`` are set.
2. User-provided connection secret (``MLFLOW_TRACKING_URI`` without ``MLFLOW_RUN_ID``).
3. Disabled when no tracking URI is available.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

logger = logging.getLogger(__name__)

TRACKING_ARTIFACT_FILENAME = "mlflow_tracking.json"

MlflowMode = Literal["disabled", "kfp", "connection"]

# Mounted from the pipeline Kubernetes secret ``mlflow-connection`` via use_secret_as_env.
MLFLOW_CONNECTION_SECRET_NAME = "mlflow-connection"

MLFLOW_CONNECTION_SECRET_KEY_TO_ENV: dict[str, str] = {
    "MLFLOW_TRACKING_URI": "MLFLOW_TRACKING_URI",
    "MLFLOW_TRACKING_AUTH": "MLFLOW_TRACKING_AUTH",
    "MLFLOW_WORKSPACE": "MLFLOW_WORKSPACE",
    "MLFLOW_EXPERIMENT_NAME": "MLFLOW_EXPERIMENT_NAME",
    "MLFLOW_TRACKING_TOKEN": "MLFLOW_TRACKING_TOKEN",
    "MLFLOW_TRACKING_INSECURE_TLS": "MLFLOW_TRACKING_INSECURE_TLS",
}

OPTIONAL_METRIC_ARTIFACTS = (
    "feature_importance.json",
    "confusion_matrix.json",
    "curves.json",
    "back_testing.json",
)


@dataclass(frozen=True)
class MlflowConfig:
    """Resolved MLflow settings for the current pod."""

    mode: MlflowMode
    tracking_uri: str
    experiment_id: str = ""
    run_id: str = ""
    workspace: str = ""
    tracking_auth: str = ""
    experiment_name: str = ""


def resolve_mlflow_config() -> MlflowConfig | None:
    """Resolve MLflow config from environment variables.

    Platform integration is preferred when ``MLFLOW_RUN_ID`` is present.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if not tracking_uri:
        return None

    run_id = os.getenv("MLFLOW_RUN_ID", "").strip()
    mode: MlflowMode = "kfp" if run_id else "connection"
    return MlflowConfig(
        mode=mode,
        tracking_uri=tracking_uri,
        experiment_id=os.getenv("MLFLOW_EXPERIMENT_ID", "").strip(),
        run_id=run_id,
        workspace=os.getenv("MLFLOW_WORKSPACE", "").strip(),
        tracking_auth=os.getenv("MLFLOW_TRACKING_AUTH", "").strip(),
        experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "").strip(),
    )


def is_mlflow_enabled() -> bool:
    """Return True when a MLflow tracking URI is available."""
    return resolve_mlflow_config() is not None


def read_mlflow_env() -> dict[str, str]:
    """Collect MLflow-related environment variables from the pod."""
    config = resolve_mlflow_config()
    if config is None:
        return {
            "mlflow_tracking_uri": "",
            "mlflow_experiment_id": "",
            "mlflow_run_id": "",
            "mlflow_workspace": "",
            "mlflow_tracking_auth": "",
            "mlflow_experiment_name": "",
            "tracking_mode": "disabled",
        }
    return {
        "mlflow_tracking_uri": config.tracking_uri,
        "mlflow_experiment_id": config.experiment_id,
        "mlflow_run_id": config.run_id,
        "mlflow_workspace": config.workspace,
        "mlflow_tracking_auth": config.tracking_auth,
        "mlflow_experiment_name": config.experiment_name,
        "tracking_mode": config.mode,
    }


def build_mlflow_run_url(tracking_uri: str, experiment_id: str, run_id: str) -> str:
    """Build a deep-link URL to the MLflow UI parent run view."""
    base = tracking_uri.rstrip("/")
    if not base or not experiment_id or not run_id:
        return ""
    return f"{base}/#/experiments/{experiment_id}/runs/{run_id}"


def resolve_leaderboard_html_path(html_artifact_path: str | Path) -> Path | None:
    """Resolve the leaderboard HTML file from a KFP ``dsl.HTML`` artifact path.

    KFP may mount the artifact as a file path or as a directory containing the HTML.
    """
    path = Path(html_artifact_path)
    if path.is_file():
        return path
    if path.is_dir():
        for candidate in (path / "index.html", path / "leaderboard.html"):
            if candidate.is_file():
                return candidate
        html_files = sorted(path.glob("*.html"))
        if len(html_files) == 1:
            return html_files[0]
    return None


def build_tracking_artifact_payload(
    *,
    pipeline_name: str,
    kfp_run_id: str,
    kfp_run_name: str = "",
    mlflow_run_id: str = "",
    mlflow_experiment_id: str = "",
    tracking_mode: str = "",
    mlflow_error: str = "",
) -> dict[str, Any]:
    """Build the JSON payload for the KFP mlflow_tracking artifact."""
    config = resolve_mlflow_config()
    resolved_mode = tracking_mode or (config.mode if config else "disabled")
    tracking_enabled = config is not None or bool(mlflow_run_id)
    if tracking_enabled and resolved_mode == "disabled":
        resolved_mode = "connection" if mlflow_run_id else (config.mode if config else "disabled")

    payload: dict[str, Any] = {
        "tracking_enabled": tracking_enabled,
        "tracking_mode": resolved_mode,
        "kfp_run_id": kfp_run_id,
        "pipeline_name": pipeline_name,
    }
    if kfp_run_name:
        payload["kfp_run_name"] = kfp_run_name
    if mlflow_error:
        payload["mlflow_error"] = mlflow_error

    if not tracking_enabled:
        return payload

    tracking_uri = config.tracking_uri if config else os.getenv("MLFLOW_TRACKING_URI", "").strip()
    experiment_id = mlflow_experiment_id or (config.experiment_id if config else "")
    run_id = mlflow_run_id or (config.run_id if config else "")
    workspace = config.workspace if config else os.getenv("MLFLOW_WORKSPACE", "").strip()
    experiment_name = config.experiment_name if config else os.getenv("MLFLOW_EXPERIMENT_NAME", "").strip()
    payload.update(
        {
            "mlflow_tracking_uri": tracking_uri,
            "mlflow_experiment_id": experiment_id,
            "mlflow_run_id": run_id,
            "mlflow_workspace": workspace,
            "mlflow_experiment_name": experiment_name,
            "mlflow_run_url": build_mlflow_run_url(tracking_uri, experiment_id, run_id),
        }
    )
    return payload


def write_tracking_artifact(
    artifact_path: str | Path,
    *,
    pipeline_name: str,
    kfp_run_id: str,
    kfp_run_name: str = "",
    mlflow_run_id: str = "",
    mlflow_experiment_id: str = "",
    tracking_mode: str = "",
    mlflow_error: str = "",
) -> Path:
    """Write mlflow_tracking.json under the KFP artifact directory."""
    output_dir = Path(artifact_path)
    if output_dir.is_file():
        output_file = output_dir
        output_dir = output_dir.parent
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / TRACKING_ARTIFACT_FILENAME
    payload = build_tracking_artifact_payload(
        pipeline_name=pipeline_name,
        kfp_run_id=kfp_run_id,
        kfp_run_name=kfp_run_name,
        mlflow_run_id=mlflow_run_id,
        mlflow_experiment_id=mlflow_experiment_id,
        tracking_mode=tracking_mode,
        mlflow_error=mlflow_error,
    )
    payload_text = json.dumps(payload, indent=2)
    with output_file.open("w", encoding="utf-8") as f:
        f.write(payload_text)
        f.flush()
    # Some KFP/S3 artifact browsers expect a ``data`` object under the artifact prefix.
    data_file = output_dir / "data"
    if data_file != output_file:
        with data_file.open("w", encoding="utf-8") as f:
            f.write(payload_text)
            f.flush()
    return output_file


def configure_mlflow_client(mlflow: Any, config: MlflowConfig) -> None:
    """Apply tracking URI and workspace before MLflow API calls."""
    mlflow.set_tracking_uri(config.tracking_uri)
    if config.workspace:
        set_workspace = getattr(mlflow, "set_workspace", None)
        if callable(set_workspace):
            set_workspace(config.workspace)


@contextmanager
def parent_mlflow_run(
    mlflow: Any,
    config: MlflowConfig,
    *,
    pipeline_name: str,
    kfp_run_name: str = "",
) -> Iterator[Any]:
    """Open the parent MLflow run for platform or connection-backed tracking."""
    configure_mlflow_client(mlflow, config)
    if config.mode == "kfp":
        with mlflow.start_run(run_id=config.run_id) as run:
            yield run
        return

    experiment_name = config.experiment_name or pipeline_name
    mlflow.set_experiment(experiment_name)
    parent_run_name = kfp_run_name or pipeline_name
    with mlflow.start_run(run_name=parent_run_name) as run:
        yield run


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


def _log_runs_under_parent(
    mlflow: Any,
    *,
    models_artifact: Any,
    models_path: Path,
    model_names: list[str],
    valid_metrics: dict[str, dict[str, Any]],
    html_artifact_path: str | Path,
    eval_metric: str,
    pipeline_name: str,
    kfp_run_id: str,
    kfp_run_name: str,
    task_type: str,
    preset: str,
    top_n: int,
    context: dict[str, Any],
    autogluon_version: str,
    scores: list[float],
) -> str:
    best_model_name = max(
        valid_metrics,
        key=lambda name: float(valid_metrics[name].get(eval_metric, float("-inf"))),
    )

    mlflow.set_tags(
        {
            "pipeline_name": pipeline_name,
            "kfp_run_id": kfp_run_id,
            "kfp_run_name": kfp_run_name,
            "task_type": task_type,
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

    html_path = resolve_leaderboard_html_path(html_artifact_path)
    if html_path is not None:
        mlflow.log_artifact(str(html_path), artifact_path="reports")
    else:
        logger.warning("Leaderboard HTML not found at %s; skipping MLflow report artifact.", html_artifact_path)

    base_uri = str(models_artifact.uri).rstrip("/")
    for model_name in model_names:
        metrics = valid_metrics.get(model_name)
        if not metrics:
            logger.warning("Skipping MLflow child run for %s: no metrics.", model_name)
            continue

        model_type, stack_level = parse_model_name(model_name)
        model_uri = f"{base_uri}/{model_name}"

        child_params: dict[str, Any] = {
            "model_type": model_type,
            "stack_level": stack_level,
            "metrics_path": f"{model_uri}/metrics",
            "predictor_path": f"{model_uri}/predictor",
            "notebook_path": f"{model_uri}/notebooks/automl_predictor_notebook.ipynb",
        }
        if "fit_time" in metrics:
            child_params["fit_time"] = metrics["fit_time"]
        if "pred_time_val" in metrics:
            child_params["predict_time"] = metrics["pred_time_val"]

        with mlflow.start_run(run_name=model_name, nested=True):
            mlflow.log_params(child_params)
            mlflow.log_metrics(_scalar_metrics(metrics))

            metrics_dir = models_path / model_name / "metrics"
            _log_optional_metric_artifacts(mlflow, metrics_dir)

    active_run = mlflow.active_run()
    if active_run is None:
        return ""
    return active_run.info.run_id


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
) -> tuple[bool, dict[str, str]]:
    """Log AutoML experiment results to MLflow.

    Returns ``(logged, tracking_info)`` where ``tracking_info`` may contain
    ``mlflow_run_id`` and ``mlflow_experiment_id`` after connection-mode logging.
    """
    config = resolve_mlflow_config()
    if config is None:
        logger.info("MLflow not enabled (MLFLOW_TRACKING_URI unset); skipping logging.")
        return False, {}

    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow package is not installed in the runtime image; skipping MLflow logging.")
        return False, {"error": "mlflow package not installed"}

    models_path = Path(models_artifact.path)
    model_names = _load_model_names(models_artifact)
    if not model_names:
        logger.warning("No model_names in models artifact metadata; skipping MLflow logging.")
        return False, {}

    model_metrics = {name: _load_metrics_json(models_path, name) for name in model_names}
    valid_metrics = {name: metrics for name, metrics in model_metrics.items() if metrics}
    if not valid_metrics:
        logger.warning("No metrics.json files found for any model; skipping MLflow logging.")
        return False, {}

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

    try:
        parent_run_id = ""
        with parent_mlflow_run(
            mlflow,
            config,
            pipeline_name=pipeline_name,
            kfp_run_name=kfp_run_name,
        ):
            parent_run_id = _log_runs_under_parent(
                mlflow,
                models_artifact=models_artifact,
                models_path=models_path,
                model_names=model_names,
                valid_metrics=valid_metrics,
                html_artifact_path=html_artifact_path,
                eval_metric=eval_metric,
                pipeline_name=pipeline_name,
                kfp_run_id=kfp_run_id,
                kfp_run_name=kfp_run_name,
                task_type=resolved_task_type,
                preset=preset,
                top_n=top_n,
                context=context,
                autogluon_version=autogluon_version,
                scores=scores,
            )

        experiment_id = config.experiment_id
        if config.mode == "connection":
            active_experiment = mlflow.get_experiment_by_name(config.experiment_name or pipeline_name)
            if active_experiment is not None:
                experiment_id = active_experiment.experiment_id

        tracking_info = {
            "mlflow_run_id": parent_run_id or config.run_id,
            "mlflow_experiment_id": experiment_id,
            "tracking_mode": config.mode,
        }
        logger.info(
            "Logged AutoML results to MLflow (%s mode, parent run %s, %d child runs).",
            config.mode,
            tracking_info["mlflow_run_id"],
            len(valid_metrics),
        )
        return True, tracking_info
    except Exception as exc:
        logger.exception("MLflow logging failed for pipeline run %s.", kfp_run_id)
        return False, {"error": str(exc), "tracking_mode": config.mode}
