"""MLflow tracking helpers for AutoML pipeline components.

Reads ``MLFLOW_*`` environment variables from the pod (typically mounted from the
``mlflow-connection`` secret via ``mlflow_connection_secret_name``).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

logger = logging.getLogger(__name__)

MlflowMode = Literal["disabled", "kfp", "connection"]

# Default example secret name when documenting manual connection setup.
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

# Metrics logged on child runs, keyed by AutoML task type.
TASK_TYPE_METRIC_KEYS: dict[str, tuple[str, ...]] = {
    "binary": ("accuracy", "balanced_accuracy", "f1", "precision", "recall", "roc_auc", "mcc"),
    "multiclass": ("accuracy", "balanced_accuracy", "f1", "precision", "recall"),
    "regression": ("r2", "root_mean_squared_error", "mean_squared_error", "mean_absolute_error"),
    "time_series": ("MASE", "WQL", "sMAPE", "RMSE", "mean_wQuantileLoss"),
}

RUN_TYPE_PIPELINE = "pipeline"
RUN_TYPE_MODEL = "model"
MLFLOW_PARENT_RUN_ID_TAG = "mlflow.parentRunId"

# Tag key used to mark the deployment stage of a registered best-model version.
TARGET_STAGE_TAG = "target_stage"

# Relative locations inside each ``<model>_FULL`` directory of the models artifact.
MODEL_PREDICTOR_SUBDIR = "predictor"
MODEL_NOTEBOOK_RELPATH = "notebooks/automl_predictor_notebook.ipynb"

# HTTP header RHOAI's multi-tenant MLflow requires to scope every request to a workspace
# (the project namespace, e.g. "ns-automl-benchmarking"). Sent from MLFLOW_WORKSPACE.
MLFLOW_WORKSPACE_HEADER = "x-mlflow-workspace"


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


def build_mlflow_stage_map_block(
    *,
    tracking_uri: str | None = None,
    experiment_id: str | None = None,
    run_id: str | None = None,
    workspace: str | None = None,
) -> dict[str, Any]:
    """Build the ``mlflow`` object embedded in ``component_stage_map.json`` (ADR schema)."""
    uri = (tracking_uri if tracking_uri is not None else os.getenv("MLFLOW_TRACKING_URI", "")).strip()
    if not uri:
        return {"tracking_enabled": False}

    exp_id = (experiment_id if experiment_id is not None else os.getenv("MLFLOW_EXPERIMENT_ID", "")).strip()
    parent_run_id = (run_id if run_id is not None else os.getenv("MLFLOW_RUN_ID", "")).strip()
    ws = (workspace if workspace is not None else os.getenv("MLFLOW_WORKSPACE", "")).strip()

    block: dict[str, Any] = {
        "tracking_enabled": True,
        "tracking_uri": uri,
    }
    if exp_id:
        block["experiment_id"] = exp_id
    if parent_run_id:
        block["run_id"] = parent_run_id
    if ws:
        block["workspace"] = ws
    run_url = build_mlflow_run_url(uri, exp_id, parent_run_id)
    if run_url:
        block["run_url"] = run_url
    return block


def configure_mlflow_client(mlflow: Any, config: MlflowConfig) -> None:
    """Apply tracking URI and workspace before MLflow API calls."""
    mlflow.set_tracking_uri(config.tracking_uri)
    if config.workspace:
        _apply_workspace(mlflow, config.tracking_uri, config.workspace)


def _apply_workspace(mlflow: Any, tracking_uri: str, workspace: str) -> None:
    """Route the workspace to the MLflow server on every request.

    RHOAI's multi-tenant MLflow rejects any call that does not identify a workspace
    (``INVALID_PARAMETER_VALUE: Workspace context is required``). Newer MLflow clients
    expose ``mlflow.set_workspace``; upstream/plain MLflow does not, so we attach the
    ``x-mlflow-workspace`` header on the underlying ``requests`` session instead.
    Best-effort: a failure here leaves tracking disabled but never fails the pipeline.
    """
    set_workspace = getattr(mlflow, "set_workspace", None)
    if callable(set_workspace):
        set_workspace(workspace)
        return
    try:
        _install_workspace_request_header(tracking_uri, workspace)
    except Exception:
        logger.exception("Failed to install MLflow workspace request header for %r.", workspace)


def _install_workspace_request_header(tracking_uri: str, workspace: str) -> None:
    """Monkeypatch ``requests`` so calls to the tracking host carry the workspace header.

    Scoped to the tracking host so it never affects artifact stores (e.g. S3) on other
    hosts. Idempotent: re-applying in the same process is a no-op.
    """
    from urllib.parse import urlsplit

    import requests

    host = urlsplit(tracking_uri).netloc
    if not host:
        return

    original_request = requests.Session.request
    if getattr(original_request, "_automl_workspace_patch", False):
        return

    def request_with_workspace(self, method, url, *args, **kwargs):
        try:
            if urlsplit(url).netloc == host:
                headers = dict(kwargs.get("headers") or {})
                headers.setdefault(MLFLOW_WORKSPACE_HEADER, workspace)
                kwargs["headers"] = headers
        except Exception:
            logger.debug("Could not attach MLflow workspace header", exc_info=True)
        return original_request(self, method, url, *args, **kwargs)

    request_with_workspace._automl_workspace_patch = True
    requests.Session.request = request_with_workspace


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


def display_model_run_name(model_name: str) -> str:
    """Return a concise MLflow run name for a refitted AutoGluon model."""
    if model_name.endswith("_FULL"):
        return model_name[: -len("_FULL")]
    return model_name


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


def _normalize_model_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Flatten artifact metadata that nests scores under ``test_data``."""
    test_data = metrics.get("test_data")
    if isinstance(test_data, dict) and test_data:
        return dict(test_data)
    return dict(metrics)


def _load_model_metrics(
    models_artifact: Any,
    models_path: Path,
    model_name: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Load per-model metrics from the local artifact tree or pipeline metadata."""
    metrics = _load_metrics_json(models_path, model_name)
    if metrics:
        return metrics
    for model in context.get("models", []):
        if model.get("name") != model_name:
            continue
        raw_metrics = model.get("metrics", {})
        if isinstance(raw_metrics, dict):
            return raw_metrics
    return {}


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


def _metrics_for_task(task_type: str, metrics: dict[str, Any]) -> dict[str, float]:
    """Select task-relevant scalar metrics for a model child run."""
    normalized = _normalize_model_metrics(metrics)
    scalars = _scalar_metrics(normalized)
    preferred_keys = TASK_TYPE_METRIC_KEYS.get(task_type)
    if preferred_keys:
        selected = {key: value for key, value in scalars.items() if key in preferred_keys}
        if selected:
            return selected
    return scalars


def _stringify_params(params: dict[str, Any]) -> dict[str, str]:
    """MLflow params must be strings."""
    return {key: str(value) for key, value in params.items()}


def _resolve_autogluon_version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version

        for package in ("autogluon.tabular", "autogluon.core", "autogluon"):
            try:
                return version(package)
            except PackageNotFoundError:
                continue
    except Exception:
        logger.debug("Could not resolve autogluon version from package metadata", exc_info=True)
    try:
        import autogluon

        return getattr(autogluon, "__version__", "unknown")
    except Exception:
        logger.debug("Could not import autogluon for version lookup", exc_info=True)
    return "unknown"


def _safe_log_artifact(mlflow: Any, file_path: Path, artifact_path: str) -> bool:
    """Upload a local file to MLflow, logging and continuing on failure."""
    if not file_path.is_file():
        return False
    try:
        mlflow.log_artifact(str(file_path), artifact_path=artifact_path)
        return True
    except Exception:
        logger.exception("Failed to upload MLflow artifact %s to %s", file_path, artifact_path)
        return False


def _safe_log_artifacts_dir(mlflow: Any, dir_path: Path, artifact_path: str) -> bool:
    """Upload a local directory tree to MLflow, logging and continuing on failure."""
    if not dir_path.is_dir():
        return False
    try:
        mlflow.log_artifacts(str(dir_path), artifact_path=artifact_path)
        return True
    except Exception:
        logger.exception("Failed to upload MLflow artifacts dir %s to %s", dir_path, artifact_path)
        return False


def _write_temp_json(tmp_dir: Path, filename: str, payload: dict[str, Any]) -> Path:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    output = tmp_dir / filename
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


def _load_plot_renderer() -> Any:
    """Return the ``mlflow_plots`` module, or ``None`` when it cannot be imported.

    Available on the AutoML runtime image (ships ``kfp_components``). In the rare
    embedded-fallback path (older images), only ``mlflow_tracking.py`` is embedded, so
    plot rendering is skipped.
    """
    try:
        from kfp_components.components.training.automl.shared import mlflow_plots

        return mlflow_plots
    except Exception:
        try:
            import mlflow_plots  # type: ignore[import-not-found]

            return mlflow_plots
        except Exception:
            logger.info("mlflow_plots module unavailable; skipping plot rendering.")
            return None


def _log_model_and_notebook_artifacts(mlflow: Any, model_dir: Path) -> dict[str, bool]:
    """Upload the deployment predictor (model.pkl) and the model notebook to MLflow."""
    results = {"model": False, "notebook": False}
    predictor_dir = model_dir / MODEL_PREDICTOR_SUBDIR
    if _safe_log_artifacts_dir(mlflow, predictor_dir, "model"):
        results["model"] = True
    notebook_path = model_dir / MODEL_NOTEBOOK_RELPATH
    if _safe_log_artifact(mlflow, notebook_path, "notebooks"):
        results["notebook"] = True
    return results


def _log_rendered_plots(
    mlflow: Any,
    *,
    task_type: str,
    model_dir: Path,
    tmp_dir: Path,
    plot_renderer: Any,
) -> int:
    """Render confusion-matrix/ROC (or back-testing) PNGs and upload them to MLflow."""
    if plot_renderer is None:
        return 0
    try:
        rendered = plot_renderer.render_model_plots(task_type, model_dir, tmp_dir)
    except Exception:
        logger.exception("Plot rendering failed for %s", model_dir)
        return 0
    uploaded = 0
    for plot_path in rendered:
        if _safe_log_artifact(mlflow, Path(plot_path), "plots"):
            uploaded += 1
    return uploaded


def _register_best_model(
    mlflow: Any,
    *,
    best_child_run_id: str,
    model_registry_name: str,
    target_stage: str,
) -> dict[str, str]:
    """Register the best model's ``model`` artifacts and tag the version with target_stage."""
    if not best_child_run_id or not model_registry_name:
        return {}
    model_uri = f"runs:/{best_child_run_id}/model"
    try:
        registered = mlflow.register_model(model_uri, model_registry_name)
    except Exception as exc:
        logger.exception("MLflow model registration failed for %s.", model_registry_name)
        return {"mlflow_registry_error": str(exc)}

    version = str(getattr(registered, "version", ""))
    info: dict[str, str] = {
        "mlflow_registered_model": model_registry_name,
        "mlflow_model_version": version,
    }
    if target_stage:
        try:
            client = mlflow.MlflowClient()
            client.set_model_version_tag(
                name=model_registry_name,
                version=version,
                key=TARGET_STAGE_TAG,
                value=target_stage,
            )
            info["mlflow_target_stage"] = target_stage
        except Exception as exc:
            logger.exception("Failed to set target_stage tag on %s v%s.", model_registry_name, version)
            info["mlflow_registry_error"] = str(exc)
    return info


def _build_leaderboard_summary(
    *,
    model_names: list[str],
    valid_metrics: dict[str, dict[str, Any]],
    eval_metric: str,
) -> dict[str, Any]:
    models = []
    for model_name in model_names:
        metrics = valid_metrics.get(model_name)
        if not metrics:
            continue
        models.append(
            {
                "model_name": model_name,
                "display_name": display_model_run_name(model_name),
                "metrics": _normalize_model_metrics(metrics),
            }
        )
    return {"eval_metric": eval_metric, "models": models}


@contextmanager
def _child_mlflow_run(
    mlflow: Any,
    *,
    experiment_id: str,
    parent_run_id: str,
    run_name: str,
    tags: dict[str, str],
) -> Iterator[Any]:
    """Open a nested child run, falling back to explicit parent linkage when needed."""
    try:
        with mlflow.start_run(run_name=run_name, nested=True) as run:
            yield run
            return
    except Exception as exc:
        logger.warning(
            "MLflow nested child run failed for %s (%s); trying MlflowClient.create_run.",
            run_name,
            exc,
        )

    run_tags = [mlflow.entities.RunTag(MLFLOW_PARENT_RUN_ID_TAG, parent_run_id)]
    for key, value in tags.items():
        run_tags.append(mlflow.entities.RunTag(key, value))

    client = mlflow.MlflowClient()
    created_run = client.create_run(experiment_id=experiment_id, run_name=run_name, tags=run_tags)
    try:
        with mlflow.start_run(run_id=created_run.info.run_id) as run:
            yield run
    except Exception as exc:
        client.set_terminated(created_run.info.run_id, status="FAILED")
        raise exc


def _aggregate_scores(model_metrics: dict[str, dict[str, Any]], eval_metric: str) -> list[float]:
    scores: list[float] = []
    for metrics in model_metrics.values():
        normalized = _normalize_model_metrics(metrics)
        if eval_metric in normalized and isinstance(normalized[eval_metric], (int, float)):
            scores.append(float(normalized[eval_metric]))
    return scores


def _log_optional_metric_artifacts(
    mlflow: Any,
    metrics_dir: Path,
    *,
    metrics: dict[str, Any],
    display_name: str,
    tmp_dir: Path,
) -> None:
    uploaded = False
    metrics_json = metrics_dir / "metrics.json"
    if _safe_log_artifact(mlflow, metrics_json, "metrics"):
        uploaded = True

    for filename in OPTIONAL_METRIC_ARTIFACTS:
        artifact_file = metrics_dir / filename
        if _safe_log_artifact(mlflow, artifact_file, "metrics"):
            uploaded = True

    if not uploaded:
        summary = _write_temp_json(
            tmp_dir,
            f"{display_name}_metrics.json",
            _normalize_model_metrics(metrics),
        )
        _safe_log_artifact(mlflow, summary, "metrics")


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
    log_model_artifacts: bool,
    register_best_model: bool,
    model_registry_name: str,
    target_stage: str,
) -> tuple[str, list[str], list[str], dict[str, str]]:
    parent_active = mlflow.active_run()
    if parent_active is None:
        logger.warning("No active MLflow parent run; skipping child run creation.")
        return "", [], [], {}

    parent_run_id = parent_active.info.run_id
    experiment_id = parent_active.info.experiment_id
    child_run_ids: list[str] = []
    child_run_errors: list[str] = []
    child_run_id_by_model: dict[str, str] = {}
    tmp_dir = Path(tempfile.mkdtemp(prefix="automl-mlflow-"))
    plot_renderer = _load_plot_renderer()

    best_model_name = max(
        valid_metrics,
        key=lambda name: float(_normalize_model_metrics(valid_metrics[name]).get(eval_metric, float("-inf"))),
    )

    mlflow.set_tags(
        {
            "pipeline_name": pipeline_name,
            "kfp_run_id": kfp_run_id,
            "kfp_run_name": kfp_run_name,
            "task_type": task_type,
            "autogluon_version": autogluon_version,
            "run_type": RUN_TYPE_PIPELINE,
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

    mlflow.log_params(_stringify_params(parent_params))

    if scores:
        mlflow.log_metric("best_score", max(scores))
        mlflow.log_metric("worst_score", min(scores))
        mlflow.log_metric("mean_score", sum(scores) / len(scores))
    mlflow.log_metric("num_models_trained", len(valid_metrics))

    html_path = resolve_leaderboard_html_path(html_artifact_path)
    if html_path is not None:
        _safe_log_artifact(mlflow, html_path, "reports")
    else:
        logger.warning("Leaderboard HTML not found at %s.", html_artifact_path)

    summary_path = _write_temp_json(
        tmp_dir,
        "leaderboard_summary.json",
        _build_leaderboard_summary(
            model_names=model_names,
            valid_metrics=valid_metrics,
            eval_metric=eval_metric,
        ),
    )
    _safe_log_artifact(mlflow, summary_path, "reports")

    base_uri = str(models_artifact.uri).rstrip("/")
    for model_name in model_names:
        metrics = valid_metrics.get(model_name)
        if not metrics:
            logger.warning("Skipping MLflow child run for %s: no metrics.", model_name)
            continue

        model_type, stack_level = parse_model_name(model_name)
        display_name = display_model_run_name(model_name)
        model_uri = f"{base_uri}/{model_name}"
        model_dir = models_path / model_name
        task_metrics = _metrics_for_task(task_type, metrics)

        child_params: dict[str, Any] = {
            "model_name": model_name,
            "model_type": model_type,
            "stack_level": stack_level,
            "metrics_path": f"{model_uri}/metrics",
            "predictor_path": f"{model_uri}/predictor",
            "notebook_path": f"{model_uri}/notebooks/automl_predictor_notebook.ipynb",
        }
        normalized = _normalize_model_metrics(metrics)
        if "fit_time" in normalized:
            child_params["fit_time"] = normalized["fit_time"]
        if "pred_time_val" in normalized:
            child_params["predict_time"] = normalized["pred_time_val"]

        child_tags = {
            "run_type": RUN_TYPE_MODEL,
            "model_name": model_name,
            "model_type": model_type,
            "stack_level": str(stack_level),
            "kfp_run_id": kfp_run_id,
        }

        try:
            with _child_mlflow_run(
                mlflow,
                experiment_id=experiment_id,
                parent_run_id=parent_run_id,
                run_name=display_name,
                tags=child_tags,
            ) as _child_run:
                mlflow.set_tags(child_tags)
                mlflow.log_params(_stringify_params(child_params))
                if task_metrics:
                    mlflow.log_metrics(task_metrics)
                else:
                    logger.warning("No scalar metrics to log for MLflow child run %s.", display_name)

                metrics_dir = model_dir / "metrics"
                _log_optional_metric_artifacts(
                    mlflow,
                    metrics_dir,
                    metrics=metrics,
                    display_name=display_name,
                    tmp_dir=tmp_dir,
                )

                _log_rendered_plots(
                    mlflow,
                    task_type=task_type,
                    model_dir=model_dir,
                    tmp_dir=tmp_dir / f"{display_name}_plots",
                    plot_renderer=plot_renderer,
                )

                if log_model_artifacts:
                    _log_model_and_notebook_artifacts(mlflow, model_dir)

                active_child = mlflow.active_run()
                if active_child is not None and active_child.info.run_id:
                    child_run_id = str(active_child.info.run_id)
                    child_run_ids.append(child_run_id)
                    child_run_id_by_model[model_name] = child_run_id
        except Exception as exc:
            message = f"{model_name}: {exc}"
            child_run_errors.append(message)
            logger.exception("Failed to create MLflow child run for model %s.", model_name)

    if child_run_ids:
        mlflow.log_metric("child_run_count", len(child_run_ids))
    else:
        logger.warning(
            "No MLflow child runs were created under parent run %s. Errors: %s",
            parent_run_id,
            child_run_errors,
        )
    if child_run_errors:
        mlflow.set_tag("child_run_errors", json.dumps(child_run_errors)[:500])

    registry_info: dict[str, str] = {}
    if register_best_model:
        best_child_run_id = child_run_id_by_model.get(best_model_name, "")
        if not best_child_run_id:
            logger.warning(
                "register_best_model requested but no child run for best model %s; skipping registration.",
                best_model_name,
            )
        elif not model_registry_name:
            logger.warning("register_best_model requested but model_registry_name is empty; skipping.")
        else:
            registry_info = _register_best_model(
                mlflow,
                best_child_run_id=best_child_run_id,
                model_registry_name=model_registry_name,
                target_stage=target_stage,
            )
            for key, value in registry_info.items():
                mlflow.set_tag(key, value)

    return parent_run_id, child_run_ids, child_run_errors, registry_info


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
    log_model_artifacts: bool = True,
    register_best_model: bool = False,
    model_registry_name: str = "",
    target_stage: str = "",
) -> tuple[bool, dict[str, str]]:
    """Log AutoML experiment results to MLflow.

    Each refitted model becomes a nested child run under the parent experiment run, with
    its metrics, params, metric-JSON artifacts, rendered plots, and (when
    ``log_model_artifacts``) its predictor (model.pkl) and notebook. When
    ``register_best_model`` and ``model_registry_name`` are set, the best model is
    registered in the MLflow Model Registry and its version tagged with ``target_stage``.

    Returns ``(logged, tracking_info)`` where ``tracking_info`` may contain
    ``mlflow_run_id``, ``mlflow_experiment_id``, ``mlflow_registered_model`` and
    ``mlflow_model_version`` after logging.
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

    context_raw = models_artifact.metadata.get("context", {})
    if isinstance(context_raw, str):
        context = json.loads(context_raw)
    else:
        context = dict(context_raw) if context_raw else {}

    model_metrics = {name: _load_model_metrics(models_artifact, models_path, name, context) for name in model_names}
    valid_metrics = {name: metrics for name, metrics in model_metrics.items() if metrics}
    if not valid_metrics:
        logger.warning("No metrics.json files found for any model; skipping MLflow logging.")
        return False, {}

    resolved_task_type = task_type or str(context.get("task_type", ""))
    autogluon_version = _resolve_autogluon_version()

    scores = _aggregate_scores(valid_metrics, eval_metric)
    if not scores:
        logger.warning("No scores found for eval_metric=%r; parent aggregate metrics may be empty.", eval_metric)

    try:
        parent_run_id = ""
        child_run_ids: list[str] = []
        child_run_errors: list[str] = []
        registry_info: dict[str, str] = {}
        with parent_mlflow_run(
            mlflow,
            config,
            pipeline_name=pipeline_name,
            kfp_run_name=kfp_run_name,
        ):
            parent_run_id, child_run_ids, child_run_errors, registry_info = _log_runs_under_parent(
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
                log_model_artifacts=log_model_artifacts,
                register_best_model=register_best_model,
                model_registry_name=model_registry_name,
                target_stage=target_stage,
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
            "mlflow_child_run_ids": ",".join(child_run_ids),
            "mlflow_child_run_count": str(len(child_run_ids)),
        }
        tracking_info.update(registry_info)
        if child_run_errors:
            tracking_info["mlflow_child_run_errors"] = json.dumps(child_run_errors)
        logger.info(
            "Logged AutoML results to MLflow (%s mode, parent run %s, %d child runs).",
            config.mode,
            tracking_info["mlflow_run_id"],
            len(child_run_ids),
        )
        return True, tracking_info
    except Exception as exc:
        logger.exception("MLflow logging failed for pipeline run %s.", kfp_run_id)
        return False, {"error": str(exc), "tracking_mode": config.mode}
