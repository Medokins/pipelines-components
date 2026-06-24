from pathlib import Path

from kfp import dsl
from kfp_components.utils.consts import AUTOML_IMAGE  # pyright: ignore[reportMissingImports]

_AUTOML_SHARED = Path(__file__).resolve().parents[1] / "shared"


@dsl.component(
    base_image=AUTOML_IMAGE,  # noqa: E501
    embedded_artifact_path=str(_AUTOML_SHARED / "mlflow_tracking.py"),
    install_kfp_package=False,
    packages_to_install=["mlflow[kubernetes]>=3.11"],
)
def automl_mlflow_logger(
    models_artifact: dsl.Input[dsl.Model],
    html_artifact: dsl.Input[dsl.HTML],
    eval_metric: str,
    pipeline_name: str,
    run_id: str,
    task_type: str,
    component_status: dsl.Output[dsl.Artifact],
    mlflow_tracking_artifact: dsl.Output[dsl.Artifact],
    preset: str = "speed",
    top_n: int = 3,
    embedded_artifact: dsl.EmbeddedInput[dsl.Dataset] = None,
) -> None:
    """Log AutoML experiment results to MLflow at the end of the pipeline run.

    MLflow configuration is resolved in this order:

    1. KFP/RHOAI platform env vars (``MLFLOW_TRACKING_URI`` + ``MLFLOW_RUN_ID``).
    2. User connection secret env vars (``MLFLOW_TRACKING_URI`` without ``MLFLOW_RUN_ID``).
    3. Disabled when no tracking URI is available.

    Args:
        models_artifact: Combined models artifact from training with ``metadata["model_names"]``.
        html_artifact: Leaderboard HTML artifact from ``leaderboard_evaluation``.
        eval_metric: Metric used for ranking (e.g. ``accuracy``, ``MASE``).
        pipeline_name: Pipeline name for MLflow tags (from ``dsl.PIPELINE_JOB_RESOURCE_NAME_PLACEHOLDER``).
        run_id: KFP run ID (from ``dsl.PIPELINE_JOB_ID_PLACEHOLDER``).
        task_type: ML task type (``binary``, ``multiclass``, ``regression``, or ``time_series``).
        component_status: Output artifact with stage progress (``component_status.json``).
        mlflow_tracking_artifact: Final MLflow tracking metadata including run IDs when logging succeeds.
        preset: Training quality preset logged on the parent run.
        top_n: Number of top models logged on the parent run.
        embedded_artifact: Embedded ``mlflow_tracking.py`` helper injected by KFP at compile time.

    Raises:
        TypeError: If required string parameters are empty.
        ValueError: If ``top_n`` is not positive.
    """
    import importlib.util
    import logging
    import os
    import sys
    from pathlib import Path

    from kfp_components.components.training.automl.shared.component_status import ComponentStatusTracker

    def _load_mlflow_tracking_helpers():
        if embedded_artifact is not None:
            embedded_path = Path(embedded_artifact.path)
            module_path = embedded_path if embedded_path.is_file() else embedded_path / "mlflow_tracking.py"
            if module_path.is_file():
                module_name = "kfp_components_automl_embedded_mlflow_tracking"
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                if spec is None or spec.loader is None:
                    raise ModuleNotFoundError(
                        f"Cannot load embedded mlflow_tracking from {module_path}"
                    ) from None
                module = importlib.util.module_from_spec(spec)
                # Required before exec_module when the file uses @dataclass.
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                return module.log_automl_results, module.write_tracking_artifact

        from kfp_components.components.training.automl.shared.mlflow_tracking import (
            log_automl_results as _log_automl_results,
        )
        from kfp_components.components.training.automl.shared.mlflow_tracking import (
            write_tracking_artifact as _write_tracking_artifact,
        )

        return _log_automl_results, _write_tracking_artifact

    log_automl_results, write_tracking_artifact = _load_mlflow_tracking_helpers()

    logger = logging.getLogger(__name__)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    logger.info(
        "MLflow env: MLFLOW_TRACKING_URI=%s MLFLOW_RUN_ID=%s MLFLOW_WORKSPACE=%s MLFLOW_EXPERIMENT_NAME=%s",
        "set" if tracking_uri else "unset",
        "set" if os.getenv("MLFLOW_RUN_ID") else "unset",
        os.getenv("MLFLOW_WORKSPACE", "") or "(empty)",
        os.getenv("MLFLOW_EXPERIMENT_NAME", "") or "(empty)",
    )
    if not tracking_uri:
        logger.warning(
            "MLFLOW_TRACKING_URI is unset in this pod. "
            "Enable RHOAI project MLflow integration or verify the mlflow-connection secret "
            "is mounted on exec-automl-mlflow-logger(-2) in the compiled pipeline.yaml."
        )

    if not isinstance(eval_metric, str) or not eval_metric.strip():
        raise TypeError("eval_metric must be a non-empty string.")
    if not isinstance(pipeline_name, str) or not pipeline_name.strip():
        raise TypeError("pipeline_name must be a non-empty string.")
    if not isinstance(run_id, str) or not run_id.strip():
        raise TypeError("run_id must be a non-empty string.")
    if not isinstance(task_type, str) or not task_type.strip():
        raise TypeError("task_type must be a non-empty string.")
    if top_n <= 0:
        raise ValueError(f"top_n must be a positive integer; got {top_n}.")

    status = ComponentStatusTracker(component_status.path, "automl_mlflow_logger")
    with status:
        status.record("log_mlflow_results", "started")

        logged, tracking_info = log_automl_results(
            models_artifact=models_artifact,
            html_artifact_path=html_artifact.path,
            eval_metric=eval_metric,
            pipeline_name=pipeline_name,
            kfp_run_id=run_id,
            kfp_run_name=pipeline_name,
            task_type=task_type,
            preset=preset,
            top_n=top_n,
        )

        tracking_file = write_tracking_artifact(
            mlflow_tracking_artifact.path,
            pipeline_name=pipeline_name,
            kfp_run_id=run_id,
            kfp_run_name=pipeline_name,
            mlflow_run_id=tracking_info.get("mlflow_run_id", ""),
            mlflow_experiment_id=tracking_info.get("mlflow_experiment_id", ""),
            tracking_mode=tracking_info.get("tracking_mode", ""),
            mlflow_error=tracking_info.get("error", ""),
        )
        mlflow_tracking_artifact.metadata["display_name"] = "MLflow Tracking Info"
        mlflow_tracking_artifact.metadata["tracking_enabled"] = str(logged)
        if tracking_info.get("mlflow_run_id"):
            mlflow_tracking_artifact.metadata["mlflow_run_id"] = tracking_info["mlflow_run_id"]
        if tracking_info.get("mlflow_experiment_id"):
            mlflow_tracking_artifact.metadata["mlflow_experiment_id"] = tracking_info["mlflow_experiment_id"]
        if tracking_info.get("tracking_mode"):
            mlflow_tracking_artifact.metadata["tracking_mode"] = tracking_info["tracking_mode"]
        if tracking_info.get("error"):
            mlflow_tracking_artifact.metadata["mlflow_error"] = tracking_info["error"]
        if tracking_uri:
            mlflow_tracking_artifact.metadata["mlflow_tracking_uri"] = tracking_uri

        if logged:
            logger.info("MLflow logging completed for pipeline run %s.", run_id)
            status.record(
                "log_mlflow_results",
                "completed",
                tracking_enabled=True,
                tracking_mode=tracking_info.get("tracking_mode", ""),
            )
        else:
            mlflow_error = tracking_info.get("error", "")
            if mlflow_error:
                logger.warning("MLflow logging failed for pipeline run %s: %s", run_id, mlflow_error)
            else:
                logger.info("MLflow logging skipped for pipeline run %s.", run_id)
            status.record(
                "log_mlflow_results",
                "completed",
                tracking_enabled=False,
                mlflow_error=mlflow_error or None,
            )

        component_status.metadata["display_name"] = "MLflow Logging Status"
        for label, artifact_path in (
            ("component_status", component_status.path),
            ("mlflow_tracking_artifact", mlflow_tracking_artifact.path),
        ):
            path = Path(artifact_path)
            if path.is_dir():
                files = sorted(path.iterdir())
            elif path.is_file():
                files = [path]
            else:
                files = []
            logger.info(
                "Output artifact %s path=%s files=%s",
                label,
                artifact_path,
                [str(file) for file in files],
            )
        print(f"MLflow tracking artifact written to: {tracking_file}")
