from pathlib import Path

from kfp import dsl
from kfp_components.utils.consts import AUTOML_IMAGE  # pyright: ignore[reportMissingImports]

_AUTOML_SHARED = Path(__file__).resolve().parents[1] / "shared"


@dsl.component(
    base_image=AUTOML_IMAGE,
    embedded_artifact_path=str(_AUTOML_SHARED / "mlflow_tracking.py"),
    install_kfp_package=False,
)
def automl_mlflow_logger(
    models_artifact: dsl.Input[dsl.Model],
    html_artifact: dsl.Input[dsl.HTML],
    eval_metric: str,
    pipeline_name: str,
    run_id: str,
    task_type: str,
    component_status: dsl.Output[dsl.Artifact],
    run_name: str = "",
    preset: str = "speed",
    top_n: int = 3,
    log_model_artifacts: bool = True,
    register_best_model: bool = False,
    model_registry_name: str = "",
    target_stage: str = "",
    embedded_artifact: dsl.EmbeddedInput[dsl.Dataset] = None,
) -> None:
    """Log AutoML experiment results to MLflow at the end of the pipeline run.

    Expects ``MLFLOW_*`` environment variables from the ``mlflow_connection_secret_name``
    pipeline parameter (mounted on this step only). When ``MLFLOW_TRACKING_URI`` is unset,
    logging is skipped and the step completes successfully.

    Each refitted model becomes a nested child run under the parent experiment run, logging
    task-specific metrics, params, metric-JSON artifacts, rendered plots (confusion matrix,
    ROC curves, back-testing) and — when ``log_model_artifacts`` — its predictor (model.pkl)
    and filled notebook. When ``register_best_model`` and ``model_registry_name`` are set,
    the best model is registered in the MLflow Model Registry and its version tagged with
    ``target_stage``.

    ``mlflow_tracking.py`` is embedded at compile time when the runtime image does not yet
    ship ``kfp_components...mlflow_tracking``.

    Args:
        models_artifact: Combined models artifact from training with ``metadata["model_names"]``.
        html_artifact: Leaderboard HTML artifact from the training component.
        eval_metric: Metric used for ranking (e.g. ``accuracy``, ``MASE``).
        pipeline_name: Stable pipeline name used as the MLflow experiment name (and tags) when
            the connection does not set ``MLFLOW_EXPERIMENT_NAME``. Pass the pipeline's logical
            name (e.g. ``PIPELINE_NAME``) so runs aggregate under one experiment.
        run_id: KFP run ID (from ``dsl.PIPELINE_JOB_ID_PLACEHOLDER``).
        task_type: ML task type (``binary``, ``multiclass``, ``regression``, or ``time_series``).
        component_status: Output artifact with stage progress (``component_status.json``).
        run_name: Per-execution MLflow parent run name. Pass the run's display name (from
            ``dsl.PIPELINE_JOB_NAME_PLACEHOLDER``) so the MLflow run matches the OpenShift run
            name. Falls back to ``pipeline_name`` when empty.
        preset: Training quality preset logged on the parent run.
        top_n: Number of top models logged on the parent run.
        log_model_artifacts: When True, upload each model's predictor (model.pkl) and notebook.
        register_best_model: When True, register the best model in the MLflow Model Registry.
        model_registry_name: Registered-model name to use when ``register_best_model`` is True.
        target_stage: Optional deployment-stage value set as a ``target_stage`` tag on the
            registered best-model version.
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

    def _load_log_automl_results():
        try:
            from kfp_components.components.training.automl.shared.mlflow_tracking import (
                log_automl_results as _log_automl_results,
            )

            return _log_automl_results
        except ModuleNotFoundError:
            if embedded_artifact is None:
                raise
            embedded_path = Path(embedded_artifact.path)
            module_path = embedded_path if embedded_path.is_file() else embedded_path / "mlflow_tracking.py"
            if not module_path.is_file():
                raise ModuleNotFoundError(
                    f"mlflow_tracking not found in image or embedded artifact at {module_path}"
                ) from None
            module_name = "kfp_components_automl_embedded_mlflow_tracking"
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                raise ModuleNotFoundError(f"Cannot load embedded mlflow_tracking from {module_path}") from None
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module.log_automl_results

    log_automl_results = _load_log_automl_results()

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
            "MLFLOW_TRACKING_URI is unset. Set pipeline parameter mlflow_connection_secret_name "
            "(for example mlflow-connection) when starting the run."
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
    if register_best_model and not model_registry_name.strip():
        raise ValueError("model_registry_name must be set when register_best_model is True.")

    status = ComponentStatusTracker(component_status.path, "automl_mlflow_logger")
    with status:
        status.record("log_mlflow_results", "started")

        logged, tracking_info = log_automl_results(
            models_artifact=models_artifact,
            html_artifact_path=html_artifact.path,
            eval_metric=eval_metric,
            pipeline_name=pipeline_name,
            kfp_run_id=run_id,
            kfp_run_name=run_name or pipeline_name,
            task_type=task_type,
            preset=preset,
            top_n=top_n,
            log_model_artifacts=log_model_artifacts,
            register_best_model=register_best_model,
            model_registry_name=model_registry_name,
            target_stage=target_stage,
        )

        component_status.metadata["display_name"] = "MLflow Logging Status"
        component_status.metadata["tracking_enabled"] = str(logged)
        for key in (
            "mlflow_run_id",
            "mlflow_experiment_id",
            "tracking_mode",
            "mlflow_child_run_count",
            "mlflow_child_run_ids",
            "mlflow_child_run_errors",
            "mlflow_registered_model",
            "mlflow_model_version",
            "mlflow_target_stage",
            "mlflow_registry_error",
        ):
            if tracking_info.get(key):
                component_status.metadata[key] = tracking_info[key]
        if tracking_info.get("error"):
            component_status.metadata["mlflow_error"] = tracking_info["error"]
        if tracking_uri:
            component_status.metadata["mlflow_tracking_uri"] = tracking_uri

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


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        automl_mlflow_logger,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
