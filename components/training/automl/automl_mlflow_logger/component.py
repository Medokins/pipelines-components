from kfp import dsl
from kfp_components.utils.consts import AUTOML_IMAGE  # pyright: ignore[reportMissingImports]


@dsl.component(
    base_image=AUTOML_IMAGE,  # noqa: E501
)
def automl_mlflow_logger(
    models_artifact: dsl.Input[dsl.Model],
    html_artifact: dsl.Input[dsl.HTML],
    eval_metric: str,
    pipeline_name: str,
    run_id: str,
    task_type: str,
    component_status: dsl.Output[dsl.Artifact],
    preset: str = "speed",
    top_n: int = 3,
) -> None:
    """Log AutoML experiment results to MLflow at the end of the pipeline run.

    When ``MLFLOW_TRACKING_URI`` is set by the KFP runtime (RHOAI MLflow integration),
    resumes the KFP-managed parent run and creates nested child runs per refitted model.
    Logs aggregate metrics, leaderboard HTML, and per-model metrics/artifacts. When MLflow
    is not configured, this component completes successfully without side effects.

    Args:
        models_artifact: Combined models artifact from training with ``metadata["model_names"]``.
        html_artifact: Leaderboard HTML artifact from ``leaderboard_evaluation``.
        eval_metric: Metric used for ranking (e.g. ``accuracy``, ``MASE``).
        pipeline_name: Pipeline name for MLflow tags (from ``dsl.PIPELINE_JOB_RESOURCE_NAME_PLACEHOLDER``).
        run_id: KFP run ID (from ``dsl.PIPELINE_JOB_ID_PLACEHOLDER``).
        task_type: ML task type (``binary``, ``multiclass``, ``regression``, or ``time_series``).
        component_status: Output artifact with stage progress (``component_status.json``).
        preset: Training quality preset logged on the parent run.
        top_n: Number of top models logged on the parent run.

    Raises:
        TypeError: If required string parameters are empty.
        ValueError: If ``top_n`` is not positive.
    """
    import logging

    from kfp_components.components.training.automl.shared.component_status import ComponentStatusTracker
    from kfp_components.components.training.automl.shared.mlflow_tracking import log_automl_results

    logger = logging.getLogger(__name__)

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

        logged = log_automl_results(
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

        if logged:
            logger.info("MLflow logging completed for pipeline run %s.", run_id)
            status.record("log_mlflow_results", "completed", tracking_enabled=True)
        else:
            logger.info("MLflow logging skipped for pipeline run %s.", run_id)
            status.record("log_mlflow_results", "completed", tracking_enabled=False)

        component_status.metadata["display_name"] = "MLflow Logging Status"
