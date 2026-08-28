# Automl Mlflow Logger ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Log AutoML experiment results to MLflow at the end of the pipeline run.

Expects ``MLFLOW_*`` environment variables from the ``mlflow_connection_secret_name`` pipeline parameter (mounted on this step only). When ``MLFLOW_TRACKING_URI`` is unset, logging is skipped and the step completes successfully.

Each refitted model becomes a nested child run under the parent experiment run, logging task-specific metrics, params, metric-JSON artifacts, rendered plots (confusion matrix, ROC curves, back-testing) and — when ``log_model_artifacts`` — its predictor (model.pkl) and filled notebook. When
``register_best_model`` and ``model_registry_name`` are set, the best model is registered in the MLflow Model Registry and its version tagged with ``target_stage``.

``mlflow_tracking.py`` is embedded at compile time when the runtime image does not yet ship ``kfp_components...mlflow_tracking``.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `models_artifact` | `dsl.Input[dsl.Model]` | `None` | Combined models artifact from training with ``metadata["model_names"]``. |
| `html_artifact` | `dsl.Input[dsl.HTML]` | `None` | Leaderboard HTML artifact from the training component. |
| `eval_metric` | `str` | `None` | Metric used for ranking (e.g. ``accuracy``, ``MASE``). |
| `pipeline_name` | `str` | `None` | Stable pipeline name used as the MLflow experiment name (and tags) when the connection does not set ``MLFLOW_EXPERIMENT_NAME``. Pass the pipeline's logical name (e.g. ``PIPELINE_NAME``) so runs aggregate under one experiment. |
| `run_id` | `str` | `None` | KFP run ID (from ``dsl.PIPELINE_JOB_ID_PLACEHOLDER``). |
| `task_type` | `str` | `None` | ML task type (``binary``, ``multiclass``, ``regression``, or ``time_series``). |
| `component_status` | `dsl.Output[dsl.Artifact]` | `None` | Output artifact with stage progress (``component_status.json``). |
| `run_name` | `str` | `""` | Per-execution MLflow parent run name. Pass the run's display name (from ``dsl.PIPELINE_JOB_NAME_PLACEHOLDER``) so the MLflow run matches the OpenShift run name. Falls back to ``pipeline_name`` when empty. |
| `preset` | `str` | `speed` | Training quality preset logged on the parent run. |
| `top_n` | `int` | `3` | Number of top models logged on the parent run. |
| `log_model_artifacts` | `bool` | `True` | When True, upload each model's predictor (model.pkl) and notebook. |
| `register_best_model` | `bool` | `False` | When True, register the best model in the MLflow Model Registry. |
| `model_registry_name` | `str` | `""` | Registered-model name to use when ``register_best_model`` is True. |
| `target_stage` | `str` | `""` | Optional deployment-stage value set as a ``target_stage`` tag on the registered best-model version. |
| `embedded_artifact` | `dsl.EmbeddedInput[dsl.Dataset]` | `None` | Embedded ``mlflow_tracking.py`` helper injected by KFP at compile time. |

## Outputs 📤

| Name | Type | Description |
| ---- | ---- | ----------- |
| Output | `None` |  |

## Metadata 🗂️

- **Name**: automl_mlflow_logger
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: MLflow, Version: >=2.9.0
- **Tags**:
  - automl
  - mlflow
- **Last Verified**: 2026-08-26 00:00:00+00:00
- **Owners**:
  - No Parent Owners: Yes
  - Approvers:
    - LukaszCmielowski
    - DorotaDR
  - Reviewers:
    - Mateusz-Switala
    - DorotaDR
