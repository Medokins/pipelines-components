# Automl Mlflow Logger

> **Stability: alpha** — This asset is not yet stable and may change.

## Overview

Log AutoML experiment results to MLflow at the end of the pipeline run.

When `MLFLOW_TRACKING_URI` is set by the KFP runtime (RHOAI MLflow integration), resumes the KFP-managed parent run and creates nested child runs per refitted model. When MLflow is not configured, this component completes successfully without side effects.

## Inputs

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `models_artifact` | `dsl.Input[dsl.Model]` | — | Combined models artifact from training. |
| `html_artifact` | `dsl.Input[dsl.HTML]` | — | Leaderboard HTML from `leaderboard_evaluation`. |
| `eval_metric` | `str` | — | Metric used for ranking. |
| `pipeline_name` | `str` | — | Pipeline name for MLflow tags. |
| `run_id` | `str` | — | KFP run ID. |
| `task_type` | `str` | — | `binary`, `multiclass`, `regression`, or `time_series`. |
| `component_status` | `dsl.Output[dsl.Artifact]` | — | Stage progress artifact. |
| `preset` | `str` | `speed` | Training preset logged on the parent run. |
| `top_n` | `int` | `3` | Number of top models logged on the parent run. |

## Metadata

- **Name**: automl_mlflow_logger
- **Stability**: alpha
- **Tags**: automl, mlflow
