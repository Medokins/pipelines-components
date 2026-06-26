# Automl Mlflow Logger

> **Stability: alpha** — This asset is not yet stable and may change.

## Overview

Log AutoML experiment results to MLflow at the end of the pipeline run.

Creates a **parent run** for the pipeline execution and **child runs** for each refitted
model (for example ``WeightedEnsemble_L3``, ``CatBoost_BAG_L1``). Child runs log
task-specific metrics (``accuracy``, ``f1``, ``roc_auc`` for binary classification) and
model parameters. Open the parent run in the MLflow UI to see nested child runs.

The logger reads ``MLFLOW_*`` environment variables mounted from the
``mlflow_connection_secret_name`` pipeline parameter. When ``MLFLOW_TRACKING_URI`` is unset,
logging is skipped and the step completes successfully. API failures are recorded in
``mlflow_tracking.json`` (``mlflow_error``) and do not fail the pipeline run.

## Prerequisites

Complete these **once per Data Science project** before running the pipeline.

### 1. MLflow tracking secret

Create a Kubernetes secret in the project namespace (same namespace as pipeline runs):

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: mlflow-connection
  namespace: <project-namespace>
type: Opaque
stringData:
  MLFLOW_TRACKING_URI: "https://<dashboard-host>/mlflow"
  MLFLOW_TRACKING_AUTH: "kubernetes-namespaced"
  MLFLOW_WORKSPACE: "<project-namespace>"
  MLFLOW_EXPERIMENT_NAME: "automl-experiments"
```

```bash
oc apply -f mlflow-connection-secret.yaml
```

Optional keys: ``MLFLOW_TRACKING_TOKEN``, ``MLFLOW_TRACKING_INSECURE_TLS``.

### 2. MLflow artifact storage (recommended)

Create an S3-compatible connection secret named ``mlflow-artifact-connection`` in the project
(can reuse the same bucket/credentials as training data). Add an ``MLflowConfig`` CR named
``mlflow`` in the project that references this secret so MLflow can store artifacts
(leaderboard HTML, metrics JSON).

See [RHOAI MLflow documentation](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.5/html/working_with_mlflow/installing-and-authenticating-mlflow-sdk_mlflow).

### 3. Pipeline service account RBAC

The pipeline step pods must be allowed to create/read experiments in ``MLFLOW_WORKSPACE``.
Without this, the logger may log ``PERMISSION_DENIED`` even when ``MLFLOW_TRACKING_URI=set``.

Grant the pipeline service account MLflow workspace permissions per RHOAI MLflow RBAC docs,
or add ``MLFLOW_TRACKING_TOKEN`` to the secret as a dev workaround.

### 4. Custom ``odh-automl`` runtime image

The AutoML image used at **compile time** (``RELATED_IMAGE_ODH_AUTOML_IMAGE``) must include:

- ``mlflow`` with Kubernetes auth support
- ``kfp_components`` from this repository (including ``shared/mlflow_tracking.py``)

### 5. Compile and upload

```bash
RELATED_IMAGE_ODH_AUTOML_IMAGE=quay.io/opendatahub/odh-automl@sha256:<digest> \
  uv run python pipelines/training/automl/autogluon_tabular_training_pipeline/pipeline.py
```

Upload the generated ``pipeline.yaml`` to the pipeline UI.

## Run the pipeline

When starting a run, set:

| Parameter | Example | Purpose |
| --- | --- | --- |
| ``train_data_secret_name`` | ``aws-connection-data-storage`` | S3 credentials for training CSV |
| ``mlflow_connection_secret_name`` | ``mlflow-connection`` | MLflow tracking env vars (logger step only) |
| ``train_data_bucket_name`` / ``train_data_file_key`` | your bucket and CSV path | Training data location |

Results appear in the MLflow UI at ``MLFLOW_TRACKING_URI``, workspace ``MLFLOW_WORKSPACE``,
experiment ``MLFLOW_EXPERIMENT_NAME``.

Verify in the ``automl-mlflow-logger`` pod logs: ``MLFLOW_TRACKING_URI=set`` and
``MLflow logging completed``.

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
| `mlflow_tracking_artifact` | `dsl.Output[dsl.Artifact]` | — | Final MLflow tracking metadata JSON. |
| `preset` | `str` | `speed` | Training preset logged on the parent run. |
| `top_n` | `int` | `3` | Number of top models logged on the parent run. |

## Troubleshooting

### `ModuleNotFoundError: mlflow_tracking`

The runtime image may include ``mlflow`` but not yet ship ``kfp_components...mlflow_tracking``.
Recompile the pipeline from this repository — the component embeds ``mlflow_tracking.py`` at compile
time so the pod does not depend on a newer runtime image for that module.

### `NotAcceptable` / HTTP 406 in the KFP UI

This Kubernetes API error in the UI is often **not** the root cause. Check the logger pod logs for
the Python traceback (for example ``ModuleNotFoundError`` or MLflow API errors).

### `env | grep MLFLOW` is empty in the logger pod

The secret is mounted only on the MLflow logger step. Set ``mlflow_connection_secret_name``
when creating the run (for example ``mlflow-connection``).

```bash
oc get secret mlflow-connection -n <project>
oc exec -n <project> <automl-mlflow-logger-pod> -c main -- env | grep MLFLOW
```

### `PERMISSION_DENIED` from MLflow

The pipeline reached MLflow but the pipeline service account lacks workspace RBAC. Grant
experiment permissions in ``MLFLOW_WORKSPACE`` or use ``MLFLOW_TRACKING_TOKEN`` in the secret.

### S3 `NoSuchKey` when opening `mlflow_tracking_artifact` in the KFP UI

The KFP UI may request the artifact prefix rather than ``mlflow_tracking.json``. Check run
artifact metadata (``mlflow_run_id``, ``mlflow_error``) or open the run directly in MLflow.

## Metadata

- **Name**: automl_mlflow_logger
- **Stability**: alpha
- **Tags**: automl, mlflow
