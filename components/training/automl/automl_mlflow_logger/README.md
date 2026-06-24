# Automl Mlflow Logger

> **Stability: alpha** — This asset is not yet stable and may change.

## Overview

Log AutoML experiment results to MLflow at the end of the pipeline run.

MLflow configuration is resolved automatically in this order:

1. **KFP/RHOAI platform** — `MLFLOW_TRACKING_URI` and `MLFLOW_RUN_ID` injected by the cluster.
2. **User connection secret** — fixed secret name `mlflow-connection` mounted into the MLflow logger step.
3. **Disabled** — no tracking URI available; the step completes successfully.

MLflow logging is **optional**: API failures are recorded in `mlflow_tracking.json` (`mlflow_error`) and do not fail the pipeline run.

## Cluster setup (before running the pipeline)

### Prerequisites

- MLflow installed and reachable from the Data Science project namespace.
- `mlflow[kubernetes]>=3.11` in the AutoML runtime image (`odh-automl`).
- Pipeline service account has RBAC to log experiments in the target MLflow workspace.

### Option A: RHOAI KFP MLflow integration (recommended)

Enable MLflow integration for the Data Science project in OpenShift AI. KFP injects:

- `MLFLOW_TRACKING_URI`
- `MLFLOW_RUN_ID` (parent run per pipeline execution)
- `MLFLOW_EXPERIMENT_ID`
- `MLFLOW_WORKSPACE`
- `MLFLOW_TRACKING_AUTH=kubernetes-namespaced`

Leave the `mlflow-connection` secret unset when using this mode (platform env vars take precedence).

### Option B: User-provided MLflow connection secret

Create a secret in the **same namespace** as the pipeline run:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: mlflow-connection
  namespace: my-ds-project
type: Opaque
stringData:
  MLFLOW_TRACKING_URI: "https://<dashboard-host>/mlflow"
  MLFLOW_TRACKING_AUTH: "kubernetes-namespaced"
  MLFLOW_WORKSPACE: "my-ds-project"
  MLFLOW_EXPERIMENT_NAME: "automl-experiments"
```

Apply:

```bash
oc apply -f mlflow-connection-secret.yaml
```

The pipeline mounts `mlflow-connection` automatically on the MLflow logger step.

#### Optional secret keys

| Key | Purpose |
| --- | ------- |
| `MLFLOW_TRACKING_TOKEN` | Manual bearer token (dev/CI); omit when using `kubernetes-namespaced` |
| `MLFLOW_TRACKING_INSECURE_TLS` | Set to `true` for clusters with untrusted TLS |

### Artifact storage (recommended for connection mode)

Create `MLflowConfig` and `mlflow-artifact-connection` in the project so artifact uploads work.
See [RHOAI MLflow SDK documentation](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.5/html/working_with_mlflow/installing-and-authenticating-mlflow-sdk_mlflow).

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

## MLflow connection secret

The pipeline mounts a Kubernetes secret named **`mlflow-connection`** into the MLflow-related steps.
The name is fixed (not a pipeline parameter) due to a Kubeflow Pipelines limitation with
conditional secret mounts.

Create the secret in your project namespace before running (see cluster setup below).

## Troubleshooting

### `ModuleNotFoundError: mlflow_tracking`

The stable `odh-automl` image may not yet ship `kfp_components...mlflow_tracking`. Recompile the
pipeline after pulling the latest `automl_mlflow_logger` component — it embeds `mlflow_tracking.py`
at compile time so the pod does not depend on a newer runtime image for that module.

If you see `AttributeError: 'NoneType' object has no attribute '__dict__'` while loading the
embedded module, update to the latest component code (registers the module in `sys.modules` before
`exec_module`, which `@dataclass` requires).

### `env | grep MLFLOW` is empty in the logger pod

The `mlflow-connection` secret is mounted only on the MLflow logger step via compiled pipeline
`platforms.kubernetes` metadata. If the logger pod has no `MLFLOW_*` variables, MLflow logging is
skipped and nothing appears in the MLflow UI.

Verify:

1. Secret exists: `oc get secret mlflow-connection -n <project>`
2. Compiled `pipeline.yaml` includes `secretAsEnv` under `exec-automl-mlflow-logger` /
   `exec-automl-mlflow-logger-2`
3. Check inside the **running logger pod**, not your local shell:

```bash
oc exec -n <project> <automl-mlflow-logger-pod> -c main -- env | grep MLFLOW
```

**Recommended:** enable RHOAI 3.5+ project MLflow integration so KFP injects `MLFLOW_RUN_ID`
(platform parent run). The connection secret alone creates a separate parent run in connection mode.

### S3 `NoSuchKey` when opening `mlflow_tracking_artifact` in the KFP UI

The artifact is a directory prefix in S3. The component writes `mlflow_tracking.json` and `data`
under that prefix. The KFP UI may request the prefix itself and show `NoSuchKey` even when files
exist. Check artifact metadata on the run (``mlflow_run_id``, ``mlflow_error``, etc.) or list S3:

```bash
aws s3 ls s3://<bucket>/<pipeline>/<run-id>/automl-mlflow-logger-2/<pod-id>/mlflow_tracking_artifact/
```

### `NotAcceptable` / HTTP 406 with Kubernetes `Status` JSON

This is **not** MLflow rejecting the HTML leaderboard. It is a **Kubernetes API** error body,
usually returned when the MLflow client cannot reach a valid MLflow tracking endpoint.

Check:

1. **`MLFLOW_TRACKING_URI`** must be the MLflow UI/tracking URL (e.g. `https://<dashboard>/mlflow`), not a Kubernetes API URL (`https://kubernetes.default.svc` or `/api/v1/...`).
2. **RHOAI 3.5+ KFP integration (recommended):** enable project MLflow integration so KFP injects `MLFLOW_RUN_ID` and the logger **resumes the platform parent run** (see ADR implementation guidance).
3. **Connection secret mode:** requires `mlflow[kubernetes]` in the `odh-automl` image and RBAC for the pipeline service account in the MLflow workspace.
4. **Pod logs** on `automl-mlflow-logger` / `automl-mlflow-logger-2` for the full Python traceback.

After a failed MLflow API call, the pipeline step should still complete and write
`mlflow_tracking.json` with an `mlflow_error` field when using the latest component code.

### `PERMISSION_DENIED` from MLflow (`kubernetes-namespaced` auth)

Example from the logger pod:

```text
mlflow_error=INTERNAL_ERROR: Response: {'error': {'code': 'PERMISSION_DENIED', ...}}
```

The pipeline reached MLflow (`MLFLOW_TRACKING_URI=set`) but the **pipeline service account**
is not authorized to create or read experiments in workspace `MLFLOW_WORKSPACE`.

**Fix (pick one):**

1. **RHOAI 3.5+ KFP MLflow integration (recommended):** enable MLflow for the Data Science
   project in OpenShift AI. KFP injects `MLFLOW_RUN_ID` and provisions a parent run with the
   correct permissions. Remove or stop relying on the connection secret for URI/auth.

2. **Grant MLflow workspace RBAC to the pipeline service account** (connection-secret mode):
   find the SA on a logger pod and grant it experiment permissions in workspace
   `ns-automl-benchmarking` per
   [RHOAI MLflow SDK / RBAC docs](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.5/html/working_with_mlflow/installing-and-authenticating-mlflow-sdk_mlflow).

   ```bash
   # Service account used by pipeline step pods (name may vary)
   oc get pod -n ns-automl-benchmarking <automl-mlflow-logger-impl-pod> \
     -o jsonpath='{.spec.serviceAccountName}{"\n"}'
   ```

3. **Dev workaround:** add `MLFLOW_TRACKING_TOKEN` to `mlflow-connection` with a token from a
   user/service account that already has MLflow access (omit `kubernetes-namespaced` auth flow).

Also ensure `MLflowConfig` + `mlflow-artifact-connection` exist in the project if you expect
artifact uploads (leaderboard HTML, metrics JSON) in addition to params/metrics.

## Metadata

- **Name**: automl_mlflow_logger
- **Stability**: alpha
- **Tags**: automl, mlflow
