# Run Status Artifact Initialization ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Seed workspace run status and publish an initial KFP artifact for dashboards.

Creates ``{workspace_path}/.automl/run_status.json`` with every pipeline component and stage from the manifest set to ``pending``, then copies that document into ``run_status_artifact`` so UIs can show progress before data loading starts.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `workspace_path` | `str` | `None` | PVC workspace directory for ``.automl/run_status.json``. |
| `pipeline_name` | `str` | `None` | KFP pipeline job resource name (``dsl.PIPELINE_JOB_RESOURCE_NAME_PLACEHOLDER``). |
| `run_id` | `str` | `None` | KFP run ID (``dsl.PIPELINE_JOB_ID_PLACEHOLDER``). |
| `run_status_pipeline_id` | `str` | `None` | Static ``@dsl.pipeline`` name; must match ``run_status_templates/pipelines/<name>.json`` (e.g. ``autogluon-tabular-training-pipeline``). |
| `run_status_artifact` | `dsl.Output[dsl.Artifact]` | `None` | Output artifact containing the initial ``run_status.json`` snapshot. |

## Outputs 📤

| Name | Type | Description |
| ---- | ---- | ----------- |
| Output | `None` |  |

## Metadata 🗂️

- **Name**: run_status_artifact_initialization
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
- **Tags**:
  - automl
  - run-status
- **Last Verified**: 2026-05-27 00:00:00+00:00
- **Owners**:
  - Approvers:
    - LukaszCmielowski
    - DorotaDR
  - Reviewers:
    - Mateusz-Switala
    - DorotaDR
