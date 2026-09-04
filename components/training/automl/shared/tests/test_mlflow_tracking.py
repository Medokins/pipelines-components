"""Unit tests for MLflow tracking helpers."""

import json
import os
import sys
from pathlib import Path
from unittest import mock

import pytest
from kfp_components.components.training.automl.shared import mlflow_tracking
from kfp_components.components.training.automl.shared.mlflow_tracking import (
    MLFLOW_WORKSPACE_HEADER,
    MlflowConfig,
    _metrics_for_task,
    _normalize_model_metrics,
    build_mlflow_run_url,
    build_mlflow_stage_map_block,
    configure_mlflow_client,
    display_model_run_name,
    is_mlflow_enabled,
    log_automl_results,
    parse_model_name,
    resolve_leaderboard_html_path,
    resolve_mlflow_config,
)


def _set_kfp_mlflow_config(
    monkeypatch,
    *,
    endpoint="https://mlflow.example.com",
    experiment_id="1",
    parent_run_id="parent-run",
    workspace="",
    workspaces_enabled=None,
    auth_type="",
    timeout="30s",
    extra=None,
):
    """Set KFP_MLFLOW_CONFIG to a JSON blob matching what the platform injects."""
    cfg = {
        "endpoint": endpoint,
        "experimentId": experiment_id,
        "parentRunId": parent_run_id,
        "authType": auth_type,
        "timeout": timeout,
    }
    if workspace:
        cfg["workspace"] = workspace
        cfg["workspacesEnabled"] = True if workspaces_enabled is None else workspaces_enabled
    elif workspaces_enabled is not None:
        cfg["workspacesEnabled"] = workspaces_enabled
    if extra:
        cfg.update(extra)
    monkeypatch.setenv("KFP_MLFLOW_CONFIG", json.dumps(cfg))
    return cfg


class TestMlflowTrackingHelpers:
    """Tests for MLflow env helpers and tracking artifact builders."""

    def test_is_mlflow_enabled_false_when_unset(self, monkeypatch):
        """Return False when KFP_MLFLOW_CONFIG is unset."""
        monkeypatch.delenv("KFP_MLFLOW_CONFIG", raising=False)
        assert is_mlflow_enabled() is False

    def test_is_mlflow_enabled_true_when_set(self, monkeypatch):
        """Return True when KFP_MLFLOW_CONFIG carries an endpoint."""
        _set_kfp_mlflow_config(monkeypatch)
        assert is_mlflow_enabled() is True

    def test_resolve_mlflow_config_parses_blob(self, monkeypatch):
        """Parse endpoint/parentRunId/experimentId/workspace/authType from the blob."""
        _set_kfp_mlflow_config(
            monkeypatch,
            endpoint="https://mlflow.example.com/mlflow",
            experiment_id="8",
            parent_run_id="6aae16e6",
            workspace="ns-automl-benchmarking",
            auth_type="kubernetes",
        )
        config = resolve_mlflow_config()
        assert config is not None
        assert config.mode == "kfp"
        assert config.tracking_uri == "https://mlflow.example.com/mlflow"
        assert config.experiment_id == "8"
        assert config.run_id == "6aae16e6"
        assert config.workspace == "ns-automl-benchmarking"
        assert config.auth_type == "kubernetes"

    def test_resolve_mlflow_config_none_when_absent(self, monkeypatch):
        """Return None when KFP_MLFLOW_CONFIG is not set."""
        monkeypatch.delenv("KFP_MLFLOW_CONFIG", raising=False)
        assert resolve_mlflow_config() is None

    def test_resolve_mlflow_config_none_when_invalid_json(self, monkeypatch):
        """Return None (not raise) when KFP_MLFLOW_CONFIG is malformed."""
        monkeypatch.setenv("KFP_MLFLOW_CONFIG", "{not json")
        assert resolve_mlflow_config() is None

    def test_resolve_mlflow_config_none_when_no_endpoint(self, monkeypatch):
        """Return None when the blob has no endpoint."""
        monkeypatch.setenv("KFP_MLFLOW_CONFIG", json.dumps({"parentRunId": "x"}))
        assert resolve_mlflow_config() is None

    def test_resolve_mlflow_config_ignores_workspace_when_disabled(self, monkeypatch):
        """Drop the workspace when workspacesEnabled is false."""
        _set_kfp_mlflow_config(monkeypatch, workspace="ns-automl-benchmarking", workspaces_enabled=False)
        config = resolve_mlflow_config()
        assert config is not None
        assert config.workspace == ""

    def test_build_mlflow_run_url(self):
        """Build a deep-link URL for the MLflow UI."""
        url = build_mlflow_run_url("https://mlflow.example.com/", "5", "abc123")
        assert url == "https://mlflow.example.com/#/experiments/5/runs/abc123"

    def test_build_mlflow_stage_map_block_disabled(self, monkeypatch):
        """Emit minimal mlflow block when tracking is disabled."""
        monkeypatch.delenv("KFP_MLFLOW_CONFIG", raising=False)
        block = build_mlflow_stage_map_block()
        assert block == {"tracking_enabled": False}

    def test_build_mlflow_stage_map_block_uri_only(self, monkeypatch):
        """Include tracking URI when only an endpoint (no ids/workspace) is available."""
        _set_kfp_mlflow_config(
            monkeypatch,
            experiment_id="",
            parent_run_id="",
        )
        block = build_mlflow_stage_map_block()
        assert block == {
            "tracking_enabled": True,
            "tracking_uri": "https://mlflow.example.com",
        }

    def test_build_mlflow_stage_map_block_full(self, monkeypatch):
        """Include MLflow IDs and workspace when the full blob is present."""
        _set_kfp_mlflow_config(
            monkeypatch,
            experiment_id="7",
            parent_run_id="parent-run",
            workspace="ds-project",
        )
        block = build_mlflow_stage_map_block()
        assert block == {
            "tracking_enabled": True,
            "tracking_uri": "https://mlflow.example.com",
            "experiment_id": "7",
            "run_id": "parent-run",
            "workspace": "ds-project",
            "run_url": "https://mlflow.example.com/#/experiments/7/runs/parent-run",
        }

    def test_configure_uses_native_set_workspace_when_available(self):
        """Prefer mlflow.set_workspace when the client exposes it."""
        mlflow = mock.Mock(spec=["set_tracking_uri", "set_workspace"])
        config = MlflowConfig(
            mode="kfp",
            tracking_uri="https://mlflow.example.com/mlflow",
            workspace="ns-automl-benchmarking",
        )
        configure_mlflow_client(mlflow, config)
        mlflow.set_tracking_uri.assert_called_once_with("https://mlflow.example.com/mlflow")
        mlflow.set_workspace.assert_called_once_with("ns-automl-benchmarking")

    def test_configure_injects_workspace_header_when_no_set_workspace(self, monkeypatch):
        """Fall back to the x-mlflow-workspace request header on plain MLflow."""
        import requests

        recorded: list[tuple[str, dict | None]] = []

        def fake_request(self, method, url, *args, **kwargs):
            recorded.append((url, kwargs.get("headers")))
            return "ok"

        monkeypatch.setattr(requests.Session, "request", fake_request)

        mlflow = mock.Mock(spec=["set_tracking_uri"])  # no set_workspace attribute
        config = MlflowConfig(
            mode="kfp",
            tracking_uri="https://mlflow.example.com/mlflow",
            workspace="ns-automl-benchmarking",
        )
        configure_mlflow_client(mlflow, config)

        session = requests.Session()
        session.request("GET", "https://mlflow.example.com/api/2.0/mlflow/experiments/search")
        session.request("GET", "https://s3.other.example.com/bucket/object")

        tracking_headers = recorded[0][1]
        assert tracking_headers is not None
        assert tracking_headers[MLFLOW_WORKSPACE_HEADER] == "ns-automl-benchmarking"

        other_host_headers = recorded[1][1] or {}
        assert MLFLOW_WORKSPACE_HEADER not in other_host_headers

    def test_configure_skips_workspace_when_empty(self):
        """No workspace handling when the workspace is empty."""
        mlflow = mock.Mock(spec=["set_tracking_uri", "set_workspace"])
        config = MlflowConfig(mode="kfp", tracking_uri="https://mlflow.example.com", workspace="")
        configure_mlflow_client(mlflow, config)
        mlflow.set_workspace.assert_not_called()

    def test_configure_kubernetes_auth_sets_token(self, monkeypatch, tmp_path):
        """Populate MLFLOW_TRACKING_TOKEN from the SA token file when authType=kubernetes."""
        token_file = tmp_path / "token"
        token_file.write_text("sa-token-value\n", encoding="utf-8")
        monkeypatch.setattr(mlflow_tracking, "SERVICE_ACCOUNT_TOKEN_PATH", str(token_file))
        monkeypatch.delenv("MLFLOW_TRACKING_TOKEN", raising=False)

        mlflow = mock.Mock(spec=["set_tracking_uri"])
        config = MlflowConfig(mode="kfp", tracking_uri="https://mlflow.example.com", auth_type="kubernetes")
        configure_mlflow_client(mlflow, config)

        assert os.environ["MLFLOW_TRACKING_TOKEN"] == "sa-token-value"

    def test_configure_kubernetes_auth_missing_token_does_not_raise(self, monkeypatch, tmp_path):
        """A missing SA token leaves auth unset without raising."""
        monkeypatch.setattr(mlflow_tracking, "SERVICE_ACCOUNT_TOKEN_PATH", str(tmp_path / "absent"))
        monkeypatch.delenv("MLFLOW_TRACKING_TOKEN", raising=False)

        mlflow = mock.Mock(spec=["set_tracking_uri"])
        config = MlflowConfig(mode="kfp", tracking_uri="https://mlflow.example.com", auth_type="kubernetes")
        configure_mlflow_client(mlflow, config)

        assert "MLFLOW_TRACKING_TOKEN" not in os.environ

    @pytest.mark.parametrize(
        ("model_name", "expected_type", "expected_level"),
        [
            ("WeightedEnsemble_L3_FULL", "WeightedEnsemble", 3),
            ("CatBoost_BAG_L1_FULL", "CatBoost", 1),
            ("Naive", "Naive", 1),
        ],
    )
    def test_parse_model_name(self, model_name, expected_type, expected_level):
        """Parse model family and stack level from AutoGluon model names."""
        model_type, stack_level = parse_model_name(model_name)
        assert model_type == expected_type
        assert stack_level == expected_level

    @pytest.mark.parametrize(
        ("model_name", "expected_display"),
        [
            ("WeightedEnsemble_L3_FULL", "WeightedEnsemble_L3"),
            ("CatBoost_BAG_L1_FULL", "CatBoost_BAG_L1"),
            ("Naive", "Naive"),
        ],
    )
    def test_display_model_run_name(self, model_name, expected_display):
        """Strip refit suffix from MLflow child run names."""
        assert display_model_run_name(model_name) == expected_display

    def test_normalize_model_metrics_flattens_test_data(self):
        """Flatten nested test_data metrics from artifact metadata."""
        payload = {"test_data": {"accuracy": 0.91, "f1": 0.88}}
        assert _normalize_model_metrics(payload) == {"accuracy": 0.91, "f1": 0.88}

    def test_metrics_for_task_binary(self):
        """Log task-specific classification metrics on child runs."""
        metrics = _metrics_for_task(
            "binary",
            {"accuracy": 0.91, "f1": 0.88, "roc_auc": 0.95, "fit_time": 12.0},
        )
        assert metrics == {"accuracy": 0.91, "f1": 0.88, "roc_auc": 0.95}

    def test_resolve_leaderboard_html_path_file(self, tmp_path):
        """Resolve a direct HTML file path."""
        html_file = tmp_path / "leaderboard.html"
        html_file.write_text("<html></html>", encoding="utf-8")
        assert resolve_leaderboard_html_path(html_file) == html_file

    def test_resolve_leaderboard_html_path_directory(self, tmp_path):
        """Resolve HTML inside a KFP artifact directory."""
        artifact_dir = tmp_path / "html_artifact"
        artifact_dir.mkdir()
        html_file = artifact_dir / "index.html"
        html_file.write_text("<html></html>", encoding="utf-8")
        assert resolve_leaderboard_html_path(artifact_dir) == html_file


def _make_models_artifact(base_path: Path, model_names: list[str], *, uri: str = "s3://bucket/models"):
    artifact = mock.MagicMock()
    artifact.path = str(base_path)
    artifact.uri = uri
    artifact.metadata = {
        "model_names": json.dumps(model_names),
        "context": {"task_type": "binary"},
    }
    return artifact


def _mock_run_context(run_id: str, experiment_id: str = "1") -> mock.MagicMock:
    ctx = mock.MagicMock()
    ctx.info.run_id = run_id
    ctx.info.experiment_id = experiment_id
    ctx.__enter__ = mock.Mock(return_value=ctx)
    ctx.__exit__ = mock.Mock(return_value=False)
    return ctx


def _write_model_metrics(base_path: Path, model_name: str, metrics: dict) -> Path:
    metrics_dir = base_path / model_name / "metrics"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return metrics_dir


class TestLogAutomlResults:
    """Tests for end-of-run MLflow logging."""

    def test_skips_when_mlflow_disabled(self, tmp_path, monkeypatch):
        """Return False without calling MLflow when tracking is disabled."""
        monkeypatch.delenv("KFP_MLFLOW_CONFIG", raising=False)
        models_artifact = _make_models_artifact(tmp_path, ["Model_FULL"])
        logged, tracking_info = log_automl_results(
            models_artifact=models_artifact,
            html_artifact_path=tmp_path / "leaderboard.html",
            eval_metric="accuracy",
            pipeline_name="autogluon-tabular-training-pipeline",
            kfp_run_id="run-1",
        )
        assert logged is False
        assert tracking_info == {}

    def test_logs_parent_and_child_runs_kfp_mode(self, tmp_path, monkeypatch):
        """Create parent and nested child MLflow runs in KFP mode."""
        _set_kfp_mlflow_config(monkeypatch, parent_run_id="parent-run", experiment_id="1")

        model_name = "LightGBM_BAG_L1_FULL"
        metrics_dir = _write_model_metrics(tmp_path, model_name, {"accuracy": 0.91, "f1": 0.88})
        (metrics_dir / "confusion_matrix.json").write_text("{}", encoding="utf-8")
        (tmp_path / "leaderboard.html").write_text("<html></html>", encoding="utf-8")

        mock_mlflow = mock.MagicMock()
        parent_ctx = _mock_run_context("parent-run", "1")
        child_ctx = _mock_run_context("child-run-1", "1")
        mock_mlflow.start_run.side_effect = [parent_ctx, child_ctx]
        mock_mlflow.active_run.side_effect = [parent_ctx, child_ctx, parent_ctx]
        mock_mlflow.entities.RunTag = mock.Mock(side_effect=lambda key, value: (key, value))

        models_artifact = _make_models_artifact(tmp_path, [model_name])
        with mock.patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            logged, tracking_info = log_automl_results(
                models_artifact=models_artifact,
                html_artifact_path=tmp_path / "leaderboard.html",
                eval_metric="accuracy",
                pipeline_name="autogluon-tabular-training-pipeline",
                kfp_run_id="run-1",
                task_type="binary",
                preset="speed",
                top_n=1,
            )

        assert logged is True
        assert tracking_info["tracking_mode"] == "kfp"
        assert tracking_info["mlflow_child_run_count"] == "1"
        assert tracking_info["mlflow_child_run_ids"] == "child-run-1"
        mock_mlflow.start_run.assert_any_call(run_id="parent-run")
        mock_mlflow.start_run.assert_any_call(run_name="LightGBM_BAG_L1", nested=True)
        mock_mlflow.set_tags.assert_called()
        mock_mlflow.log_params.assert_called()
        mock_mlflow.log_metrics.assert_called()
        mock_mlflow.log_artifact.assert_called()

    def test_logs_multiple_child_runs(self, tmp_path, monkeypatch):
        """Create one child MLflow run per model under the parent run."""
        _set_kfp_mlflow_config(monkeypatch, parent_run_id="parent-run", experiment_id="99")

        model_names = ["WeightedEnsemble_L3_FULL", "CatBoost_BAG_L1_FULL"]
        for model_name in model_names:
            _write_model_metrics(tmp_path, model_name, {"accuracy": 0.9, "f1": 0.88, "roc_auc": 0.95})
        (tmp_path / "leaderboard.html").write_text("<html></html>", encoding="utf-8")

        mock_mlflow = mock.MagicMock()
        parent_ctx = _mock_run_context("parent-run", "99")
        child_ctx_1 = _mock_run_context("child-1", "99")
        child_ctx_2 = _mock_run_context("child-2", "99")
        mock_mlflow.start_run.side_effect = [parent_ctx, child_ctx_1, child_ctx_2]
        mock_mlflow.active_run.side_effect = [parent_ctx, child_ctx_1, child_ctx_2, parent_ctx, parent_ctx]
        mock_mlflow.entities.RunTag = mock.Mock(side_effect=lambda key, value: (key, value))

        models_artifact = _make_models_artifact(tmp_path, model_names)
        with mock.patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            logged, tracking_info = log_automl_results(
                models_artifact=models_artifact,
                html_artifact_path=tmp_path / "leaderboard.html",
                eval_metric="accuracy",
                pipeline_name="autogluon-tabular-training-pipeline",
                kfp_run_id="run-1",
                task_type="binary",
            )

        assert logged is True
        assert tracking_info["mlflow_child_run_count"] == "2"
        assert tracking_info["mlflow_experiment_id"] == "99"
        mock_mlflow.start_run.assert_any_call(run_id="parent-run")
        mock_mlflow.start_run.assert_any_call(run_name="WeightedEnsemble_L3", nested=True)
        mock_mlflow.start_run.assert_any_call(run_name="CatBoost_BAG_L1", nested=True)

    def test_uploads_model_and_notebook_artifacts(self, tmp_path, monkeypatch):
        """Upload the predictor dir (model.pkl) and notebook per child run."""
        _set_kfp_mlflow_config(monkeypatch, parent_run_id="parent-run", experiment_id="1")

        model_name = "LightGBM_BAG_L1_FULL"
        _write_model_metrics(tmp_path, model_name, {"accuracy": 0.91})
        predictor_dir = tmp_path / model_name / "predictor"
        predictor_dir.mkdir(parents=True)
        (predictor_dir / "model.pkl").write_bytes(b"payload")
        notebook_dir = tmp_path / model_name / "notebooks"
        notebook_dir.mkdir(parents=True)
        (notebook_dir / "automl_predictor_notebook.ipynb").write_text("{}", encoding="utf-8")
        (tmp_path / "leaderboard.html").write_text("<html></html>", encoding="utf-8")

        mock_mlflow = mock.MagicMock()
        parent_ctx = _mock_run_context("parent-run", "1")
        child_ctx = _mock_run_context("child-run-1", "1")
        mock_mlflow.start_run.side_effect = [parent_ctx, child_ctx]
        mock_mlflow.active_run.side_effect = [parent_ctx, child_ctx, parent_ctx]
        mock_mlflow.entities.RunTag = mock.Mock(side_effect=lambda key, value: (key, value))

        models_artifact = _make_models_artifact(tmp_path, [model_name])
        with mock.patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            logged, _ = log_automl_results(
                models_artifact=models_artifact,
                html_artifact_path=tmp_path / "leaderboard.html",
                eval_metric="accuracy",
                pipeline_name="autogluon-tabular-training-pipeline",
                kfp_run_id="run-1",
                task_type="binary",
                log_model_artifacts=True,
            )

        assert logged is True
        assert any(call.kwargs.get("artifact_path") == "model" for call in mock_mlflow.log_artifacts.call_args_list)
        assert any(call.kwargs.get("artifact_path") == "notebooks" for call in mock_mlflow.log_artifact.call_args_list)

    def test_registers_best_model_and_sets_target_stage(self, tmp_path, monkeypatch):
        """Register the best model and tag its version with target_stage."""
        _set_kfp_mlflow_config(monkeypatch, parent_run_id="parent-run", experiment_id="1")

        model_name = "LightGBM_BAG_L1_FULL"
        _write_model_metrics(tmp_path, model_name, {"accuracy": 0.91})
        (tmp_path / "leaderboard.html").write_text("<html></html>", encoding="utf-8")

        mock_mlflow = mock.MagicMock()
        parent_ctx = _mock_run_context("parent-run", "1")
        child_ctx = _mock_run_context("child-run-1", "1")
        mock_mlflow.start_run.side_effect = [parent_ctx, child_ctx]
        mock_mlflow.active_run.side_effect = [parent_ctx, child_ctx, parent_ctx]
        mock_mlflow.entities.RunTag = mock.Mock(side_effect=lambda key, value: (key, value))
        mock_mlflow.register_model.return_value = mock.Mock(version="3")
        mock_client = mock.MagicMock()
        mock_mlflow.MlflowClient.return_value = mock_client

        models_artifact = _make_models_artifact(tmp_path, [model_name])
        with mock.patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            logged, tracking_info = log_automl_results(
                models_artifact=models_artifact,
                html_artifact_path=tmp_path / "leaderboard.html",
                eval_metric="accuracy",
                pipeline_name="autogluon-tabular-training-pipeline",
                kfp_run_id="run-1",
                task_type="binary",
                register_best_model=True,
                model_registry_name="my-model",
                target_stage="staging",
            )

        assert logged is True
        mock_mlflow.register_model.assert_called_once_with("runs:/child-run-1/model", "my-model")
        mock_client.set_model_version_tag.assert_called_once_with(
            name="my-model",
            version="3",
            key="target_stage",
            value="staging",
        )
        assert tracking_info["mlflow_registered_model"] == "my-model"
        assert tracking_info["mlflow_model_version"] == "3"
        assert tracking_info["mlflow_target_stage"] == "staging"

    def test_returns_false_when_mlflow_api_fails(self, tmp_path, monkeypatch):
        """Do not raise when the MLflow API call fails."""
        _set_kfp_mlflow_config(monkeypatch, parent_run_id="parent-run", experiment_id="1")

        model_name = "LightGBM_BAG_L1_FULL"
        _write_model_metrics(tmp_path, model_name, {"accuracy": 0.91})
        (tmp_path / "leaderboard.html").write_text("<html></html>", encoding="utf-8")

        mock_mlflow = mock.MagicMock()
        mock_mlflow.start_run.side_effect = RuntimeError(
            '{"status": "Failure", "reason": "NotAcceptable", "code": 406}'
        )

        models_artifact = _make_models_artifact(tmp_path, [model_name])
        with mock.patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            logged, tracking_info = log_automl_results(
                models_artifact=models_artifact,
                html_artifact_path=tmp_path / "leaderboard.html",
                eval_metric="accuracy",
                pipeline_name="autogluon-tabular-training-pipeline",
                kfp_run_id="run-1",
            )

        assert logged is False
        assert "error" in tracking_info
        assert tracking_info["tracking_mode"] == "kfp"
