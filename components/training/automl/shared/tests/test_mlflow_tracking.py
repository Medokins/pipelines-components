"""Unit tests for MLflow tracking helpers."""

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

from kfp_components.components.training.automl.shared.mlflow_tracking import (
    MLFLOW_CONNECTION_SECRET_KEY_TO_ENV,
    build_mlflow_run_url,
    build_mlflow_stage_map_block,
    display_model_run_name,
    is_mlflow_enabled,
    log_automl_results,
    parse_model_name,
    resolve_leaderboard_html_path,
    resolve_mlflow_config,
    _metrics_for_task,
    _normalize_model_metrics,
)


class TestMlflowTrackingHelpers:
    """Tests for MLflow env helpers and tracking artifact builders."""

    def test_connection_secret_key_mapping_includes_tracking_uri(self):
        """Secret mount mapping exposes MLFLOW_TRACKING_URI."""
        assert "MLFLOW_TRACKING_URI" in MLFLOW_CONNECTION_SECRET_KEY_TO_ENV

    def test_is_mlflow_enabled_false_when_unset(self, monkeypatch):
        """Return False when MLFLOW_TRACKING_URI is unset."""
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        assert is_mlflow_enabled() is False

    def test_is_mlflow_enabled_true_when_set(self, monkeypatch):
        """Return True when MLFLOW_TRACKING_URI is set."""
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.com")
        assert is_mlflow_enabled() is True

    def test_resolve_mlflow_config_kfp_mode(self, monkeypatch):
        """Prefer KFP mode when MLFLOW_RUN_ID is present."""
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.com")
        monkeypatch.setenv("MLFLOW_RUN_ID", "parent-run")
        config = resolve_mlflow_config()
        assert config is not None
        assert config.mode == "kfp"
        assert config.run_id == "parent-run"

    def test_resolve_mlflow_config_connection_mode(self, monkeypatch):
        """Use connection mode when URI is set without MLFLOW_RUN_ID."""
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.com")
        monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
        monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "automl-experiments")
        config = resolve_mlflow_config()
        assert config is not None
        assert config.mode == "connection"
        assert config.experiment_name == "automl-experiments"

    def test_build_mlflow_run_url(self):
        """Build a deep-link URL for the MLflow UI."""
        url = build_mlflow_run_url("https://mlflow.example.com/", "5", "abc123")
        assert url == "https://mlflow.example.com/#/experiments/5/runs/abc123"

    def test_build_mlflow_stage_map_block_disabled(self, monkeypatch):
        """Emit minimal mlflow block when tracking is disabled."""
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        block = build_mlflow_stage_map_block()
        assert block == {"tracking_enabled": False}

    def test_build_mlflow_stage_map_block_connection_uri_only(self, monkeypatch):
        """Include tracking URI when only MLFLOW_TRACKING_URI is available."""
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.com")
        monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
        block = build_mlflow_stage_map_block()
        assert block == {
            "tracking_enabled": True,
            "tracking_uri": "https://mlflow.example.com",
        }

    def test_build_mlflow_stage_map_block_kfp_mode(self, monkeypatch):
        """Include MLflow IDs when tracking is enabled in KFP mode."""
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.com")
        monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", "7")
        monkeypatch.setenv("MLFLOW_RUN_ID", "parent-run")
        monkeypatch.setenv("MLFLOW_WORKSPACE", "ds-project")
        block = build_mlflow_stage_map_block()
        assert block == {
            "tracking_enabled": True,
            "tracking_uri": "https://mlflow.example.com",
            "experiment_id": "7",
            "run_id": "parent-run",
            "workspace": "ds-project",
            "run_url": "https://mlflow.example.com/#/experiments/7/runs/parent-run",
        }

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


class TestLogAutomlResults:
    """Tests for end-of-run MLflow logging."""

    def test_skips_when_mlflow_disabled(self, tmp_path, monkeypatch):
        """Return False without calling MLflow when tracking is disabled."""
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
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
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.com")
        monkeypatch.setenv("MLFLOW_RUN_ID", "parent-run")
        monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", "1")

        model_name = "LightGBM_BAG_L1_FULL"
        metrics_dir = tmp_path / model_name / "metrics"
        metrics_dir.mkdir(parents=True)
        (metrics_dir / "metrics.json").write_text(
            json.dumps({"accuracy": 0.91, "f1": 0.88}),
            encoding="utf-8",
        )
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
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.com")
        monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
        monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "automl-experiments")

        model_names = ["WeightedEnsemble_L3_FULL", "CatBoost_BAG_L1_FULL"]
        for model_name in model_names:
            metrics_dir = tmp_path / model_name / "metrics"
            metrics_dir.mkdir(parents=True)
            (metrics_dir / "metrics.json").write_text(
                json.dumps({"accuracy": 0.9, "f1": 0.88, "roc_auc": 0.95}),
                encoding="utf-8",
            )
        (tmp_path / "leaderboard.html").write_text("<html></html>", encoding="utf-8")

        mock_mlflow = mock.MagicMock()
        parent_ctx = _mock_run_context("parent-run", "99")
        child_ctx_1 = _mock_run_context("child-1", "99")
        child_ctx_2 = _mock_run_context("child-2", "99")
        mock_mlflow.start_run.side_effect = [parent_ctx, child_ctx_1, child_ctx_2]
        mock_mlflow.active_run.side_effect = [parent_ctx, child_ctx_1, parent_ctx, child_ctx_2, parent_ctx]
        mock_mlflow.entities.RunTag = mock.Mock(side_effect=lambda key, value: (key, value))
        mock_experiment = mock.MagicMock()
        mock_experiment.experiment_id = "99"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment

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
        mock_mlflow.start_run.assert_any_call(run_name="WeightedEnsemble_L3", nested=True)
        mock_mlflow.start_run.assert_any_call(run_name="CatBoost_BAG_L1", nested=True)

    def test_logs_connection_mode_creates_parent_run(self, tmp_path, monkeypatch):
        """Create a new parent run when only connection secret env vars are present."""
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.com")
        monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
        monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "automl-experiments")

        model_name = "LightGBM_BAG_L1_FULL"
        metrics_dir = tmp_path / model_name / "metrics"
        metrics_dir.mkdir(parents=True)
        (metrics_dir / "metrics.json").write_text(json.dumps({"accuracy": 0.91}), encoding="utf-8")
        (tmp_path / "leaderboard.html").write_text("<html></html>", encoding="utf-8")

        mock_mlflow = mock.MagicMock()
        parent_ctx = _mock_run_context("connection-parent-run", "99")
        child_ctx = _mock_run_context("child-run-1", "99")
        mock_mlflow.start_run.side_effect = [parent_ctx, child_ctx]
        mock_mlflow.active_run.side_effect = [parent_ctx, child_ctx, parent_ctx]
        mock_mlflow.entities.RunTag = mock.Mock(side_effect=lambda key, value: (key, value))
        mock_experiment = mock.MagicMock()
        mock_experiment.experiment_id = "99"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment

        models_artifact = _make_models_artifact(tmp_path, [model_name])
        with mock.patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            logged, tracking_info = log_automl_results(
                models_artifact=models_artifact,
                html_artifact_path=tmp_path / "leaderboard.html",
                eval_metric="accuracy",
                pipeline_name="autogluon-tabular-training-pipeline",
                kfp_run_id="run-1",
            )

        assert logged is True
        assert tracking_info["tracking_mode"] == "connection"
        assert tracking_info["mlflow_run_id"] == "connection-parent-run"
        assert tracking_info["mlflow_experiment_id"] == "99"
        assert tracking_info["mlflow_child_run_count"] == "1"
        mock_mlflow.set_experiment.assert_called_once_with("automl-experiments")
        mock_mlflow.start_run.assert_any_call(run_name="autogluon-tabular-training-pipeline")
        mock_mlflow.start_run.assert_any_call(run_name="LightGBM_BAG_L1", nested=True)

    def test_returns_false_when_mlflow_api_fails(self, tmp_path, monkeypatch):
        """Do not raise when the MLflow API call fails."""
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.com")
        monkeypatch.setenv("MLFLOW_RUN_ID", "parent-run")

        model_name = "LightGBM_BAG_L1_FULL"
        metrics_dir = tmp_path / model_name / "metrics"
        metrics_dir.mkdir(parents=True)
        (metrics_dir / "metrics.json").write_text(json.dumps({"accuracy": 0.91}), encoding="utf-8")
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
