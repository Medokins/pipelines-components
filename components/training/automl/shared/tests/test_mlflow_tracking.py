"""Unit tests for MLflow tracking helpers."""

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

from kfp_components.components.training.automl.shared.mlflow_tracking import (
    TRACKING_ARTIFACT_FILENAME,
    MLFLOW_CONNECTION_SECRET_KEY_TO_ENV,
    build_mlflow_run_url,
    build_tracking_artifact_payload,
    is_mlflow_enabled,
    log_automl_results,
    parse_model_name,
    resolve_leaderboard_html_path,
    resolve_mlflow_config,
    write_tracking_artifact,
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

    def test_build_tracking_artifact_payload_disabled(self, monkeypatch):
        """Emit minimal payload when tracking is disabled."""
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        payload = build_tracking_artifact_payload(
            pipeline_name="autogluon-tabular-training-pipeline",
            kfp_run_id="run-1",
        )
        assert payload == {
            "tracking_enabled": False,
            "tracking_mode": "disabled",
            "kfp_run_id": "run-1",
            "pipeline_name": "autogluon-tabular-training-pipeline",
        }

    def test_build_tracking_artifact_payload_connection(self, monkeypatch):
        """Include connection mode metadata when only URI is available."""
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.com")
        monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
        payload = build_tracking_artifact_payload(
            pipeline_name="autogluon-tabular-training-pipeline",
            kfp_run_id="run-1",
        )
        assert payload["tracking_enabled"] is True
        assert payload["tracking_mode"] == "connection"
        assert payload["mlflow_tracking_uri"] == "https://mlflow.example.com"

    def test_build_tracking_artifact_payload_enabled(self, monkeypatch):
        """Include MLflow IDs when tracking is enabled in KFP mode."""
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.com")
        monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", "7")
        monkeypatch.setenv("MLFLOW_RUN_ID", "parent-run")
        monkeypatch.setenv("MLFLOW_WORKSPACE", "ds-project")
        payload = build_tracking_artifact_payload(
            pipeline_name="autogluon-tabular-training-pipeline",
            kfp_run_id="run-1",
            kfp_run_name="my-run",
        )
        assert payload["tracking_enabled"] is True
        assert payload["tracking_mode"] == "kfp"
        assert payload["mlflow_experiment_id"] == "7"
        assert payload["mlflow_run_id"] == "parent-run"
        assert payload["kfp_run_name"] == "my-run"
        assert "mlflow_run_url" in payload

    def test_write_tracking_artifact(self, tmp_path, monkeypatch):
        """Write mlflow_tracking.json under the artifact directory."""
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        output_file = write_tracking_artifact(
            tmp_path,
            pipeline_name="autogluon-tabular-training-pipeline",
            kfp_run_id="run-abc",
        )
        assert output_file == tmp_path / TRACKING_ARTIFACT_FILENAME
        payload = json.loads(output_file.read_text(encoding="utf-8"))
        assert payload["tracking_enabled"] is False
        assert payload["kfp_run_id"] == "run-abc"
        assert (tmp_path / "data").read_text(encoding="utf-8") == output_file.read_text(encoding="utf-8")

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
        parent_ctx = mock.MagicMock()
        child_ctx = mock.MagicMock()
        parent_ctx.info.run_id = "parent-run"
        mock_mlflow.start_run.side_effect = [parent_ctx, child_ctx]
        mock_mlflow.active_run.return_value = parent_ctx

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
        mock_mlflow.start_run.assert_any_call(run_id="parent-run")
        mock_mlflow.start_run.assert_any_call(run_name=model_name, nested=True)
        mock_mlflow.set_tags.assert_called_once()
        mock_mlflow.log_params.assert_called()
        mock_mlflow.log_metrics.assert_called_once()
        mock_mlflow.log_artifact.assert_called()

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
        parent_ctx = mock.MagicMock()
        child_ctx = mock.MagicMock()
        parent_ctx.info.run_id = "connection-parent-run"
        mock_mlflow.start_run.side_effect = [parent_ctx, child_ctx]
        mock_mlflow.active_run.return_value = parent_ctx
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
        mock_mlflow.set_experiment.assert_called_once_with("automl-experiments")
        mock_mlflow.start_run.assert_any_call(run_name="autogluon-tabular-training-pipeline")

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
