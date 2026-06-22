"""Unit tests for MLflow tracking helpers."""

import json
import sys
from pathlib import Path
from unittest import mock

import pytest
from kfp_components.components.training.automl.shared.mlflow_tracking import (
    TRACKING_ARTIFACT_FILENAME,
    build_mlflow_run_url,
    build_tracking_artifact_payload,
    is_mlflow_enabled,
    log_automl_results,
    parse_model_name,
    write_tracking_artifact,
)


class TestMlflowTrackingHelpers:
    """Tests for MLflow env helpers and tracking artifact builders."""

    def test_is_mlflow_enabled_false_when_unset(self, monkeypatch):
        """Return False when MLFLOW_TRACKING_URI is unset."""
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        assert is_mlflow_enabled() is False

    def test_is_mlflow_enabled_true_when_set(self, monkeypatch):
        """Return True when MLFLOW_TRACKING_URI is set."""
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.com")
        assert is_mlflow_enabled() is True

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
            "kfp_run_id": "run-1",
            "pipeline_name": "autogluon-tabular-training-pipeline",
        }

    def test_build_tracking_artifact_payload_enabled(self, monkeypatch):
        """Include MLflow IDs when tracking is enabled."""
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
        assert payload["mlflow_experiment_id"] == "7"
        assert payload["mlflow_run_id"] == "parent-run"
        assert payload["mlflow_workspace"] == "ds-project"
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
        assert (
            log_automl_results(
                models_artifact=models_artifact,
                html_artifact_path=tmp_path / "leaderboard.html",
                eval_metric="accuracy",
                pipeline_name="autogluon-tabular-training-pipeline",
                kfp_run_id="run-1",
            )
            is False
        )

    def test_logs_parent_and_child_runs(self, tmp_path, monkeypatch):
        """Create parent and nested child MLflow runs from model artifacts."""
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
        mock_mlflow.start_run.side_effect = [parent_ctx, child_ctx]

        models_artifact = _make_models_artifact(tmp_path, [model_name])
        with mock.patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            assert (
                log_automl_results(
                    models_artifact=models_artifact,
                    html_artifact_path=tmp_path / "leaderboard.html",
                    eval_metric="accuracy",
                    pipeline_name="autogluon-tabular-training-pipeline",
                    kfp_run_id="run-1",
                    task_type="binary",
                    preset="speed",
                    top_n=1,
                )
                is True
            )

        mock_mlflow.start_run.assert_any_call(run_id="parent-run")
        mock_mlflow.start_run.assert_any_call(run_name=model_name, nested=True)
        mock_mlflow.set_tags.assert_called_once()
        mock_mlflow.log_params.assert_called()
        mock_mlflow.log_metrics.assert_called_once()
        mock_mlflow.log_artifact.assert_called()
