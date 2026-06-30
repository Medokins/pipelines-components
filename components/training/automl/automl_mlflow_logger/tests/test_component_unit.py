"""Unit tests for automl_mlflow_logger."""

import json
from pathlib import Path
from unittest import mock

import pytest

from ..component import automl_mlflow_logger


@pytest.fixture
def component_status_artifact(tmp_path):
    """Mock KFP output artifact for component_status."""
    artifact = mock.MagicMock()
    artifact.path = str(tmp_path / "component_status")
    artifact.metadata = {}
    return artifact


def _make_models_artifact(base_path: Path, model_names: list[str]):
    artifact = mock.MagicMock()
    artifact.path = str(base_path)
    artifact.uri = "s3://bucket/models"
    artifact.metadata = {
        "model_names": json.dumps(model_names),
        "context": {"task_type": "binary"},
    }
    return artifact


def _make_html_artifact(html_path: Path):
    artifact = mock.MagicMock()
    artifact.path = str(html_path)
    return artifact


class TestAutomlMlflowLogger:
    """Unit tests for automl_mlflow_logger."""

    def test_component_is_callable(self):
        """Component is importable and exposes python_func."""
        assert callable(automl_mlflow_logger)
        assert hasattr(automl_mlflow_logger, "python_func")

    @mock.patch(
        "kfp_components.components.training.automl.shared.mlflow_tracking.log_automl_results",
        return_value=(False, {}),
    )
    def test_skips_when_mlflow_disabled(
        self,
        mock_log,
        tmp_path,
        component_status_artifact,
    ):
        """Delegates to shared logger and records status when MLflow is disabled."""
        model_name = "LightGBM_BAG_L1_FULL"
        metrics_dir = tmp_path / model_name / "metrics"
        metrics_dir.mkdir(parents=True)
        (metrics_dir / "metrics.json").write_text('{"accuracy": 0.9}', encoding="utf-8")
        html_path = tmp_path / "leaderboard.html"
        html_path.write_text("<html></html>", encoding="utf-8")

        automl_mlflow_logger.python_func(
            models_artifact=_make_models_artifact(tmp_path, [model_name]),
            html_artifact=_make_html_artifact(html_path),
            eval_metric="accuracy",
            pipeline_name="autogluon-tabular-training-pipeline",
            run_id="run-1",
            task_type="binary",
            component_status=component_status_artifact,
            preset="speed",
            top_n=1,
        )

        mock_log.assert_called_once()
        assert component_status_artifact.metadata["display_name"] == "MLflow Logging Status"
        assert component_status_artifact.metadata["tracking_enabled"] == "False"

    @mock.patch(
        "kfp_components.components.training.automl.shared.mlflow_tracking.log_automl_results",
        return_value=(
            True,
            {
                "mlflow_run_id": "run-abc",
                "mlflow_experiment_id": "7",
                "tracking_mode": "connection",
            },
        ),
    )
    def test_records_metadata_after_connection_logging(
        self,
        mock_log,
        tmp_path,
        component_status_artifact,
    ):
        """Record MLflow run IDs on component_status metadata after successful logging."""
        html_path = tmp_path / "leaderboard.html"
        html_path.write_text("<html></html>", encoding="utf-8")

        automl_mlflow_logger.python_func(
            models_artifact=_make_models_artifact(tmp_path, ["Model_FULL"]),
            html_artifact=_make_html_artifact(html_path),
            eval_metric="accuracy",
            pipeline_name="autogluon-tabular-training-pipeline",
            run_id="run-1",
            task_type="binary",
            component_status=component_status_artifact,
        )

        mock_log.assert_called_once()
        assert component_status_artifact.metadata["tracking_enabled"] == "True"
        assert component_status_artifact.metadata["mlflow_run_id"] == "run-abc"
        assert component_status_artifact.metadata["mlflow_experiment_id"] == "7"
        assert component_status_artifact.metadata["tracking_mode"] == "connection"

    @mock.patch(
        "kfp_components.components.training.automl.shared.mlflow_tracking.log_automl_results",
        return_value=(
            False,
            {
                "error": '{"reason": "NotAcceptable", "code": 406}',
                "tracking_mode": "connection",
            },
        ),
    )
    def test_completes_when_mlflow_api_fails(
        self,
        mock_log,
        tmp_path,
        component_status_artifact,
    ):
        """Step succeeds and records the MLflow error when logging fails."""
        html_path = tmp_path / "leaderboard.html"
        html_path.write_text("<html></html>", encoding="utf-8")

        automl_mlflow_logger.python_func(
            models_artifact=_make_models_artifact(tmp_path, ["Model_FULL"]),
            html_artifact=_make_html_artifact(html_path),
            eval_metric="accuracy",
            pipeline_name="autogluon-tabular-training-pipeline",
            run_id="run-1",
            task_type="binary",
            component_status=component_status_artifact,
        )

        mock_log.assert_called_once()
        assert component_status_artifact.metadata["mlflow_error"] == '{"reason": "NotAcceptable", "code": 406}'
        assert component_status_artifact.metadata["tracking_enabled"] == "False"

    def test_rejects_empty_eval_metric(self, tmp_path, component_status_artifact):
        """Reject blank eval_metric."""
        with pytest.raises(TypeError, match="eval_metric"):
            automl_mlflow_logger.python_func(
                models_artifact=_make_models_artifact(tmp_path, ["Model_FULL"]),
                html_artifact=_make_html_artifact(tmp_path / "leaderboard.html"),
                eval_metric="",
                pipeline_name="autogluon-tabular-training-pipeline",
                run_id="run-1",
                task_type="binary",
                component_status=component_status_artifact,
            )

    def test_rejects_invalid_top_n(self, tmp_path, component_status_artifact):
        """Reject non-positive top_n."""
        with pytest.raises(ValueError, match="top_n"):
            automl_mlflow_logger.python_func(
                models_artifact=_make_models_artifact(tmp_path, ["Model_FULL"]),
                html_artifact=_make_html_artifact(tmp_path / "leaderboard.html"),
                eval_metric="accuracy",
                pipeline_name="autogluon-tabular-training-pipeline",
                run_id="run-1",
                task_type="binary",
                component_status=component_status_artifact,
                top_n=0,
            )
