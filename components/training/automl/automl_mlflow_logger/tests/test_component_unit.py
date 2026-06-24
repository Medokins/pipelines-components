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


@pytest.fixture
def mlflow_tracking_artifact(tmp_path):
    """Mock KFP output artifact for mlflow_tracking_artifact."""
    artifact = mock.MagicMock()
    artifact.path = str(tmp_path / "mlflow_tracking")
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
        mlflow_tracking_artifact,
    ):
        """Delegates to shared logger and records status when MLflow is disabled."""
        model_name = "LightGBM_BAG_L1_FULL"
        metrics_dir = tmp_path / model_name / "metrics"
        metrics_dir.mkdir(parents=True)
        (metrics_dir / "metrics.json").write_text(json.dumps({"accuracy": 0.9}), encoding="utf-8")
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
            mlflow_tracking_artifact=mlflow_tracking_artifact,
            preset="speed",
            top_n=1,
        )

        mock_log.assert_called_once()
        tracking_file = Path(mlflow_tracking_artifact.path) / "mlflow_tracking.json"
        assert tracking_file.is_file()
        assert component_status_artifact.metadata["display_name"] == "MLflow Logging Status"

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
    def test_writes_tracking_artifact_after_connection_logging(
        self,
        mock_log,
        tmp_path,
        component_status_artifact,
        mlflow_tracking_artifact,
    ):
        """Write final tracking artifact with MLflow run IDs after successful logging."""
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
            mlflow_tracking_artifact=mlflow_tracking_artifact,
        )

        mock_log.assert_called_once()
        payload = json.loads(
            (Path(mlflow_tracking_artifact.path) / "mlflow_tracking.json").read_text(encoding="utf-8")
        )
        assert payload["mlflow_run_id"] == "run-abc"
        assert payload["mlflow_experiment_id"] == "7"
        assert payload["tracking_mode"] == "connection"

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
        mlflow_tracking_artifact,
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
            mlflow_tracking_artifact=mlflow_tracking_artifact,
        )

        mock_log.assert_called_once()
        payload = json.loads(
            (Path(mlflow_tracking_artifact.path) / "mlflow_tracking.json").read_text(encoding="utf-8")
        )
        assert "mlflow_error" in payload
        assert mlflow_tracking_artifact.metadata["tracking_enabled"] == "False"

    def test_rejects_empty_eval_metric(self, tmp_path, component_status_artifact, mlflow_tracking_artifact):
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
                mlflow_tracking_artifact=mlflow_tracking_artifact,
            )

    def test_rejects_invalid_top_n(self, tmp_path, component_status_artifact, mlflow_tracking_artifact):
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
                mlflow_tracking_artifact=mlflow_tracking_artifact,
                top_n=0,
            )

    def test_embedded_mlflow_tracking_module_load(self, tmp_path):
        """Embedded mlflow_tracking.py loads when registered in sys.modules."""
        import importlib.util
        import sys

        source = Path(__file__).resolve().parents[2] / "shared" / "mlflow_tracking.py"
        module_path = tmp_path / "mlflow_tracking.py"
        module_path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

        module_name = "embedded_mlflow_tracking_test"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        assert callable(module.log_automl_results)
        assert callable(module.write_tracking_artifact)
