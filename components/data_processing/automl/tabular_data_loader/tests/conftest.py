"""Test fixtures for tabular_data_loader."""

from unittest import mock

import pytest


def _make_run_status_artifact(tmp_path):
    art = mock.MagicMock()
    art.path = str(tmp_path / "run_status_artifact")
    art.metadata = {}
    return art


@pytest.fixture(autouse=True)
def inject_run_status_defaults(monkeypatch, tmp_path):
    """Inject KFP placeholder args when tests omit pipeline_name and run_id."""
    from ..component import automl_data_loader

    original = automl_data_loader.python_func

    def wrapper(*args, **kwargs):
        kwargs.setdefault("run_status_artifact", _make_run_status_artifact(tmp_path))
        if "workspace_path" in kwargs:
            from kfp_components.components.training.automl.shared.run_status import (
                PIPELINE_TABULAR_TRAINING,
                init_run_status,
            )

            init_run_status(
                kwargs["workspace_path"],
                kfp_run_id="test-run-id",
                pipeline_name="test-pipeline",
                run_status_pipeline_id=PIPELINE_TABULAR_TRAINING,
            )
        return original(*args, **kwargs)

    monkeypatch.setattr(automl_data_loader, "python_func", wrapper)
