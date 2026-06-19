"""Tests for time series integration test config JSON loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from . import test_configs

_TESTS_DIR = Path(__file__).resolve().parent


def _minimal_timeseries_entry(**overrides) -> dict:
    base = {
        "id": "cfg-1",
        "dataset_path": "data/timeseries_sales.csv",
        "target": "target",
        "id_column": "item_id",
        "timestamp_column": "timestamp",
        "known_covariates_names": ["promo"],
        "prediction_length": 2,
        "top_n": 2,
        "tags": [],
    }
    base.update(overrides)
    return base


def test_all_dataset_paths_exist() -> None:
    """Every dataset_path in test_configs.json resolves to a file under tests/."""
    for config in test_configs.TEST_CONFIGS:
        dataset = _TESTS_DIR / config.dataset_path
        assert dataset.is_file(), f"Missing dataset for config {config.id!r}: {dataset}"


def test_eval_metric_forwarded_in_pipeline_arguments() -> None:
    """Configs with eval_metric pass it through get_pipeline_arguments()."""
    wql_config = next(c for c in test_configs.TEST_CONFIGS if c.id == "timeseries_wql_eval_metric")
    args = wql_config.get_pipeline_arguments("bucket", "key", "secret")
    assert args["eval_metric"] == "WQL"

    smoke_config = next(c for c in test_configs.TEST_CONFIGS if c.id == "timeseries_smoke")
    smoke_args = smoke_config.get_pipeline_arguments("bucket", "key", "secret")
    assert "eval_metric" not in smoke_args


def test_load_configs_rejects_blank_eval_metric(tmp_path: Path) -> None:
    """Blank eval_metric values fail at config load time."""
    bad = tmp_path / "configs.json"
    bad.write_text(
        json.dumps([_minimal_timeseries_entry(eval_metric="   ")]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"test_configs\.json\[0\] 'eval_metric'"):
        test_configs._load_configs(bad)


def test_load_configs_rejects_non_string_eval_metric(tmp_path: Path) -> None:
    """Non-string eval_metric values fail at config load time."""
    bad = tmp_path / "configs.json"
    bad.write_text(
        json.dumps([_minimal_timeseries_entry(eval_metric=123)]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"test_configs\.json\[0\] 'eval_metric'"):
        test_configs._load_configs(bad)


def test_load_configs_accepts_valid_eval_metric(tmp_path: Path) -> None:
    """Valid eval_metric is stripped and forwarded through pipeline arguments."""
    path = tmp_path / "configs.json"
    path.write_text(
        json.dumps([_minimal_timeseries_entry(eval_metric=" WQL ")]),
        encoding="utf-8",
    )
    loaded = test_configs._load_configs(path)
    assert len(loaded) == 1
    assert loaded[0].eval_metric == "WQL"
    args = loaded[0].get_pipeline_arguments("bucket", "key", "secret")
    assert args["eval_metric"] == "WQL"
