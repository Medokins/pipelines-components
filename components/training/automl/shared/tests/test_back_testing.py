"""Tests for time series back_testing.json builder."""

from __future__ import annotations

from unittest import mock

import pandas as pd
import pytest

from ..back_testing import (
    _build_series_analysis,
    _forecast_data_for_item,
    _holdout_frame,
    _item_window_metrics,
    _series_ranking_metric,
    build_back_testing_json,
    filter_finite_metrics,
    serialize_date,
    serialize_timestamp,
)


def _make_panel(item_ids: list[str], timestamps: list[str], target_values: list[float]) -> pd.DataFrame:
    rows = []
    for item_id in item_ids:
        for ts, value in zip(timestamps, target_values, strict=True):
            rows.append((item_id, pd.Timestamp(ts), value))
    index = pd.MultiIndex.from_tuples(
        [(item_id, ts) for item_id, ts, _ in rows],
        names=["item_id", "timestamp"],
    )
    return pd.DataFrame({"target": [value for _, _, value in rows]}, index=index)


class TestSerialization:
    """Tests for serialization helpers."""

    def test_filter_finite_metrics_drops_nan(self):
        """Non-finite metric values are omitted."""
        assert filter_finite_metrics({"MASE": 0.5, "MAPE": float("nan"), "RMSE": float("inf")}) == {"MASE": 0.5}

    def test_serialize_timestamp_utc(self):
        """Timestamps serialize to ISO strings with UTC suffix."""
        assert serialize_timestamp(pd.Timestamp("2025-12-08T00:00:00Z")) == "2025-12-08T00:00:00Z"
        assert serialize_timestamp(pd.Timestamp("2025-12-08")) == "2025-12-08T00:00:00Z"

    def test_serialize_date(self):
        """Window bounds serialize to YYYY-MM-DD."""
        assert serialize_date(pd.Timestamp("2025-12-08T15:30:00Z")) == "2025-12-08"


class TestHoldoutHelpers:
    """Tests for holdout and forecast helpers."""

    def test_holdout_frame_takes_last_prediction_length_per_item(self):
        """Holdout uses the last prediction_length rows per series."""
        ts = ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]
        panel = _make_panel(["A"], ts, [1.0, 2.0, 3.0, 4.0])
        holdout = _holdout_frame(panel, prediction_length=2)
        assert len(holdout) == 2
        assert holdout["target"].tolist() == [3.0, 4.0]

    def test_holdout_frame_single_series_no_multiindex(self):
        """Holdout works for single time-series (non-MultiIndex DataFrame)."""
        timestamps = pd.date_range("2025-01-01", periods=5, freq="D")
        single_series = pd.DataFrame(
            {"target": [10.0, 20.0, 30.0, 40.0, 50.0]},
            index=timestamps,
        )
        holdout = _holdout_frame(single_series, prediction_length=2)
        assert len(holdout) == 2
        assert holdout["target"].tolist() == [40.0, 50.0]
        assert not isinstance(holdout.index, pd.MultiIndex)

    def test_item_window_metrics_computes_mape(self):
        """Per-item window metrics include MAPE from point forecasts."""
        timestamps = ["2025-01-03", "2025-01-04"]
        targets = _make_panel(["A"], timestamps, [100.0, 200.0])
        predictions = pd.DataFrame(
            {"mean": [110.0, 180.0]},
            index=targets.index,
        )
        metrics = _item_window_metrics(predictions, targets, "A", "target", prediction_length=2)
        assert "MAPE" in metrics
        assert metrics["MAPE"] == pytest.approx(10.0)

    def test_forecast_data_includes_actual_and_predicted(self):
        """Forecast rows include actual, predicted, and optional quantile bounds."""
        timestamps = ["2025-01-03"]
        targets = _make_panel(["A"], timestamps, [100.0])
        predictions = pd.DataFrame({"mean": [105.0], "0.1": [95.0], "0.9": [115.0]}, index=targets.index)
        rows = _forecast_data_for_item(predictions, targets, "A", "target", prediction_length=1)
        assert rows[0]["actual"] == 100.0
        assert rows[0]["predicted"] == 105.0
        assert rows[0]["lower_bound"] == 95.0
        assert rows[0]["upper_bound"] == 115.0

    def test_forecast_data_limits_to_holdout_horizon(self):
        """Forecast rows cover holdout steps only, not the full prediction window history."""
        timestamps = ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05"]
        targets = _make_panel(["A"], timestamps, [1.0, 2.0, 3.0, 4.0, 5.0])
        predictions = pd.DataFrame({"mean": [1.1, 2.1, 3.1, 4.1, 5.1]}, index=targets.index)
        rows = _forecast_data_for_item(predictions, targets, "A", "target", prediction_length=2)
        assert len(rows) == 2
        assert rows[0]["timestamp"].startswith("2025-01-04")
        assert rows[1]["timestamp"].startswith("2025-01-05")

    def test_forecast_data_omits_bounds_without_quantiles(self):
        """Forecast rows omit bounds when prediction output has no quantile columns."""
        timestamps = ["2025-01-03"]
        targets = _make_panel(["A"], timestamps, [100.0])
        predictions = pd.DataFrame({"mean": [105.0]}, index=targets.index)
        rows = _forecast_data_for_item(predictions, targets, "A", "target", prediction_length=1)
        assert "lower_bound" not in rows[0]
        assert "upper_bound" not in rows[0]


class TestSeriesRanking:
    """Tests for series ranking metric selection."""

    def test_series_ranking_metric_uses_eval_metric_when_available(self):
        """Uses eval_metric when it's a point metric and available."""
        series_averages = {"A": {"MAPE": 5.0, "RMSE": 10.0}, "B": {"MAPE": 8.0, "RMSE": 12.0}}
        assert _series_ranking_metric("MAPE", series_averages) == "MAPE"
        assert _series_ranking_metric("RMSE", series_averages) == "RMSE"

    def test_series_ranking_metric_falls_back_when_mape_unavailable(self):
        """Falls back to RMSE when MAPE can't be computed (zero denominators)."""
        # MAPE missing due to zero denominators
        series_averages = {"A": {"RMSE": 10.0, "MAE": 5.0}, "B": {"RMSE": 12.0, "MAE": 6.0}}
        assert _series_ranking_metric("MAPE", series_averages) == "RMSE"

    def test_series_ranking_metric_uses_mae_as_last_resort(self):
        """Uses MAE when both MAPE and RMSE unavailable."""
        series_averages = {"A": {"MAE": 5.0}, "B": {"MAE": 6.0}}
        assert _series_ranking_metric("MAPE", series_averages) == "MAE"

    def test_series_ranking_metric_defaults_to_mape_for_non_point_metrics(self):
        """For non-point metrics (MASE, WQL), falls back to MAPE if available."""
        series_averages = {"A": {"MAPE": 5.0, "MASE": 0.5}, "B": {"MAPE": 8.0, "MASE": 0.7}}
        assert _series_ranking_metric("MASE", series_averages) == "MAPE"


class TestSeriesAnalysis:
    """Tests for series analysis payload construction."""

    def test_no_performers_when_no_series_metrics(self):
        """Best/worst performers are omitted when no series metrics are available."""
        analysis = _build_series_analysis([], [], target="target", prediction_length=2, eval_metric="MAPE")
        assert analysis["num_series_evaluated"] == 0
        assert analysis["best_performer"] is None
        assert analysis["worst_performer"] is None


class TestBuildBackTestingJson:
    """Tests for build_back_testing_json orchestration."""

    def test_cutoff_calculation_matches_autogluon_api(self):
        """Cutoff values are calculated correctly for AutoGluon evaluate() API.

        AutoGluon's cutoff parameter: negative integer where evaluation starts.
        cutoff=-N evaluates from -N-th to (-N + prediction_length)-th time step.

        For num_val_windows=3, prediction_length=2:
        - cutoffs should be [-6, -4, -2]
        - window 0: evaluate steps -6 to -4
        - window 1: evaluate steps -4 to -2
        - window 2: evaluate steps -2 to 0 (end)
        """
        timestamps = ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05", "2025-01-06"]
        train_data = _make_panel(["A"], timestamps, [100.0, 110.0, 120.0, 130.0, 140.0, 150.0])

        predictor = mock.MagicMock()
        # Create 3 prediction windows
        pred1 = pd.DataFrame({"mean": [121.0, 131.0]}, index=train_data.tail(2).index)
        pred2 = pd.DataFrame({"mean": [131.0, 141.0]}, index=train_data.tail(4).head(2).index)
        pred3 = pd.DataFrame({"mean": [111.0, 121.0]}, index=train_data.tail(6).head(2).index)

        predictor.backtest_predictions.return_value = [pred3, pred2, pred1]  # chronological order
        predictor.backtest_targets.return_value = [
            train_data.tail(6).head(2),
            train_data.tail(4).head(2),
            train_data.tail(2),
        ]

        # Track cutoff values passed to evaluate()
        cutoff_calls = []

        def mock_evaluate(**kwargs):
            cutoff_calls.append(kwargs.get("cutoff"))
            return {"MASE": 0.5}

        predictor.evaluate.side_effect = mock_evaluate

        build_back_testing_json(
            predictor,
            model_name="DeepAR",
            model_name_full="DeepAR_FULL",
            train_data=train_data,
            eval_metric="MASE",
            target="target",
            id_column="item_id",
            timestamp_column="timestamp",
            prediction_length=2,
            num_val_windows=3,
        )

        # Verify cutoff values: should be [-6, -4, -2] for 3 windows, prediction_length=2
        assert cutoff_calls == [-6, -4, -2], f"Expected [-6, -4, -2], got {cutoff_calls}"

    def test_builds_schema_with_mock_predictor(self):
        """Builder emits ADR-shaped payload from mocked AutoGluon backtest APIs."""
        timestamps = ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]
        train_data = _make_panel(["good", "bad"], timestamps, [100.0, 110.0, 120.0, 130.0])

        window_targets = _holdout_frame(train_data, prediction_length=2)
        good_preds = pd.DataFrame({"mean": [121.0, 131.0]}, index=window_targets.loc["good"].index)
        bad_preds = pd.DataFrame({"mean": [200.0, 210.0]}, index=window_targets.loc["bad"].index)
        predictions = pd.concat({"good": good_preds, "bad": bad_preds}, names=["item_id", "timestamp"])

        predictor = mock.MagicMock()
        predictor.backtest_predictions.return_value = [predictions]
        predictor.backtest_targets.return_value = [window_targets]
        predictor.evaluate.return_value = {"MASE": 0.42, "MAPE": 5.0}

        payload = build_back_testing_json(
            predictor,
            model_name="DeepAR",
            model_name_full="DeepAR_FULL",
            train_data=train_data,
            eval_metric="MASE",
            target="target",
            id_column="item_id",
            timestamp_column="timestamp",
            prediction_length=2,
            num_val_windows=1,
            metrics=["MASE", "MAPE"],
        )

        assert payload["model_name"] == "DeepAR_FULL"
        assert payload["num_val_windows"] == 1
        assert payload["per_window_metrics"][0]["test_start"] == "2025-01-03"
        assert payload["per_window_metrics"][0]["metrics"]["MASE"] == 0.42
        assert payload["series_analysis"]["num_series_evaluated"] == 2
        assert payload["series_analysis"]["best_performer"]["item_id"] == "good"
        assert payload["series_analysis"]["worst_performer"]["item_id"] == "bad"
        assert payload["series_analysis"]["best_performer"]["windows"][0]["forecast_data"]
        assert payload["series_analysis"]["best_performer"]["windows"][0]["forecast_data"][0]["timestamp"].endswith("Z")
        assert "schema_version" not in payload
        assert "ranking_metric" not in payload["series_analysis"]

    def test_ranks_by_point_metric_matching_eval_metric(self):
        """Best/worst selection uses eval_metric when it is a computed point-forecast metric."""
        timestamps = ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]
        train_data = _make_panel(["good", "bad"], timestamps, [100.0, 110.0, 120.0, 130.0])

        window_targets = _holdout_frame(train_data, prediction_length=2)
        good_preds = pd.DataFrame({"mean": [121.0, 131.0]}, index=window_targets.loc["good"].index)
        bad_preds = pd.DataFrame({"mean": [200.0, 210.0]}, index=window_targets.loc["bad"].index)
        predictions = pd.concat({"good": good_preds, "bad": bad_preds}, names=["item_id", "timestamp"])

        predictor = mock.MagicMock()
        predictor.backtest_predictions.return_value = [predictions]
        predictor.backtest_targets.return_value = [window_targets]
        predictor.evaluate.return_value = {"RMSE": 1.0}

        payload = build_back_testing_json(
            predictor,
            model_name="DeepAR",
            model_name_full="DeepAR_FULL",
            train_data=train_data,
            eval_metric="RMSE",
            target="target",
            id_column="item_id",
            timestamp_column="timestamp",
            prediction_length=2,
            num_val_windows=1,
        )

        assert payload["series_analysis"]["best_performer"]["item_id"] == "good"
        assert "ranking_metric" not in payload["series_analysis"]

    def test_requires_backtest_api(self):
        """Missing backtest methods raise AttributeError."""
        predictor = mock.MagicMock(spec=[])
        with pytest.raises(AttributeError, match="backtest API"):
            build_back_testing_json(
                predictor,
                model_name="DeepAR",
                model_name_full="DeepAR_FULL",
                train_data=pd.DataFrame(),
                eval_metric="MASE",
                target="target",
                id_column="item_id",
                timestamp_column="timestamp",
                prediction_length=1,
            )

    def test_single_time_series_no_multiindex(self):
        """Builder handles single time-series (non-MultiIndex) correctly."""
        timestamps = pd.date_range("2025-01-01", periods=6, freq="D")
        train_data = pd.DataFrame(
            {"target": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0]},
            index=timestamps,
        )

        # Mock predictions for last 2 points (holdout)
        holdout_timestamps = timestamps[-2:]
        predictions = pd.DataFrame(
            {"mean": [141.0, 151.0]},
            index=holdout_timestamps,
        )

        window_targets = train_data.tail(2)

        predictor = mock.MagicMock()
        predictor.backtest_predictions.return_value = [predictions]
        predictor.backtest_targets.return_value = [window_targets]
        predictor.evaluate.return_value = {"MASE": 0.5, "MAPE": 2.0}

        payload = build_back_testing_json(
            predictor,
            model_name="DeepAR",
            model_name_full="DeepAR_FULL",
            train_data=train_data,
            eval_metric="MASE",
            target="target",
            id_column=None,
            timestamp_column="timestamp",
            prediction_length=2,
            num_val_windows=1,
            metrics=["MASE", "MAPE"],
        )

        assert payload["model_name"] == "DeepAR_FULL"
        assert payload["num_val_windows"] == 1
        assert payload["per_window_metrics"][0]["test_start"] == "2025-01-05"
        assert payload["per_window_metrics"][0]["test_end"] == "2025-01-06"
        assert payload["per_window_metrics"][0]["metrics"]["MASE"] == 0.5
        # Single series: series_analysis should show num_series_evaluated = 1
        assert payload["series_analysis"]["num_series_evaluated"] == 1
        # For single series, best and worst should be None (only one series)
        assert payload["series_analysis"]["best_performer"] is not None
        assert payload["series_analysis"]["worst_performer"] is not None
