from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from prophet_forecasting_tool.evaluate import (
    _parallel_mode,
    calculate_mae,
    calculate_mape,
    calculate_rmse,
    evaluate_with_cross_validation,
)


class TestMetricFunctions:
    def test_mae(self):
        y = pd.Series([10, 20, 30])
        p = pd.Series([11, 19, 32])
        assert calculate_mae(y, p) == pytest.approx((1 + 1 + 2) / 3)

    def test_mape_safe_against_zero(self):
        y = pd.Series([0, 20, 30])
        p = pd.Series([0, 19, 32])
        # Should not return inf even with y_true == 0.
        result = calculate_mape(y, p)
        assert pd.notna(result)
        assert result < 1000

    def test_rmse(self):
        y = pd.Series([10, 20, 30])
        p = pd.Series([11, 19, 32])
        assert calculate_rmse(y, p) == pytest.approx(((1 + 1 + 4) / 3) ** 0.5)


class TestParallelMode:
    def test_single_jobs_is_none(self):
        assert _parallel_mode(1) is None

    def test_minus_one_is_processes(self):
        assert _parallel_mode(-1) == "processes"

    def test_integer_is_processes(self):
        # Bug fix: previously broke for n_jobs=4 (passed an int to Prophet).
        assert _parallel_mode(4) == "processes"


class TestEvaluateWithCV:
    @patch("prophet_forecasting_tool.evaluate.train_prophet_model")
    @patch("prophet_forecasting_tool.evaluate.cross_validation")
    @patch("prophet_forecasting_tool.evaluate.performance_metrics")
    def test_success(self, mock_perf, mock_cv, mock_train):
        mock_model = MagicMock()
        mock_train.return_value = mock_model
        mock_cv.return_value = pd.DataFrame({"ds": [], "y": [], "yhat": [], "cutoff": []})
        mock_perf.return_value = pd.DataFrame({
            "horizon": ["30 days", "60 days"],
            "mae": [10.0, 20.0],
            "rmse": [12.0, 22.0],
        })

        sample = pd.DataFrame({
            "ds": pd.date_range("2020-01-01", periods=100),
            "y": list(range(100)),
        })
        results = evaluate_with_cross_validation(sample, metrics=["mae", "rmse"], n_jobs=1)

        assert results["mae"] == pytest.approx(15.0)
        assert results["rmse"] == pytest.approx(17.0)
        mock_train.assert_called()
        mock_cv.assert_called()

    def test_empty_df_raises(self):
        with pytest.raises(ValueError):
            evaluate_with_cross_validation(pd.DataFrame(columns=["ds", "y"]))
