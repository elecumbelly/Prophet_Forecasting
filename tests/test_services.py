"""ForecastingService tests using a temporary models dir and mocked Prophet calls."""
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from prophet_forecasting_tool.services import ForecastingService


@pytest.fixture
def sample_df():
    df = pd.DataFrame({
        "ds": pd.date_range("2022-01-01", periods=400, freq="D"),
        "y": [100 + i + (i % 7) * 5 for i in range(400)],
    })
    return df


@pytest.fixture
def service(tmp_path):
    mock_engine = MagicMock()
    mock_settings = MagicMock()
    return ForecastingService(mock_engine, mock_settings, tmp_path / "models")


class TestDataFingerprint:
    def test_empty(self, service):
        assert service._data_fingerprint(pd.DataFrame(columns=["ds", "y"])) == "empty_data"

    def test_changes_with_y(self, service):
        a = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=3), "y": [1, 2, 3]})
        b = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=3), "y": [1, 2, 4]})
        # Old fingerprint was (max_date, row_count) only; the new one must differ.
        assert service._data_fingerprint(a) != service._data_fingerprint(b)


class TestOutlierFilter:
    def test_preserves_holidays(self, service):
        df = pd.DataFrame({
            "ds": pd.date_range("2024-12-20", periods=10, freq="D"),
            "y": [10, 10, 10, 10, 1000, 10, 10, 10, 10, 10],  # spike on day 5
        })
        # Day 5 = 2024-12-24 — declare it as a holiday so it must survive.
        holidays_df = pd.DataFrame({
            "ds": [pd.Timestamp("2024-12-24")],
            "holiday": ["Test"],
            "lower_window": [0],
            "upper_window": [0],
        })
        kept, removed = service._filter_outliers(df, holidays_df)
        assert removed == 0
        assert pd.Timestamp("2024-12-24") in kept["ds"].values

    def test_removes_non_holiday_outlier(self, service):
        df = pd.DataFrame({
            "ds": pd.date_range("2024-01-01", periods=10, freq="D"),
            "y": [10, 10, 10, 10, 1000, 10, 10, 10, 10, 10],
        })
        kept, removed = service._filter_outliers(df, holidays_df=None)
        assert removed == 1
        assert 1000 not in kept["y"].values


class TestGetForecast:
    @patch("prophet_forecasting_tool.services.forecast_with_prophet")
    @patch("prophet_forecasting_tool.services.train_prophet_model")
    @patch("prophet_forecasting_tool.services.load_time_series")
    @patch("prophet_forecasting_tool.services.get_max_date_for_table")
    def test_returns_forecast_payload(
        self, mock_max_date, mock_load, mock_train, mock_forecast, service, sample_df
    ):
        mock_max_date.return_value = sample_df["ds"].max()
        mock_load.return_value = sample_df
        model = MagicMock()
        model.extra_regressors = {}
        fig = MagicMock()
        fig.gca.return_value = MagicMock()
        model.plot.return_value = fig
        model.plot_components.return_value = fig
        mock_train.return_value = model
        mock_forecast.return_value = pd.DataFrame({
            "ds": pd.date_range("2024-01-01", periods=30),
            "yhat": list(range(30)),
        })

        result = service.get_forecast(
            table_name="demo",
            ts_column="ts",
            y_column="y",
            freq="D",
            horizon="30D",
            series_name="test_series",
            regressors=[],
            resample_to_freq=None,
            training_window_duration="180 days",
        )
        assert result["series_name"] == "test_series"
        assert "forecast_df" in result
        assert "training_rows" in result

    def test_rejects_bad_table(self, service):
        with pytest.raises(ValueError):
            service.get_forecast(
                table_name="bad-table",
                ts_column="ts",
                y_column="y",
                freq="D",
                horizon="30D",
                series_name="s",
                regressors=[],
            )

    def test_rejects_bad_horizon(self, service):
        with pytest.raises(ValueError):
            service.get_forecast(
                table_name="demo",
                ts_column="ts",
                y_column="y",
                freq="D",
                horizon="forever",
                series_name="s",
                regressors=[],
            )
