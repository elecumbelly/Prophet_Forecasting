from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, mock_open

import pandas as pd
import pytest

from prophet_forecasting_tool.model import (
    forecast_with_prophet,
    get_uk_bank_holidays,
    load_model,
    save_model,
    train_prophet_model,
)


class MockProphet:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.df = None
        self.extra_regressors = {}

    def fit(self, df):
        self.df = df

    def add_regressor(self, name):
        self.extra_regressors[name] = True

    def make_future_dataframe(self, periods, freq, include_history=True):
        last_date = self.df["ds"].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq=freq)
        future_dates_series = pd.Series(future_dates)
        if include_history:
            return pd.DataFrame({"ds": pd.concat([self.df["ds"], future_dates_series])}).reset_index(drop=True)
        return pd.DataFrame({"ds": future_dates_series})

    def predict(self, future_df):
        return pd.DataFrame({
            "ds": future_df["ds"],
            "yhat": [100.0] * len(future_df),
            "yhat_lower": [90.0] * len(future_df),
            "yhat_upper": [110.0] * len(future_df),
        })


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "ds": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(365 * 2)],
        "y": [100 + i + (i % 7) * 5 for i in range(365 * 2)],
    })


class TestTrainModel:
    @patch("prophet_forecasting_tool.model.Prophet", new=MockProphet)
    def test_success(self, sample_df):
        model = train_prophet_model(sample_df)
        assert isinstance(model, MockProphet)
        assert model.df is not None

    def test_empty_df_raises(self):
        with pytest.raises(ValueError):
            train_prophet_model(pd.DataFrame(columns=["ds", "y"]))

    def test_missing_cols_raises(self):
        with pytest.raises(ValueError):
            train_prophet_model(pd.DataFrame({"date": [], "value": []}))

    @patch("prophet_forecasting_tool.model.Prophet", new=MockProphet)
    def test_unknown_regressor_raises(self, sample_df):
        with pytest.raises(ValueError):
            train_prophet_model(sample_df, regressors=["not_a_column"])


class TestForecast:
    @patch("prophet_forecasting_tool.model.Prophet", new=MockProphet)
    def test_basic(self, sample_df):
        m = MockProphet()
        m.fit(sample_df)
        out = forecast_with_prophet(m, periods=30, freq="D")
        assert "yhat" in out.columns
        future_only = out[out["ds"] > sample_df["ds"].max()]
        assert len(future_only) == 30

    @patch("prophet_forecasting_tool.model.Prophet", new=MockProphet)
    def test_missing_future_regressors_raises(self, sample_df):
        m = MockProphet()
        m.fit(sample_df)
        m.extra_regressors = {"answer_rate": True}
        with pytest.raises(ValueError, match="future_regressors_df"):
            forecast_with_prophet(m, periods=30, freq="D")


class TestUkBankHolidays:
    def test_shape(self):
        df = get_uk_bank_holidays()
        assert not df.empty
        assert {"ds", "holiday", "lower_window", "upper_window"}.issubset(df.columns)

    def test_christmas_has_wider_window(self):
        df = get_uk_bank_holidays(years=[2024])
        christmas_rows = df[df["holiday"].str.contains("Christmas Day", case=False, na=False)]
        assert not christmas_rows.empty
        assert (christmas_rows["lower_window"] == -1).all()
        assert (christmas_rows["upper_window"] == 2).all()

    def test_year_range_respected(self):
        df = get_uk_bank_holidays(years=[2022])
        assert df["ds"].dt.year.unique().tolist() == [2022]


class TestSaveLoadModel:
    @patch("prophet_forecasting_tool.model.model_to_json")
    @patch("prophet_forecasting_tool.model.model_from_json")
    @patch("prophet_forecasting_tool.model.json")
    def test_roundtrip(self, mock_json, mock_from_json, mock_to_json):
        mock_model = MagicMock()
        model_json_str = '{"model": "json"}'
        mock_to_json.return_value = model_json_str
        mock_from_json.return_value = mock_model
        mock_json.load.return_value = model_json_str

        with patch("builtins.open", mock_open()):
            save_model(mock_model, "test_model.json")

        with patch("builtins.open", mock_open()):
            loaded = load_model("test_model.json")
            assert loaded is mock_model
