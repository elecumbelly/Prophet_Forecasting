import unittest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, mock_open

# Assuming model.py is in the correct path relative to tests/
from prophet_forecasting_tool.model import train_prophet_model, forecast_with_prophet, get_uk_bank_holidays, save_model, load_model

# Mock the Prophet dependency to avoid actual model fitting
class MockProphet:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.df = None

    def fit(self, df):
        self.df = df

    def make_future_dataframe(self, periods, freq, include_history=True):
        last_date = self.df['ds'].max()
        # Mock logic for future dates
        if freq == 'D':
            offset = pd.Timedelta(days=1)
            future_dates = pd.date_range(start=last_date + offset, periods=periods, freq=freq)
        else:
            # Simplified mock handling for other freqs
            future_dates = pd.date_range(start=last_date, periods=periods + 1, freq=freq)[1:]
            
        future_dates_series = pd.Series(future_dates)
        if include_history:
            return pd.DataFrame({'ds': pd.concat([self.df['ds'], future_dates_series])})
        return pd.DataFrame({'ds': future_dates_series})

    def predict(self, future_df):
        # Return a dummy forecast DataFrame
        return pd.DataFrame({
            'ds': future_df['ds'],
            'yhat': [100.0] * len(future_df),
            'yhat_lower': [90.0] * len(future_df),
            'yhat_upper': [110.0] * len(future_df),
        })
    
    def plot(self, forecast_df):
        # Mock matplotlib figure
        mock_fig = MagicMock()
        mock_fig.savefig = MagicMock()
        return mock_fig
        
    def plot_components(self, forecast_df):
        # Mock matplotlib figure for components
        mock_fig = MagicMock()
        mock_fig.savefig = MagicMock()
        return mock_fig


class TestModel(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame for testing
        self.sample_df = pd.DataFrame({
            'ds': [datetime(2020, 1, 1) + timedelta(days=i) for i in range(365 * 2)], # Two years of data
            'y': [100 + i + (i % 7) * 5 for i in range(365 * 2)]
        })

    @patch('prophet_forecasting_tool.model.Prophet', new=MockProphet)
    def test_train_prophet_model_success(self):
        model = train_prophet_model(self.sample_df)
        self.assertIsInstance(model, MockProphet)
        self.assertIsNotNone(model.df)
        pd.testing.assert_frame_equal(model.df, self.sample_df)

    def test_train_prophet_model_empty_df(self):
        with self.assertRaises(ValueError):
            train_prophet_model(pd.DataFrame(columns=['ds', 'y']))

    def test_train_prophet_model_missing_columns(self):
        with self.assertRaises(ValueError):
            train_prophet_model(pd.DataFrame({'date': [], 'value': []}))

    @patch('prophet_forecasting_tool.model.Prophet', new=MockProphet)
    def test_forecast_with_prophet_success(self):
        model = MockProphet()
        model.fit(self.sample_df) # Fit the mock model

        # Updated to pass periods instead of horizon string
        forecast_df = forecast_with_prophet(model, periods=30, freq="D")
        self.assertIsInstance(forecast_df, pd.DataFrame)
        self.assertFalse(forecast_df.empty)
        self.assertIn('ds', forecast_df.columns)
        self.assertIn('yhat', forecast_df.columns)
        self.assertIn('yhat_lower', forecast_df.columns)
        self.assertIn('yhat_upper', forecast_df.columns)
        
        # The forecast_df from prophet.predict will include historical dates as well.
        # We are only interested in the future dates for this assertion.
        last_training_date = self.sample_df['ds'].max()
        future_forecast_df = forecast_df[forecast_df['ds'] > last_training_date]
        self.assertEqual(len(future_forecast_df), 30) # 30 days forecast



    def test_get_uk_bank_holidays(self):
        holidays_df = get_uk_bank_holidays()
        self.assertIsInstance(holidays_df, pd.DataFrame)
        self.assertFalse(holidays_df.empty)
        self.assertIn('ds', holidays_df.columns)
        self.assertIn('holiday', holidays_df.columns)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(holidays_df['ds']))

    @patch('prophet_forecasting_tool.model.model_to_json')
    @patch('prophet_forecasting_tool.model.model_from_json')
    @patch('prophet_forecasting_tool.model.json')
    def test_save_and_load_model(self, mock_json, mock_from_json, mock_to_json):
        mock_model = MagicMock()
        model_json_str = '{"model": "json"}'
        mock_to_json.return_value = model_json_str
        mock_from_json.return_value = mock_model
        
        # Setup json mock
        mock_json.load.return_value = model_json_str
        
        path = "test_model.json"
        
        # Test Save
        with patch("builtins.open", mock_open()) as mock_file:
            save_model(mock_model, path)
            mock_file.assert_called_with(path, 'w')
            # Verify json.dump was called with the string from model_to_json
            mock_json.dump.assert_called_with(model_json_str, mock_file())
            mock_to_json.assert_called_with(mock_model)

        # Test Load
        with patch("builtins.open", mock_open()) as mock_file:
            loaded_model = load_model(path)
            mock_file.assert_called_with(path, 'r')
            mock_json.load.assert_called_with(mock_file())
            mock_from_json.assert_called_with(model_json_str)
            self.assertEqual(loaded_model, mock_model)

if __name__ == '__main__':
    unittest.main()
