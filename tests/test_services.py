import unittest
from unittest.mock import patch, MagicMock, call
import pandas as pd
import pathlib
import hashlib
import os
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from prophet_forecasting_tool.services import ForecastingService
from prophet_forecasting_tool.config import Settings



class TestForecastingService(unittest.TestCase):

    @patch.object(pathlib, 'Path') # Patch the Path class itself
    def setUp(self, MockPathClass): # MockPathClass is the mocked pathlib.Path
        self.mock_engine = MagicMock()
        self.mock_settings = MagicMock(spec=Settings)

        # Mock an instance of Path for self.models_dir
        self.mock_models_dir = MockPathClass.return_value 
        self.mock_models_dir.mkdir = MagicMock()
        self.mock_models_dir.__str__.return_value = "/mock/models" # For logging/debug

        # When self.mock_models_dir / "filename" is called, it should return another mock Path instance
        self.mock_model_file_path = MagicMock() # Create a separate mock for the file path, without spec
        self.mock_model_file_path.exists.return_value = False # Default: file does not exist
        self.mock_model_file_path.__str__.return_value = "/mock/models/mocked_hash.json"

        # Configure the __truediv__ method of self.mock_models_dir to return our mock_model_file_path
        self.mock_models_dir.__truediv__.return_value = self.mock_model_file_path
        
        # Patching hashlib.md5
        self.patcher_md5 = patch('hashlib.md5')
        self.mock_md5 = self.patcher_md5.start()
        self.mock_md5.return_value.hexdigest.return_value = 'mocked_hash'

        self.service = ForecastingService(self.mock_engine, self.mock_settings, self.mock_models_dir)

        # Common mock data
        self.mock_df = pd.DataFrame({
            'ds': pd.to_datetime(['2020-01-01', '2020-01-02', '2022-01-01', '2022-01-02']),
            'y': [10, 20, 30, 40]
        })
        self.mock_df['ds'] = self.mock_df['ds'].dt.tz_localize(None)


        self.mock_train_df = pd.DataFrame({
            'ds': pd.to_datetime(['2020-01-01', '2020-01-02']),
            'y': [10, 20]
        })
        self.mock_train_df['ds'] = self.mock_train_df['ds'].dt.tz_localize(None)

        self.mock_forecast_df = pd.DataFrame({
            'ds': pd.to_datetime(['2023-01-01', '2023-01-02']),
            'yhat': [100, 110]
        })
        self.mock_forecast_df['ds'] = self.mock_forecast_df['ds'].dt.tz_localize(None)

        self.mock_model = MagicMock()
        self.mock_fig = MagicMock()
        self.mock_fig.savefig = MagicMock()
        self.mock_model.plot.return_value = self.mock_fig
        self.mock_model.plot_components.return_value = self.mock_fig

    def tearDown(self):
        self.patcher_md5.stop()

    @patch('prophet_forecasting_tool.services.load_time_series', return_value=pd.DataFrame())
    @patch('prophet_forecasting_tool.services.get_max_date_for_table', return_value=pd.Timestamp('2023-01-01'))
    def test_get_forecast_empty_data_raises_error(self, mock_get_max_date, mock_load_time_series):
        """Test get_forecast raises ValueError if no data is loaded."""
        with self.assertRaises(ValueError, msg="Expected ValueError for empty data"):
            self.service.get_forecast(
                table_name='test', ts_column='ts', y_column='y', freq='D', horizon='1D', series_name='s', regressors=[]
            )
        mock_load_time_series.assert_called_once()
        mock_get_max_date.assert_called_once()
    
    @patch('prophet_forecasting_tool.services.load_time_series')
    @patch('prophet_forecasting_tool.services.train_prophet_model')
    @patch('prophet_forecasting_tool.services.forecast_with_prophet')
    @patch('prophet_forecasting_tool.services.save_model')
    @patch('prophet_forecasting_tool.services.load_model')
    @patch('prophet_forecasting_tool.services.get_uk_bank_holidays', return_value=pd.DataFrame())
    @patch('prophet_forecasting_tool.services.get_max_date_for_table', return_value=pd.Timestamp('2023-01-01')) # New patch
    def test_get_forecast_no_cache_no_tune(self, mock_get_max_date, mock_get_holidays, mock_load_model, mock_save_model, mock_forecast_with_prophet, mock_train_prophet_model, mock_load_time_series):
        """Test forecast generation when no cached model exists and no tuning is requested."""
        mock_load_time_series.return_value = self.mock_df
        mock_train_prophet_model.return_value = self.mock_model
        mock_forecast_with_prophet.return_value = self.mock_forecast_df
        self.mock_model_file_path.exists.return_value = False # Simulate no cached model

        result = self.service.get_forecast(
            table_name='test', ts_column='ds', y_column='y', freq='D', horizon='30D', series_name='s', regressors=[],
            resample_to_freq=None, training_window_duration="730 days"
        )

        # load_time_series is called twice: once for training data, once for actual_calls (for plotting)
        self.assertEqual(mock_load_time_series.call_count, 2)
        mock_train_prophet_model.assert_called_once()
        mock_save_model.assert_called_once()
        mock_load_model.assert_not_called()
        mock_forecast_with_prophet.assert_called_once_with(self.mock_model, periods=30, freq='D', future_regressors_df=self.mock_df)

        self.assertIn('forecast_df', result)
        self.assertIn('forecast_plot', result)
        self.assertIn('components_plot', result)

    @patch('prophet_forecasting_tool.services.load_time_series')
    @patch('prophet_forecasting_tool.services.train_prophet_model')
    @patch('prophet_forecasting_tool.services.forecast_with_prophet')
    @patch('prophet_forecasting_tool.services.save_model')
    @patch('prophet_forecasting_tool.services.load_model')
    @patch('prophet_forecasting_tool.services.get_uk_bank_holidays', return_value=pd.DataFrame())
    @patch('prophet_forecasting_tool.services.get_max_date_for_table', return_value=pd.Timestamp('2023-01-01')) # New patch
    def test_get_forecast_with_cache_no_tune(self, mock_get_max_date, mock_get_holidays, mock_load_model, mock_save_model, mock_forecast_with_prophet, mock_train_prophet_model, mock_load_time_series):
        """Test forecast generation when a cached model exists and no tuning."""
        mock_load_time_series.return_value = self.mock_df
        mock_train_prophet_model.return_value = self.mock_model
        mock_forecast_with_prophet.return_value = self.mock_forecast_df
        self.mock_model_file_path.exists.return_value = True # Simulate cached model exists
        mock_load_model.return_value = self.mock_model

        result = self.service.get_forecast(
            table_name='test', ts_column='ds', y_column='y', freq='D', horizon='30D', series_name='s', regressors=[],
            resample_to_freq=None, training_window_duration="730 days"
        )

        # load_time_series is called twice: once for training data, once for actual_calls (for plotting)
        self.assertEqual(mock_load_time_series.call_count, 2)
        mock_train_prophet_model.assert_not_called() # Should not train
        mock_save_model.assert_not_called() # Should not save
        mock_load_model.assert_called_once() # Should load
        mock_forecast_with_prophet.assert_called_once()

        self.assertIn('forecast_df', result)

    @patch('prophet_forecasting_tool.services.load_time_series')
    @patch('prophet_forecasting_tool.services.train_prophet_model')
    @patch('prophet_forecasting_tool.services.forecast_with_prophet')
    @patch('prophet_forecasting_tool.services.save_model')
    @patch('prophet_forecasting_tool.services.load_model')
    @patch('prophet_forecasting_tool.services.tune_hyperparameters')
    @patch('prophet_forecasting_tool.services.get_uk_bank_holidays', return_value=pd.DataFrame())
    @patch('prophet_forecasting_tool.services.get_max_date_for_table', return_value=pd.Timestamp('2023-01-01')) # New patch
    def test_get_forecast_no_cache_with_tune(self, mock_get_max_date, mock_get_holidays, mock_tune_hyperparameters, mock_load_model, mock_save_model, mock_forecast_with_prophet, mock_train_prophet_model, mock_load_time_series):
        """Test forecast generation when no cached model and tuning is requested."""
        mock_load_time_series.return_value = self.mock_df
        self.mock_model_file_path.exists.return_value = False # Simulate no cached model
        mock_tune_hyperparameters.return_value = {'changepoint_prior_scale': 0.1} # Simulate best params
        mock_train_prophet_model.return_value = self.mock_model
        mock_forecast_with_prophet.return_value = self.mock_forecast_df

        result = self.service.get_forecast(
            table_name='test', ts_column='ds', y_column='y', freq='D', horizon='30D', series_name='s', regressors=[], auto_tune=True,
            resample_to_freq=None, training_window_duration="730 days"
        )

        mock_tune_hyperparameters.assert_called_once()
        mock_train_prophet_model.assert_called_once()
        self.assertIn('changepoint_prior_scale', mock_train_prophet_model.call_args[1])
        mock_save_model.assert_called_once()
        mock_load_model.assert_not_called()
        self.assertIn('forecast_df', result)

    @patch('prophet_forecasting_tool.services.load_time_series')
    @patch('prophet_forecasting_tool.services.evaluate_with_cross_validation')
    @patch('prophet_forecasting_tool.services.get_max_date_for_table', return_value=pd.Timestamp('2023-01-01')) # New patch
    def test_evaluate_model_success(self, mock_get_max_date, mock_evaluate_cv, mock_load_time_series):
        """Test evaluate_model successfully calls evaluation and saves metrics."""
        mock_load_time_series.return_value = self.mock_df
        mock_evaluate_cv.return_value = {'mae': 10.5, 'rmse': 15.2}

        mock_output_dir = MagicMock(spec=pathlib.Path)
        
        # Mock the path object that will be returned by mock_output_dir / "filename"
        mock_metrics_output_path = MagicMock(spec=pathlib.Path)
        mock_metrics_output_path.__str__.return_value = "/mock/output/metrics.json"
        
        # Mock the .parent attribute of mock_metrics_output_path
        mock_parent_path = MagicMock(spec=pathlib.Path)
        mock_parent_path.mkdir = MagicMock()
        mock_metrics_output_path.parent = mock_parent_path
        
        # Configure __truediv__ of mock_output_dir to return mock_metrics_output_path
        mock_output_dir.__truediv__.return_value = mock_metrics_output_path
        
        # Patch open to avoid actual file write
        with patch('builtins.open', MagicMock()) as mock_open:
            result = self.service.evaluate_model(
                table_name='test', ts_column='ds', y_column='y', freq='D', metrics=['mae'],
                initial='1D', period='1D', horizon='1D', regressors=[], output_dir=mock_output_dir
            )

            mock_load_time_series.assert_called_once()
            mock_evaluate_cv.assert_called_once_with(
                self.mock_df, 
                freq='D', 
                metrics=['mae'], 
                initial='1D', 
                period='1D', 
                horizon='1D',
                n_jobs=-1,
                regressors=[],
                save_path=None
            )
            self.assertEqual(result, {'mae': 10.5, 'rmse': 15.2})
            mock_parent_path.mkdir.assert_called_once_with(parents=True, exist_ok=True) # Assert on the parent mock
            mock_open.assert_called_once() # Verify file was attempted to be opened

    @patch('prophet_forecasting_tool.services.load_time_series', return_value=pd.DataFrame())
    @patch('prophet_forecasting_tool.services.get_max_date_for_table', return_value=pd.Timestamp('2023-01-01'))
    def test_evaluate_model_empty_data_raises_error(self, mock_get_max_date, mock_load_time_series):
        """Test evaluate_model raises ValueError if no data is loaded."""
        mock_output_dir = MagicMock(spec=pathlib.Path)

        with self.assertRaises(ValueError, msg="Expected ValueError for empty data"):
            self.service.evaluate_model(
                table_name='test', ts_column='ds', y_column='y', freq='D', metrics=['mae'],
                initial='1D', period='1D', horizon='1D', regressors=[], output_dir=mock_output_dir
            )
        mock_load_time_series.assert_called_once()
        mock_get_max_date.assert_called_once()

if __name__ == '__main__':
    unittest.main()
