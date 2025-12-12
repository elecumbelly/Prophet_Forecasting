import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import pandas as pd
import io
import base64

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Patch dependencies BEFORE importing app to handle side-effects (like create_engine)
with patch('sqlalchemy.create_engine') as mock_create_engine, \
     patch('prophet_forecasting_tool.config.Settings') as mock_settings:
    
    # Configure mocks
    mock_settings.return_value.DATABASE_URL = "postgresql://user:pass@localhost/db"
    
    from prophet_forecasting_tool.app import app, engine

class TestUI(unittest.TestCase):
    def setUp(self):
        self.app = app
        self.app.config['TESTING'] = True
        self.app.config['SECRET_KEY'] = 'test_secret_key'
        self.client = self.app.test_client()
        
    def test_index_page(self):
        """Test the home page loads correctly."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Prophet Forecasting Tool', response.data)

    @patch('prophet_forecasting_tool.app.load_time_series')
    @patch('prophet_forecasting_tool.app.inspect')
    def test_historical_data_post_success(self, mock_inspect, mock_load_data):
        """Test historical data page with valid POST request."""
        mock_inspector = MagicMock()
        mock_inspect.return_value = mock_inspector
        mock_inspector.has_table.return_value = True
        mock_inspector.get_columns.return_value = [{'name': 'ds'}, {'name': 'y'}, {'name': 'extra'}]
        all_columns = ['ds', 'y', 'extra']

        # Setup mock data
        mock_df = pd.DataFrame({
            'ds': pd.date_range(start='2020-01-01', periods=5),
            'y': [100, 110, 120, 130, 140],
            'extra': [1,2,3,4,5]
        })
        mock_load_data.return_value = mock_df
        
        response = self.client.post('/historical_data', data={
            'table': 'test_table',
            'ts_column': 'ds',
            'y_column': 'y'
        }, follow_redirects=True)
        
        self.assertEqual(response.status_code, 200)
        # Check if table data is rendered (simple check for a known value)
        self.assertIn(b'100', response.data)
        self.assertIn(b'Successfully loaded 5 rows', response.data)
        mock_load_data.assert_called_once_with(
            engine, table='test_table', ts_column='ds', y_column='y', columns_to_load=all_columns
        )

    @patch('prophet_forecasting_tool.app.load_time_series')
    @patch('prophet_forecasting_tool.app.inspect')
    def test_historical_data_post_empty(self, mock_inspect, mock_load_data):
        """Test historical data page with empty result."""
        mock_inspector = MagicMock()
        mock_inspect.return_value = mock_inspector
        mock_inspector.has_table.return_value = True
        mock_inspector.get_columns.return_value = [{'name': 'ds'}, {'name': 'y'}]
        all_columns = ['ds', 'y']

        mock_load_data.return_value = pd.DataFrame()
        
        response = self.client.post('/historical_data', data={
            'table': 'empty_table'
        }, follow_redirects=True)
        
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'No data found', response.data)
        mock_load_data.assert_called_once_with(
            engine, table='empty_table', ts_column='ds', y_column='y', columns_to_load=all_columns
        )

    @patch('prophet_forecasting_tool.app.forecasting_service')
    def test_forecast_route_calls_service(self, mock_forecasting_service):
        """Test that the /forecast route correctly calls the service and renders the template."""
        # Mock the service's get_forecast method
        mock_get_forecast_return = {
            "forecast_df": pd.DataFrame({'ds': ['2023-01-01'], 'yhat': [100]}),
            "series_name": "mock_series",
            "forecast_plot": "data:image/png;base64,mock_forecast_plot",
            "components_plot": "data:image/png;base64,mock_components_plot",
        }
        mock_forecasting_service.get_forecast.return_value = mock_get_forecast_return

        # Prepare form data
        form_data = {
            'table': 'mock_table',
            'ts_column': 'mock_ts',
            'y_column': 'mock_y',
            'freq': 'D',
            'horizon': '30D',
            'series_name': 'mock_series',
            'regressors': ['reg1', 'reg2'],
            'auto_tune': 'on',
            'resample_to_freq': 'H', # New form data
            'training_window_duration': '365 days', # New form data
        }

        response = self.client.post('/forecast', data=form_data, follow_redirects=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Forecast generated successfully!', response.data)
        self.assertIn(b'mock_forecast_plot', response.data)
        self.assertIn(b'mock_components_plot', response.data)
        self.assertIn(b'mock_series', response.data)

        # Verify get_forecast was called with the correct arguments
        mock_forecasting_service.get_forecast.assert_called_once_with(
            table_name='mock_table',
            ts_column='mock_ts',
            y_column='mock_y',
            freq='D',
            horizon='30D',
            series_name='mock_series',
            regressors=['reg1', 'reg2'],
            auto_tune=True,
            n_jobs=-1, # Default from app.py
            resample_to_freq='H', # Expected
            training_window_duration='365 days', # Expected
        )

    @patch('prophet_forecasting_tool.app.inspect')
    def test_get_columns(self, mock_inspect):
        """Test the get_columns route."""
        mock_inspector = MagicMock()
        mock_inspect.return_value = mock_inspector
        
        # Case 1: Table exists
        mock_inspector.has_table.return_value = True
        mock_inspector.get_columns.return_value = [{'name': 'ts'}, {'name': 'y'}, {'name': 'extra'}]
        
        response = self.client.get('/get_columns/test_table')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {'columns': ['ts', 'y', 'extra']})
        
        # Case 2: Table does not exist
        mock_inspector.has_table.return_value = False
        response = self.client.get('/get_columns/missing_table')
        self.assertEqual(response.status_code, 404)
