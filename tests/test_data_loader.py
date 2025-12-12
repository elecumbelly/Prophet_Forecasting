import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine

# Assuming data_loader.py is in the parent directory of tests/
from prophet_forecasting_tool.data_loader import load_time_series

class TestDataLoader(unittest.TestCase):

    @patch('prophet_forecasting_tool.data_loader.pd.read_sql')
    def test_load_time_series_success(self, mock_read_sql):
        # Mocking pd.read_sql to return a sample DataFrame
        sample_data = {
            'ds': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03']),
            'y': [10, 20, 15]
        }
        mock_read_sql.return_value = pd.DataFrame(sample_data)

        mock_engine = MagicMock(spec=create_engine("sqlite:///:memory:"))
        df = load_time_series(mock_engine, table="test_table")

        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertIn('ds', df.columns)
        self.assertIn('y', df.columns)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df['ds']))
        mock_read_sql.assert_called_once()

    @patch('prophet_forecasting_tool.data_loader.pd.read_sql')
    def test_load_time_series_with_filters(self, mock_read_sql):
        sample_data = {
            'ds': pd.to_datetime(['2020-01-01', '2020-01-02']),
            'y': [100, 110]
        }
        mock_read_sql.return_value = pd.DataFrame(sample_data)

        mock_engine = MagicMock(spec=create_engine("sqlite:///:memory:"))
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2020, 1, 3)
        filters = {"queue": "sales"}

        df = load_time_series(
            mock_engine,
            table="test_table",
            start=start_date,
            end=end_date,
            filters=filters
        )

        self.assertFalse(df.empty)
        
        # Args[0] is now a SQLAlchemy Select object
        args, kwargs = mock_read_sql.call_args
        query_obj = args[0]
        query_str = str(query_obj)
        
        # Check for general SQL structure
        self.assertIn("SELECT", query_str)
        self.assertIn("FROM test_table", query_str)
        # SQLAlchemy generates bind params like :ts_1, :ts_2, :queue_1
        self.assertIn("test_table.ts >=", query_str)
        self.assertIn("test_table.ts <=", query_str)
        self.assertIn("test_table.queue =", query_str)
        
        # Verify parameters are correctly bound
        compiled = query_obj.compile()
        # compiled.params is a dict of {param_name: value}
        param_values = list(compiled.params.values())
        
        self.assertIn(start_date, param_values)
        self.assertIn(end_date, param_values)
        self.assertIn("sales", param_values)

    @patch('prophet_forecasting_tool.data_loader.pd.read_sql')
    def test_load_time_series_empty_result(self, mock_read_sql):
        mock_read_sql.return_value = pd.DataFrame(columns=['ds', 'y'])
        mock_engine = MagicMock(spec=create_engine("sqlite:///:memory:"))
        df = load_time_series(mock_engine)
        self.assertTrue(df.empty)

    def test_load_time_series_invalid_engine(self):
        with self.assertRaises(TypeError):
            load_time_series("invalid_engine")

if __name__ == '__main__':
    unittest.main()
