import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from prophet_forecasting_tool.evaluate import (
    calculate_mae,
    calculate_mape,
    calculate_rmse,
    evaluate_with_cross_validation,
)

class TestEvaluation(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame
        self.sample_df = pd.DataFrame({
            'ds': [datetime(2020, 1, 1) + timedelta(days=i) for i in range(100)],
            'y': [100 + i for i in range(100)]
        })

    def test_calculate_mae(self):
        y_true = pd.Series([10, 20, 30])
        y_pred = pd.Series([11, 19, 32])
        self.assertAlmostEqual(calculate_mae(y_true, y_pred), (1+1+2)/3)

    def test_calculate_mape(self):
        y_true = pd.Series([10, 20, 30])
        y_pred = pd.Series([11, 19, 32])
        expected_mape = (0.1 + 0.05 + 2/30) / 3 * 100
        self.assertAlmostEqual(calculate_mape(y_true, y_pred), expected_mape)

    def test_calculate_rmse(self):
        y_true = pd.Series([10, 20, 30])
        y_pred = pd.Series([11, 19, 32])
        expected_rmse = np.sqrt(2)
        self.assertAlmostEqual(calculate_rmse(y_true, y_pred), expected_rmse)

    @patch('prophet_forecasting_tool.evaluate.train_prophet_model')
    @patch('prophet_forecasting_tool.evaluate.cross_validation')
    @patch('prophet_forecasting_tool.evaluate.performance_metrics')
    def test_evaluate_with_cross_validation_success(self, mock_perf, mock_cv, mock_train):
        # Mock training
        mock_model = MagicMock()
        mock_train.return_value = mock_model
        
        # Mock CV output (structure doesn't matter much as long as it's passed to perf)
        mock_df_cv = pd.DataFrame({'ds': [], 'y': [], 'yhat': [], 'cutoff': []})
        mock_cv.return_value = mock_df_cv
        
        # Mock performance metrics output
        # Return multiple rows to test averaging
        mock_df_p = pd.DataFrame({
            'horizon': ['30 days', '60 days'],
            'mae': [10.0, 20.0],
            'rmse': [12.0, 22.0]
        })
        mock_perf.return_value = mock_df_p
        
        results = evaluate_with_cross_validation(self.sample_df, metrics=['mae', 'rmse'])
        
        self.assertEqual(results['mae'], 15.0) # Mean of 10 and 20
        self.assertEqual(results['rmse'], 17.0) # Mean of 12 and 22
        
        mock_train.assert_called()
        mock_cv.assert_called()
        mock_perf.assert_called()

    def test_evaluate_with_cross_validation_empty_df(self):
        with self.assertRaises(ValueError):
            evaluate_with_cross_validation(pd.DataFrame(columns=['ds', 'y']))

if __name__ == '__main__':
    unittest.main()