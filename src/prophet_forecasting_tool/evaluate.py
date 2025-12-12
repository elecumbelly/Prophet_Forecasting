import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
from prophet.diagnostics import cross_validation, performance_metrics

from prophet_forecasting_tool.model import train_prophet_model, forecast_with_prophet, get_uk_bank_holidays

logger = logging.getLogger(__name__)

def calculate_mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Calculates Mean Absolute Error (MAE)."""
    return mean_absolute_error(y_true, y_pred)

def calculate_mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Calculates Mean Absolute Percentage Error (MAPE)."""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def calculate_rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Calculates Root Mean Squared Error (RMSE)."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

METRIC_FUNCTIONS = {
    "mae": calculate_mae,
    "mape": calculate_mape,
    "rmse": calculate_rmse,
}

def evaluate_with_cross_validation(
    df: pd.DataFrame,
    freq: str = "D",
    metrics: Optional[List[str]] = None,
    initial: str = '730 days',
    period: str = '180 days',
    horizon: str = '365 days',
    n_jobs: int = -1, # Add n_jobs parameter
    **prophet_kwargs,
) -> Dict[str, float]:
    """
    Performs evaluation using Prophet's built-in cross-validation.

    Args:
        df: DataFrame with 'ds' and 'y' columns.
        freq: Frequency of the time series.
        metrics: List of metrics to calculate (e.g., ["mae", "mape", "rmse"]).
        initial: Initial training period string (e.g., '730 days').
        period: Period between cutoffs string (e.g., '180 days').
        horizon: Forecast horizon string (e.g., '365 days').
        n_jobs: Number of parallel jobs to run for cross-validation. -1 means use all available CPUs.
        **prophet_kwargs: Additional keyword arguments for Prophet.

    Returns:
        A dictionary of calculated metrics (aggregated mean).
    """
    if df.empty:
        raise ValueError("Input DataFrame for evaluation cannot be empty.")
    
    # Train model on ALL data
    holidays_df = None
    if prophet_kwargs.get("support_uk_holidays", False):
        holidays_df = get_uk_bank_holidays()
        prophet_kwargs.pop("support_uk_holidays", None) 
    
    logger.info(f"Training model on full dataset ({len(df)} rows) for Cross-Validation...")
    model = train_prophet_model(df, freq=freq, holidays_df=holidays_df, **prophet_kwargs)

    # Run Cross-Validation
    logger.info(f"Starting Cross-Validation (initial={initial}, period={period}, horizon={horizon}, n_jobs={n_jobs})...")
    try:
        # Determine parallel argument for cross_validation
        parallel_arg = "processes" if n_jobs == -1 else n_jobs
        
        # Suppress some prophet logging if possible
        with logging.getLogger('prophet').handlers[0].lock if logging.getLogger('prophet').handlers else pd.option_context('mode.chained_assignment', None):
            df_cv = cross_validation(
                model, 
                initial=initial, 
                period=period, 
                horizon=horizon,
                disable_tqdm=True, # Disable progress bar for logs
                parallel=parallel_arg
            )
    except Exception as e:
        logger.error(f"Cross-Validation failed: {e}")
        # It usually fails if not enough data.
        raise ValueError(f"Cross-Validation failed. Check if data is sufficient for initial '{initial}'. Error: {e}")

    # Calculate Metrics
    metrics_to_calculate = metrics if metrics else ["mae", "mape", "rmse"]
    
    # performance_metrics returns a dataframe with metrics for different horizon windows
    df_p = performance_metrics(df_cv, metrics=metrics_to_calculate, rolling_window=1)
    
    results = {}
    for metric in metrics_to_calculate:
        if metric in df_p.columns:
            # We report the mean across all horizons for a summary scalar
            metric_val = df_p[metric].mean()
            results[metric] = metric_val
            logger.info(f"CV {metric.upper()}: {metric_val:.4f}")
        else:
            logger.warning(f"Metric '{metric}' not found in performance_metrics output.")

    return results
