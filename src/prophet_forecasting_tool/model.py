import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
from prophet.diagnostics import cross_validation, performance_metrics
from typing import Any, Dict, Optional, List, Union
import logging
import re
import json
import itertools
import joblib # Import joblib for parallel processing
import holidays # Keep this import if get_uk_bank_holidays is used within here directly, otherwise it's in data_loader

logger = logging.getLogger(__name__)

def save_model(model: Prophet, filepath: str) -> None:
    """
    Saves a trained Prophet model to a JSON file.
    
    Args:
        model: The trained Prophet model.
    filepath: The path where the model should be saved.
    """
    logger.info(f"Saving model to {filepath}...")
    with open(filepath, 'w') as f:
        json.dump(model_to_json(model), f)
    logger.info("Model saved successfully.")

def load_model(filepath: str) -> Prophet:
    """
    Loads a Prophet model from a JSON file.
    
    Args:
        filepath: The path to the saved model JSON file.
        
    Returns:
        The loaded Prophet model.
    """
    logger.info(f"Loading model from {filepath}...")
    with open(filepath, 'r') as f:
        model_json = json.load(f)
    model = model_from_json(model_json)
    logger.info("Model loaded successfully.")
    return model

def _evaluate_params(df, current_kwargs, initial, period, horizon, metric):
    """Helper function to evaluate a single set of hyperparameters."""
    # Ensure df is a copy to avoid unintended modifications in parallel processes
    df_copy = df.copy() 
    
    # Ensure we don't save models during tuning within this helper
    temp_kwargs = current_kwargs.copy()
    if 'save_path' in temp_kwargs:
        del temp_kwargs['save_path']
    
    try:
        # train_prophet_model handles holidays_df and regressors within kwargs
        m = train_prophet_model(df_copy, **temp_kwargs)
        
        # Suppress prophet output during CV to avoid log spam
        with logging.getLogger('prophet').handlers[0].lock if logging.getLogger('prophet').handlers else pd.option_context('mode.chained_assignment', None):
             df_cv = cross_validation(
                m, 
                initial=initial, 
                period=period, 
                horizon=horizon,
                disable_tqdm=True,
                # parallel="processes" # Let tune_hyperparameters manage parallelism
            )
        
        df_p = performance_metrics(df_cv, rolling_window=1)
        metric_value = df_p[metric].mean()
        
        # logger.debug(f"Evaluated params: {current_kwargs}, {metric}: {metric_value:.4f}")
        
        return metric_value, {k: v for k, v in temp_kwargs.items() if k in ['changepoint_prior_scale', 'seasonality_prior_scale']} # Return only the tuned params
        
    except Exception as e:
        logger.warning(f"Tuning failed for params {current_kwargs}: {e}")
        return float('inf'), {k: v for k, v in temp_kwargs.items() if k in ['changepoint_prior_scale', 'seasonality_prior_scale']} # Return inf for failed runs

def tune_hyperparameters(
    df: pd.DataFrame,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    metric: str = 'rmse',
    initial: str = '730 days',
    period: str = '180 days',
    horizon: str = '365 days',
    n_jobs: int = -1, # Number of jobs for parallel processing, -1 means all CPUs
    **fixed_kwargs
) -> Dict[str, Any]:
    """
    Performs hyperparameter tuning using cross-validation.
    
    Args:
        df: Training DataFrame.
        param_grid: Dictionary of parameters to tune. Defaults to changepoint/seasonality priors.
        metric: Metric to minimize ('rmse', 'mae', 'mape', etc.).
        initial: Initial training period for CV.
        period: Period between cutoffs for CV.
        horizon: Forecast horizon for CV.
        n_jobs: Number of parallel jobs to run. -1 means use all available CPUs.
        **fixed_kwargs: Fixed arguments to pass to train_prophet_model (e.g. regressors, holidays_df, freq).
        
    Returns:
        Dictionary of the best hyperparameters.
    """
    if param_grid is None:
        param_grid = {
            'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
            'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
        }

    # Generate all combinations
    keys = param_grid.keys()
    combinations_kwargs = [{k: v for k, v in zip(keys, values)} for values in itertools.product(*param_grid.values())]
    
    logger.info(f"Starting hyperparameter tuning with {len(combinations_kwargs)} combinations and {n_jobs} parallel jobs...")
    
    # Use joblib to parallelize the evaluation
    results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_evaluate_params)(
            df, 
            {**fixed_kwargs, **params}, # Merge fixed kwargs with current params for train_prophet_model
            initial, period, horizon, metric
        ) for params in combinations_kwargs
    )
    
    best_metric_value = float('inf')
    best_params = {}

    for metric_value, params_evaluated in results:
        if metric_value < best_metric_value:
            best_metric_value = metric_value
            best_params = params_evaluated # params_evaluated now contains only tuned keys from _evaluate_params
            
    logger.info(f"Tuning complete. Best params: {best_params}, {metric}: {best_metric_value:.4f}")
    return best_params

def train_prophet_model(
    df: pd.DataFrame,
    freq: str = "D",
    seasonality_yearly: Any = "auto",
    seasonality_weekly: Any = "auto",
    seasonality_daily: Any = False,
    n_changepoints: int = 25,
    changepoint_range: float = 0.8,
    holidays_df: Optional[pd.DataFrame] = None,
    regressors: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    **prophet_kwargs,
) -> Prophet:
    """
    Trains a Prophet model on the provided time series data.

    Args:
        df: DataFrame with 'ds' (datetime) and 'y' (numeric) columns.
        freq: Frequency of the time series (e.g., 'D' for daily).
        seasonality_yearly: 'auto', True, False, or period for yearly seasonality.
        seasonality_weekly: 'auto', True, False, or period for weekly seasonality.
        seasonality_daily: 'auto', True, False, or period for daily seasonality.
        n_changepoints: Number of potential changepoints to include.
        changepoint_range: Proportion of the history in which to include potential changepoints.
        holidays_df: Optional DataFrame of holidays with 'ds', 'holiday', 'lower_window', 'upper_window' columns.
        regressors: Optional list of column names to add as regressors.
        save_path: Optional path to save the trained model.
        **prophet_kwargs: Additional keyword arguments to pass to Prophet constructor.

    Returns:
        A fitted Prophet model.
    """
    if df.empty:
        raise ValueError("Input DataFrame for training cannot be empty.")
    if 'ds' not in df.columns or 'y' not in df.columns:
        raise ValueError("Input DataFrame must contain 'ds' and 'y' columns.")

    model = Prophet(
        yearly_seasonality=seasonality_yearly,
        weekly_seasonality=seasonality_weekly,
        daily_seasonality=seasonality_daily,
        n_changepoints=n_changepoints,
        changepoint_range=changepoint_range,
        holidays=holidays_df,
        **prophet_kwargs,
    )
    
    if regressors:
        for regressor in regressors:
            model.add_regressor(regressor)
    
    logger.info(f"Training Prophet model with {len(df)} data points and frequency {freq}...")
    model.fit(df)
    
    if save_path:
        save_model(model, save_path)
        
    logger.info("Prophet model training complete.")
    return model



def forecast_with_prophet(
    model: Prophet,
    periods: int,
    freq: str = "D",
    future_regressors_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Generates a forecast using a fitted Prophet model.

    Args:
        model: A fitted Prophet model.
        periods: The number of periods to forecast forward.
        freq: The frequency of the forecast (e.g., 'D', 'W', 'M').
        future_regressors_df: DataFrame containing future values for regressors (must have 'ds' column).

    Returns:
        A pandas DataFrame containing the forecast, including 'ds', 'yhat', 'yhat_lower', 'yhat_upper'.
    """
    logger.info(f"Generating forecast for {periods} periods with frequency {freq}...")
    future = model.make_future_dataframe(periods=periods, freq=freq)
    
    if future_regressors_df is not None:
        if 'ds' not in future_regressors_df.columns:
            raise ValueError("future_regressors_df must contain a 'ds' column for merging.")
        
        # Merge future regressor values
        # Note: This assumes future_regressors_df has values for the dates in 'future'.
        # Prophet will raise an error if any regressor values are NaN in the future dataframe.
        future = pd.merge(future, future_regressors_df, on='ds', how='left')
        
        # Simple check/warning for NaNs in regressors
        # We find which columns are regressors in the model
        if hasattr(model, 'extra_regressors') and model.extra_regressors:
            for reg in model.extra_regressors:
                if reg in future.columns and future[reg].isnull().any():
                     logger.warning(f"Regressor '{reg}' contains NaN values in the future dataframe. Forecast may fail or degrade.")

    forecast = model.predict(future)
    logger.info("Forecast generation complete.")
    return forecast

def get_uk_bank_holidays(years: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Returns a DataFrame of UK bank holidays suitable for Prophet.
    Uses the 'holidays' library for dynamic generation.
    
    Args:
        years: List of years to include. If None, defaults to current year +/- 5 years.
    """
    if years is None:
        current_year = pd.Timestamp.now().year
        years = list(range(current_year - 5, current_year + 6))
    
    # Fetch holidays
    uk_holidays = holidays.UK(years=years)
    
    holiday_data = []
    for date, name in uk_holidays.items():
        holiday_data.append({
            'ds': pd.to_datetime(date),
            'holiday': name,
            'lower_window': 0,
            'upper_window': 1 
        })
        
    holidays_df = pd.DataFrame(holiday_data)
    
    if not holidays_df.empty:
        holidays_df = holidays_df.sort_values('ds').reset_index(drop=True)
        
    return holidays_df
