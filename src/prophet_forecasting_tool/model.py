"""Thin wrapper around Prophet for training, forecasting, tuning and persistence."""
from __future__ import annotations

import itertools
import json
import logging
from typing import Any, Dict, List, Optional

import holidays
import joblib
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.serialize import model_from_json, model_to_json

logger = logging.getLogger(__name__)


def save_model(model: Prophet, filepath: str) -> None:
    logger.info(f"Saving model to {filepath}...")
    with open(filepath, "w") as f:
        json.dump(model_to_json(model), f)
    logger.info("Model saved successfully.")


def load_model(filepath: str) -> Prophet:
    logger.info(f"Loading model from {filepath}...")
    with open(filepath, "r") as f:
        model_json = json.load(f)
    return model_from_json(model_json)


def _evaluate_params(df, current_kwargs, initial, period, horizon, metric, tuned_keys):
    """Evaluate one hyperparameter combination on a CV split."""
    df_copy = df.copy()
    temp_kwargs = current_kwargs.copy()
    temp_kwargs.pop("save_path", None)

    try:
        m = train_prophet_model(df_copy, **temp_kwargs)
        df_cv = cross_validation(
            m,
            initial=initial,
            period=period,
            horizon=horizon,
            disable_tqdm=True,
        )
        df_p = performance_metrics(df_cv, rolling_window=1)
        metric_value = float(df_p[metric].mean())
        return metric_value, {k: temp_kwargs[k] for k in tuned_keys if k in temp_kwargs}
    except Exception as e:
        logger.warning(f"Tuning failed for params {current_kwargs}: {e}")
        return float("inf"), {k: temp_kwargs.get(k) for k in tuned_keys}


def tune_hyperparameters(
    df: pd.DataFrame,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    metric: str = "rmse",
    initial: str = "730 days",
    period: str = "180 days",
    horizon: str = "365 days",
    n_jobs: int = -1,
    **fixed_kwargs,
) -> Dict[str, Any]:
    """Grid-search hyperparameter tuning using rolling-window CV."""
    if param_grid is None:
        param_grid = {
            "changepoint_prior_scale": [0.001, 0.01, 0.1, 0.5],
            "seasonality_prior_scale": [0.01, 0.1, 1.0, 10.0],
        }

    tuned_keys = tuple(param_grid.keys())
    keys = param_grid.keys()
    combinations_kwargs = [
        {k: v for k, v in zip(keys, values)} for values in itertools.product(*param_grid.values())
    ]

    logger.info(
        f"Starting hyperparameter tuning: {len(combinations_kwargs)} combinations, "
        f"n_jobs={n_jobs}, tuned keys={tuned_keys}"
    )

    results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_evaluate_params)(
            df,
            {**fixed_kwargs, **params},
            initial,
            period,
            horizon,
            metric,
            tuned_keys,
        )
        for params in combinations_kwargs
    )

    best_metric_value = float("inf")
    best_params: Dict[str, Any] = {}
    for metric_value, params_evaluated in results:
        if metric_value < best_metric_value:
            best_metric_value = metric_value
            best_params = params_evaluated

    logger.info(f"Tuning complete. Best params: {best_params}, {metric}: {best_metric_value:.4f}")
    return best_params


def train_prophet_model(
    df: pd.DataFrame,
    freq: str = "D",
    seasonality_yearly: Any = "auto",
    seasonality_weekly: Any = "auto",
    seasonality_daily: Any = False,
    seasonality_mode: str = "additive",
    n_changepoints: int = 25,
    changepoint_range: float = 0.8,
    holidays_df: Optional[pd.DataFrame] = None,
    regressors: Optional[List[str]] = None,
    interval_width: float = 0.8,
    save_path: Optional[str] = None,
    **prophet_kwargs,
) -> Prophet:
    """Fit a Prophet model with sensible defaults and optional regressors/holidays."""
    if df.empty:
        raise ValueError("Input DataFrame for training cannot be empty.")
    if "ds" not in df.columns or "y" not in df.columns:
        raise ValueError("Input DataFrame must contain 'ds' and 'y' columns.")

    # Prophet rejects tz-aware ds; normalize defensively.
    if pd.api.types.is_datetime64tz_dtype(df["ds"]):
        df = df.copy()
        df["ds"] = df["ds"].dt.tz_localize(None)

    model = Prophet(
        yearly_seasonality=seasonality_yearly,
        weekly_seasonality=seasonality_weekly,
        daily_seasonality=seasonality_daily,
        seasonality_mode=seasonality_mode,
        n_changepoints=n_changepoints,
        changepoint_range=changepoint_range,
        holidays=holidays_df,
        interval_width=interval_width,
        **prophet_kwargs,
    )

    if regressors:
        for regressor in regressors:
            if regressor not in df.columns:
                raise ValueError(
                    f"Regressor column '{regressor}' is not present in the training data."
                )
            model.add_regressor(regressor)

    logger.info(f"Training Prophet model: {len(df)} rows, freq={freq}, mode={seasonality_mode}")
    model.fit(df)

    if save_path:
        save_model(model, save_path)

    return model


def forecast_with_prophet(
    model: Prophet,
    periods: int,
    freq: str = "D",
    future_regressors_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Generate a forecast. Raises if regressor future values are missing."""
    logger.info(f"Generating forecast: periods={periods}, freq={freq}")
    future = model.make_future_dataframe(periods=periods, freq=freq)

    extra_regressors = getattr(model, "extra_regressors", {}) or {}
    if extra_regressors and future_regressors_df is None:
        raise ValueError(
            "Model has regressors configured but no future_regressors_df was provided."
        )

    if future_regressors_df is not None and extra_regressors:
        if "ds" not in future_regressors_df.columns:
            raise ValueError("future_regressors_df must contain a 'ds' column.")
        future = pd.merge(future, future_regressors_df, on="ds", how="left")
        missing = [r for r in extra_regressors if r not in future.columns]
        if missing:
            raise ValueError(
                f"Regressor columns missing from future_regressors_df: {missing}"
            )
        nan_regressors = [r for r in extra_regressors if future[r].isnull().any()]
        if nan_regressors:
            raise ValueError(
                f"Regressor values are missing in the future window for: {nan_regressors}. "
                "Provide future values for every regressor (e.g. seasonal naive fill)."
            )

    return model.predict(future)


def get_uk_bank_holidays(years: Optional[List[int]] = None) -> pd.DataFrame:
    """Return UK bank holidays for Prophet, with per-holiday windows.

    Windows expand the holiday influence to nearby days:
    - Christmas Day: ``lower=-1`` (Christmas Eve dip) to ``upper=2`` (Boxing Day +1)
    - New Year's Day: ``lower=-1`` to ``upper=1``
    - Generic bank holidays: ``upper=1`` for the return-to-work spike
    """
    if years is None:
        current_year = pd.Timestamp.now().year
        years = list(range(current_year - 5, current_year + 6))

    uk_holidays = holidays.UK(years=years)

    rows = []
    for date, name in uk_holidays.items():
        lower, upper = 0, 1
        lname = name.lower()
        if "christmas" in lname:
            lower, upper = -1, 2
        elif "new year" in lname:
            lower, upper = -1, 1
        rows.append({
            "ds": pd.to_datetime(date),
            "holiday": name,
            "lower_window": lower,
            "upper_window": upper,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("ds").reset_index(drop=True)
    return df
