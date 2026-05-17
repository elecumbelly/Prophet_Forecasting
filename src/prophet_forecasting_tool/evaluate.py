"""Cross-validation evaluation for Prophet models.

The evaluation path mirrors the production training path so reported metrics
describe the model that will actually be deployed. MAPE is implemented with a
small epsilon to avoid divide-by-zero on quiet weekends/holidays.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error

from prophet_forecasting_tool.model import train_prophet_model, get_uk_bank_holidays

logger = logging.getLogger(__name__)


def _parallel_mode(n_jobs: int) -> Optional[str]:
    """Map an integer ``n_jobs`` to a Prophet ``parallel`` string.

    Prophet's ``cross_validation`` expects ``None`` or one of
    ``"threads"|"processes"|"dask"`` — never an int. We map any ``n_jobs != 1``
    to processes, and ``n_jobs == 1`` to single-threaded.
    """
    return None if n_jobs == 1 else "processes"


def calculate_mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def calculate_mape(y_true: pd.Series, y_pred: pd.Series, epsilon: float = 1.0) -> float:
    """Mean Absolute Percentage Error with a denominator floor.

    Standard MAPE explodes near zero; we use ``max(|y_true|, epsilon)`` to
    keep the metric finite on quiet periods. Reported as a percentage.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true_arr), epsilon)
    return float(np.mean(np.abs((y_true_arr - y_pred_arr) / denom)) * 100.0)


def calculate_rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


METRIC_FUNCTIONS = {
    "mae": calculate_mae,
    "mape": calculate_mape,
    "rmse": calculate_rmse,
}


def evaluate_with_cross_validation(
    df: pd.DataFrame,
    freq: str = "D",
    metrics: Optional[List[str]] = None,
    initial: str = "730 days",
    period: str = "180 days",
    horizon: str = "365 days",
    n_jobs: int = -1,
    regressors: Optional[List[str]] = None,
    holidays_df: Optional[pd.DataFrame] = None,
    support_uk_holidays: bool = True,
    **prophet_kwargs,
) -> Dict[str, float]:
    """Rolling-window cross-validation with metrics aggregated across folds.

    Mirrors the production model spec (holidays + regressors) so metrics
    describe the deployed model rather than a stripped-down variant.
    """
    if df.empty:
        raise ValueError("Input DataFrame for evaluation cannot be empty.")

    # The previous version of this function accepted ``support_uk_holidays`` via
    # ``**prophet_kwargs``; some call sites still pass it that way.
    prophet_kwargs.pop("support_uk_holidays", None)
    prophet_kwargs.pop("save_path", None)

    if holidays_df is None and support_uk_holidays:
        years = sorted({df["ds"].min().year, df["ds"].max().year})
        full_years = list(range(min(years), max(years) + 2))
        holidays_df = get_uk_bank_holidays(years=full_years)

    logger.info(f"Training model on full dataset ({len(df)} rows) for cross-validation...")
    model = train_prophet_model(
        df,
        freq=freq,
        holidays_df=holidays_df,
        regressors=regressors,
        **prophet_kwargs,
    )

    parallel_arg = _parallel_mode(n_jobs)
    logger.info(
        f"Starting cross-validation (initial={initial}, period={period}, "
        f"horizon={horizon}, parallel={parallel_arg})..."
    )
    try:
        df_cv = cross_validation(
            model,
            initial=initial,
            period=period,
            horizon=horizon,
            disable_tqdm=True,
            parallel=parallel_arg,
        )
    except Exception as e:
        logger.error(f"Cross-validation failed: {e}")
        raise ValueError(
            f"Cross-validation failed. Check that the data covers at least "
            f"`initial` + `horizon` ({initial} + {horizon}). Error: {e}"
        ) from e

    metrics_to_calculate = metrics if metrics else ["mae", "mape", "rmse"]

    results: Dict[str, float] = {}
    # Prefer Prophet's performance_metrics for the metrics it supports natively.
    prophet_metrics = [m for m in metrics_to_calculate if m in ("mae", "rmse")]
    if prophet_metrics:
        df_p = performance_metrics(df_cv, metrics=prophet_metrics, rolling_window=1)
        for m in prophet_metrics:
            if m in df_p.columns:
                results[m] = float(df_p[m].mean())

    if "mape" in metrics_to_calculate:
        # Use our safe MAPE rather than Prophet's, which divides by raw y.
        results["mape"] = calculate_mape(df_cv["y"], df_cv["yhat"])

    for m, v in results.items():
        logger.info(f"CV {m.upper()}: {v:.4f}")

    return results
