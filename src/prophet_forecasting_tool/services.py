"""Forecasting orchestration: load → outlier check → tune → train → forecast → plot.

This service is shared by the CLI and the Flask JSON API. Plot rendering uses
Matplotlib's OO API rather than pyplot, so concurrent requests don't fight over
the global figure manager.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import pathlib
import threading
from typing import Any, Dict, List, Optional

import pandas as pd
from matplotlib.figure import Figure
from sqlalchemy.engine import Engine

from prophet_forecasting_tool.config import Settings
from prophet_forecasting_tool.data_loader import get_max_date_for_table, load_time_series
from prophet_forecasting_tool.evaluate import evaluate_with_cross_validation
from prophet_forecasting_tool.model import (
    forecast_with_prophet,
    get_uk_bank_holidays,
    load_model,
    save_model,
    train_prophet_model,
    tune_hyperparameters,
)
from prophet_forecasting_tool.utils import (
    duration_to_periods,
    parse_duration,
    shift_timestamp,
    validate_identifier,
)

logger = logging.getLogger(__name__)

# Per-process lock registry keyed by model hash to avoid concurrent
# train/save races within a single worker.
_MODEL_LOCKS: Dict[str, threading.Lock] = {}
_REGISTRY_LOCK = threading.Lock()


def _lock_for(model_hash: str) -> threading.Lock:
    with _REGISTRY_LOCK:
        lock = _MODEL_LOCKS.get(model_hash)
        if lock is None:
            lock = threading.Lock()
            _MODEL_LOCKS[model_hash] = lock
        return lock


def _fig_to_base64(fig: Figure) -> str:
    img = io.BytesIO()
    fig.savefig(img, format="png", bbox_inches="tight")
    img.seek(0)
    return "data:image/png;base64," + base64.b64encode(img.getvalue()).decode("utf-8")


class ForecastingService:
    """Orchestrates the forecast lifecycle and isolates plotting from the web layer."""

    def __init__(
        self,
        engine: Engine,
        settings: Settings,
        models_dir: pathlib.Path,
        actual_calls_table: str = "actual_calls",
    ):
        self.engine = engine
        self.settings = settings
        self.models_dir = pathlib.Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.actual_calls_table = actual_calls_table

    def _data_fingerprint(self, df: pd.DataFrame) -> str:
        """Hash row count, min/max date, and y stats so corrections invalidate."""
        if df.empty:
            return "empty_data"
        payload = (
            f"{df['ds'].min().isoformat()}|"
            f"{df['ds'].max().isoformat()}|"
            f"{len(df)}|"
            f"{float(df['y'].sum()):.6f}|"
            f"{float(df['y'].mean()):.6f}"
        )
        return hashlib.md5(payload.encode("utf-8"), usedforsecurity=False).hexdigest()

    def _model_path(
        self,
        *,
        table_name: str,
        ts_column: str,
        y_column: str,
        series_name: str,
        freq: str,
        regressors: List[str],
        auto_tune: bool,
        resample_to_freq: Optional[str],
        training_window_duration: str,
        remove_outliers: bool,
        data_fingerprint: str,
    ) -> pathlib.Path:
        key = json.dumps(
            {
                "table": table_name,
                "ts": ts_column,
                "y": y_column,
                "series": series_name,
                "freq": freq,
                "regressors": sorted(regressors),
                "auto_tune": auto_tune,
                "resample": resample_to_freq or "",
                "window": training_window_duration,
                "outliers": remove_outliers,
                "fp": data_fingerprint,
            },
            sort_keys=True,
        )
        model_hash = hashlib.md5(key.encode("utf-8"), usedforsecurity=False).hexdigest()
        return self.models_dir / f"model_{model_hash}.json"

    @staticmethod
    def _filter_outliers(df: pd.DataFrame, holidays_df: Optional[pd.DataFrame]) -> tuple[pd.DataFrame, int]:
        """IQR outlier removal that preserves known holiday rows."""
        if df.empty:
            return df, 0
        protected = set()
        if holidays_df is not None and not holidays_df.empty:
            protected = set(pd.to_datetime(holidays_df["ds"]).dt.normalize().tolist())
        ds_normalized = df["ds"].dt.normalize() if pd.api.types.is_datetime64_any_dtype(df["ds"]) else df["ds"]
        is_protected = ds_normalized.isin(protected)

        Q1 = df["y"].quantile(0.25)
        Q3 = df["y"].quantile(0.75)
        iqr = Q3 - Q1
        lower, upper = Q1 - 1.5 * iqr, Q3 + 1.5 * iqr
        outlier_mask = ((df["y"] < lower) | (df["y"] > upper)) & ~is_protected
        kept = df.loc[~outlier_mask].copy()
        return kept, int(outlier_mask.sum())

    def get_forecast(
        self,
        table_name: str,
        ts_column: str,
        y_column: str,
        freq: str,
        horizon: str,
        series_name: str,
        regressors: List[str],
        auto_tune: bool = False,
        n_jobs: int = -1,
        resample_to_freq: Optional[str] = None,
        training_window_duration: str = "730 days",
        remove_outliers: bool = False,
        output_dir: Optional[pathlib.Path] = None,
    ) -> Dict[str, Any]:
        """Run the full forecast pipeline and return forecast_df + plot artifacts."""
        validate_identifier(table_name, "table")
        validate_identifier(ts_column, "column")
        validate_identifier(y_column, "column")
        for r in regressors:
            validate_identifier(r, "regressor column")

        # Validate durations early so the UI gets a friendly error.
        parse_duration(horizon)
        parse_duration(training_window_duration)

        latest_db_date = get_max_date_for_table(self.engine, table_name, ts_column)
        if latest_db_date is None:
            raise ValueError(
                f"No data found in table '{table_name}' or ts_column '{ts_column}' is empty."
            )

        # Pull enough history for the requested training window (plus a horizon
        # buffer so Prophet can place changepoints) AND any future regressor
        # values needed for the forecast window.
        data_load_start = shift_timestamp(latest_db_date, training_window_duration, subtract=True)
        data_load_start = shift_timestamp(data_load_start, horizon, subtract=True)
        future_regressor_end = shift_timestamp(latest_db_date, horizon)

        logger.info(
            f"Loading data: {data_load_start.date()} → {future_regressor_end.date()} "
            f"(table={table_name}, regressors={regressors})"
        )

        full_df = load_time_series(
            self.engine,
            table=table_name,
            ts_column=ts_column,
            y_column=y_column,
            regressors=regressors,
            resample_to_freq=resample_to_freq,
            start=data_load_start,
            end=future_regressor_end,
        )
        if full_df.empty:
            raise ValueError(
                f"No data returned from table '{table_name}' for the requested range."
            )

        # Training data: rows where y is known and within the training window.
        train_window_start = shift_timestamp(latest_db_date, training_window_duration, subtract=True)
        train_df = full_df.loc[
            (full_df["ds"] >= train_window_start) & (full_df["ds"] <= latest_db_date)
        ].copy()
        train_df = train_df.dropna(subset=["y"]).reset_index(drop=True)
        if train_df.empty:
            raise ValueError("Not enough data in the training window after filtering.")

        # Holidays for the full span the model will see (history + horizon).
        year_min = min(train_df["ds"].min().year, latest_db_date.year)
        year_max = max(future_regressor_end.year, latest_db_date.year)
        holidays_df = get_uk_bank_holidays(years=list(range(year_min, year_max + 1)))

        outliers_count = 0
        if remove_outliers:
            train_df, outliers_count = self._filter_outliers(train_df, holidays_df)
            logger.info(f"Removed {outliers_count} outliers (IQR, holidays preserved).")

        # Build the future regressor frame from the loaded data.
        # We require any regressor to have a non-null value within the forecast
        # window. If the caller didn't supply it, we forward-fill from the last
        # known value as a documented heuristic.
        future_regressors_df: Optional[pd.DataFrame] = None
        if regressors:
            keep_cols = ["ds"] + regressors
            future_regressors_df = full_df[keep_cols].copy()
            future_regressors_df = future_regressors_df.sort_values("ds").reset_index(drop=True)
            for r in regressors:
                if future_regressors_df[r].isna().any():
                    logger.info(
                        f"Forward-filling missing future values for regressor '{r}'."
                    )
                    future_regressors_df[r] = future_regressors_df[r].ffill().bfill()

        # Cache key/path includes everything that affects the trained weights.
        data_fingerprint = self._data_fingerprint(train_df)
        model_path = self._model_path(
            table_name=table_name,
            ts_column=ts_column,
            y_column=y_column,
            series_name=series_name,
            freq=freq,
            regressors=regressors,
            auto_tune=auto_tune,
            resample_to_freq=resample_to_freq,
            training_window_duration=training_window_duration,
            remove_outliers=remove_outliers,
            data_fingerprint=data_fingerprint,
        )
        model_hash = model_path.stem

        model = None
        lock = _lock_for(model_hash)
        with lock:
            if model_path.exists():
                try:
                    model = load_model(str(model_path))
                    logger.info(f"Loaded cached model {model_hash}")
                except Exception as e:
                    logger.warning(f"Cached model failed to load ({e}); retraining.")
                    model = None

            if model is None:
                prophet_kwargs: Dict[str, Any] = {
                    "n_changepoints": 25,
                    "changepoint_range": 0.8,
                }
                if auto_tune:
                    logger.info("Auto-tuning hyperparameters; this may take a while.")
                    best_params = tune_hyperparameters(
                        train_df,
                        freq=freq,
                        holidays_df=holidays_df,
                        regressors=regressors,
                        n_jobs=n_jobs,
                        **prophet_kwargs,
                    )
                    prophet_kwargs.update(best_params)
                    logger.info(f"Best params: {best_params}")

                model = train_prophet_model(
                    train_df,
                    freq=freq,
                    holidays_df=holidays_df,
                    regressors=regressors,
                    **prophet_kwargs,
                )
                try:
                    tmp_path = model_path.with_suffix(".json.tmp")
                    save_model(model, str(tmp_path))
                    tmp_path.replace(model_path)
                except Exception as e:
                    logger.error(f"Failed to save model: {e}")

        # 5. Forecast
        periods = duration_to_periods(horizon, freq)
        forecast_df = forecast_with_prophet(
            model,
            periods=periods,
            freq=freq,
            future_regressors_df=future_regressors_df,
        )

        # 6. Plots (OO matplotlib — no pyplot globals)
        plot_results = self._render_plots(
            model=model,
            forecast_df=forecast_df,
            train_df=train_df,
            ts_column=ts_column,
            y_column=y_column,
            output_dir=output_dir,
            series_name=series_name,
        )

        return {
            "forecast_df": forecast_df,
            "series_name": series_name,
            "outliers_removed": outliers_count,
            "training_rows": len(train_df),
            "training_start": train_df["ds"].min().isoformat(),
            "training_end": train_df["ds"].max().isoformat(),
            **plot_results,
        }

    def _render_plots(
        self,
        *,
        model,
        forecast_df: pd.DataFrame,
        train_df: pd.DataFrame,
        ts_column: str,
        y_column: str,
        output_dir: Optional[pathlib.Path],
        series_name: str,
    ) -> Dict[str, Any]:
        plot_results: Dict[str, Any] = {}

        fig = model.plot(forecast_df)
        ax = fig.gca()
        ax.plot(train_df["ds"], train_df["y"], "k-", linewidth=1, label="Historical")

        actual_df_full: Optional[pd.DataFrame] = None
        try:
            actual_df_full = load_time_series(
                self.engine,
                table=self.actual_calls_table,
                ts_column=ts_column,
                y_column=y_column,
                columns_to_load=["ds", "day", "y", "answered_calls", "abandoned_calls"],
            )
            if actual_df_full is not None and not actual_df_full.empty:
                ax.plot(actual_df_full["ds"], actual_df_full["y"], "g-", linewidth=1.5, label="Actual")
                ax.legend()
        except Exception as e:
            logger.debug(f"Skipped actuals overlay ({self.actual_calls_table}): {e}")

        fig_components = model.plot_components(forecast_df)

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            forecast_path = output_dir / f"{series_name}_forecast_plot.png"
            fig.savefig(forecast_path, bbox_inches="tight")
            plot_results["forecast_plot_path"] = str(forecast_path)

            components_path = output_dir / f"{series_name}_forecast_components_plot.png"
            fig_components.savefig(components_path, bbox_inches="tight")
            plot_results["components_plot_path"] = str(components_path)
        else:
            plot_results["forecast_plot"] = _fig_to_base64(fig)
            plot_results["components_plot"] = _fig_to_base64(fig_components)

        # Day-of-week breakdown from actuals.
        if actual_df_full is not None and not actual_df_full.empty and "day" in actual_df_full.columns:
            try:
                day_map = {
                    "mon": "Monday", "tue": "Tuesday", "wed": "Wednesday",
                    "thu": "Thursday", "fri": "Friday", "sat": "Saturday", "sun": "Sunday",
                }
                actual_df_full["day_full"] = actual_df_full["day"].astype(str).str.lower().map(day_map)
                day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

                if "answered_calls" in actual_df_full.columns and "abandoned_calls" in actual_df_full.columns:
                    day_stats = (
                        actual_df_full.groupby("day_full")
                        .agg({"answered_calls": "sum", "abandoned_calls": "sum"})
                        .reindex(day_order)
                        .fillna(0)
                    )
                    fig_bar = Figure(figsize=(10, 6))
                    ax_bar = fig_bar.subplots()
                    x = range(len(day_stats))
                    width = 0.6
                    ax_bar.bar(x, day_stats["answered_calls"], width, label="Answered", color="green")
                    ax_bar.bar(
                        x,
                        day_stats["abandoned_calls"],
                        width,
                        bottom=day_stats["answered_calls"],
                        label="Abandoned",
                        color="red",
                    )
                    ax_bar.set_xlabel("Day of Week")
                    ax_bar.set_ylabel("Number of Calls")
                    ax_bar.set_title("Actual Calls by Day of Week (Answered vs Abandoned)")
                    ax_bar.set_xticks(list(x))
                    ax_bar.set_xticklabels(day_stats.index, rotation=45, ha="right")
                    ax_bar.legend()
                    fig_bar.tight_layout()
                    if output_dir:
                        path = output_dir / f"{series_name}_day_breakdown_plot.png"
                        fig_bar.savefig(path, bbox_inches="tight")
                        plot_results["day_breakdown_plot_path"] = str(path)
                    else:
                        plot_results["day_breakdown_plot"] = _fig_to_base64(fig_bar)
            except Exception as e:
                logger.warning(f"Could not create day breakdown plot: {e}")

        return plot_results

    def evaluate_model(
        self,
        table_name: str,
        ts_column: str,
        y_column: str,
        freq: str,
        metrics: List[str],
        initial: str,
        period: str,
        horizon: str,
        regressors: List[str],
        n_jobs: int = -1,
        resample_to_freq: Optional[str] = None,
        training_window_duration: Optional[str] = None,
        output_dir: Optional[pathlib.Path] = None,
        **prophet_kwargs,
    ) -> Dict[str, Any]:
        """Rolling-window CV evaluation mirroring the production training spec."""
        validate_identifier(table_name, "table")
        validate_identifier(ts_column, "column")
        validate_identifier(y_column, "column")
        for r in regressors:
            validate_identifier(r, "regressor column")
        parse_duration(initial)
        parse_duration(period)
        parse_duration(horizon)

        latest_db_date = get_max_date_for_table(self.engine, table_name, ts_column)
        if latest_db_date is None:
            raise ValueError(
                f"No data found in table '{table_name}' or ts_column '{ts_column}' is empty."
            )

        # We need at least initial + 3*period + horizon of history for ≥3 folds.
        start = shift_timestamp(latest_db_date, initial, subtract=True)
        start = shift_timestamp(start, period, subtract=True)
        start = shift_timestamp(start, period, subtract=True)
        start = shift_timestamp(start, period, subtract=True)
        start = shift_timestamp(start, horizon, subtract=True)

        logger.info(f"Loading evaluation data: {start.date()} → {latest_db_date.date()}")
        df = load_time_series(
            self.engine,
            table=table_name,
            ts_column=ts_column,
            y_column=y_column,
            regressors=regressors,
            resample_to_freq=resample_to_freq,
            start=start,
            end=latest_db_date,
        )
        if df.empty:
            raise ValueError(
                f"No data returned from table '{table_name}' for evaluation."
            )

        results = evaluate_with_cross_validation(
            df,
            freq=freq,
            metrics=metrics,
            initial=initial,
            period=period,
            horizon=horizon,
            n_jobs=n_jobs,
            regressors=regressors,
            support_uk_holidays=True,
            **prophet_kwargs,
        )

        if output_dir is not None:
            output_dir = pathlib.Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            metrics_output_path = output_dir / f"{table_name}_{ts_column}_{y_column}_cv_metrics.json"
            with open(metrics_output_path, "w") as f:
                json.dump(results, f, indent=4)
            logger.info(f"CV metrics saved to {metrics_output_path}")

        return results
