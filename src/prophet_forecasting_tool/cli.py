"""Command-line entry point: ``prophet_forecaster {train-and-forecast|evaluate}``."""
from __future__ import annotations

import argparse
import logging
import pathlib

from sqlalchemy import create_engine

from prophet_forecasting_tool.config import Settings
from prophet_forecasting_tool.logging_config import setup_logging
from prophet_forecasting_tool.services import ForecastingService

logger = logging.getLogger(__name__)


def common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--series-name", default="default_series",
                        help="Logical label for the time series.")
    parser.add_argument("--table", default="real_call_metrics",
                        help="PostgreSQL table to load data from.")
    parser.add_argument("--ts-column", default="ds", help="Timestamp column name.")
    parser.add_argument("--y-column", default="y", help="Target column name.")
    parser.add_argument("--freq", default="D", choices=["D", "W", "M", "H"],
                        help="Frequency of the time series.")
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("./outputs"),
                        help="Directory to save forecasts/plots/metrics.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--regressors", default="",
                        help="Comma-separated list of regressor column names.")
    parser.add_argument("--auto-tune", action="store_true",
                        help="Run hyperparameter tuning before training.")
    parser.add_argument("--remove-outliers", action="store_true",
                        help="Remove IQR outliers from the training data "
                             "(holiday rows are preserved).")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Parallel workers for tuning/CV. -1 = all CPUs.")
    parser.add_argument("--resample-to-freq", default=None,
                        help="Resample input data to this frequency before training.")
    parser.add_argument("--training-window-duration", default="730 days",
                        help="Duration of the training window, e.g. '730 days' or '2Y'.")


def handle_train_and_forecast(args: argparse.Namespace, service: ForecastingService) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Training and forecasting series: {args.series_name}")
    regressors = [r.strip() for r in args.regressors.split(",") if r.strip()]
    results = service.get_forecast(
        table_name=args.table,
        ts_column=args.ts_column,
        y_column=args.y_column,
        freq=args.freq,
        horizon=args.horizon,
        series_name=args.series_name,
        regressors=regressors,
        auto_tune=args.auto_tune,
        n_jobs=args.n_jobs,
        resample_to_freq=args.resample_to_freq,
        training_window_duration=args.training_window_duration,
        remove_outliers=args.remove_outliers,
        output_dir=args.output_dir,
    )
    forecast_output_path = args.output_dir / f"{args.series_name}_forecast.csv"
    results["forecast_df"].to_csv(forecast_output_path, index=False)
    logger.info(f"Forecast saved to {forecast_output_path}")
    if results.get("forecast_plot_path"):
        logger.info(f"Forecast plot: {results['forecast_plot_path']}")
    if results.get("components_plot_path"):
        logger.info(f"Components plot: {results['components_plot_path']}")


def handle_evaluate(args: argparse.Namespace, service: ForecastingService) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Evaluating series: {args.series_name}")
    metrics_list = [m.strip() for m in args.metrics.split(",") if m.strip()]
    regressors = [r.strip() for r in args.regressors.split(",") if r.strip()]
    metrics = service.evaluate_model(
        table_name=args.table,
        ts_column=args.ts_column,
        y_column=args.y_column,
        freq=args.freq,
        metrics=metrics_list,
        initial=args.initial,
        period=args.period,
        horizon=args.horizon,
        regressors=regressors,
        n_jobs=args.n_jobs,
        resample_to_freq=args.resample_to_freq,
        training_window_duration=args.training_window_duration,
        output_dir=args.output_dir,
    )
    logger.info("Cross-validation metrics:")
    for k, v in metrics.items():
        logger.info(f"  {k.upper()}: {v:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Time series forecasting utility using Prophet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    train_parser = subparsers.add_parser(
        "train-and-forecast", help="Train a Prophet model and generate a forecast.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    common_arguments(train_parser)
    train_parser.add_argument("--horizon", default="365D",
                              help="Forecast horizon, e.g. '90D', '12M'.")
    train_parser.set_defaults(func=handle_train_and_forecast)

    eval_parser = subparsers.add_parser(
        "evaluate", help="Backtest on historical data with rolling-window CV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    common_arguments(eval_parser)
    eval_parser.add_argument("--metrics", default="mae,mape,rmse",
                             help="Comma-separated list of metrics (mae, mape, rmse).")
    eval_parser.add_argument("--initial", default="730 days",
                             help="Initial training period for CV.")
    eval_parser.add_argument("--period", default="180 days",
                             help="Period between CV cutoffs.")
    eval_parser.add_argument("--horizon", default="365 days",
                             help="Forecast horizon for CV.")
    eval_parser.set_defaults(func=handle_evaluate)

    args = parser.parse_args()
    setup_logging(args.log_level)

    settings = Settings()
    engine = create_engine(settings.DATABASE_URL)
    models_dir = args.output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    service = ForecastingService(engine, settings, models_dir)

    args.func(args, service)


if __name__ == "__main__":
    main()
