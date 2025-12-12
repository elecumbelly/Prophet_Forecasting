import argparse
import logging
import os
import pathlib
import pandas as pd
from sqlalchemy import create_engine

from prophet_forecasting_tool.config import Settings
from prophet_forecasting_tool.logging_config import setup_logging
from prophet_forecasting_tool.data_loader import load_time_series
from prophet_forecasting_tool.model import get_uk_bank_holidays
from prophet_forecasting_tool.services import ForecastingService

logger = logging.getLogger(__name__)

def common_arguments(parser):
    """Adds common arguments to subparsers."""
    parser.add_argument(
        "--series-name",
        type=str,
        default="default_series",
        help="Logical label for the time series being processed.",
    )
    parser.add_argument(
        "--table",
        type=str,
        default="real_call_metrics",
        help="Name of the PostgreSQL table to load data from.",
    )
    parser.add_argument(
        "--ts-column",
        type=str,
        default="ds",
        help="Name of the timestamp column in the table.",
    )
    parser.add_argument(
        "--y-column",
        type=str,
        default="y",
        help="Name of the target value column in the table.",
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="D",
        choices=["D", "W", "M"],
        help="Frequency of the time series (D=daily, W=weekly, M=monthly).",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default="./outputs",
        help="Directory to save output files (forecasts, plots, metrics).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level.",
    )
    parser.add_argument(
        "--regressors",
        type=str,
        default="",
        help="Comma-separated list of regressor column names.",
    )
    parser.add_argument(
        "--auto-tune",
        action="store_true",
        help="Enable automatic hyperparameter tuning.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs for hyperparameter tuning. -1 means use all CPUs.",
    )
    parser.add_argument(
        "--resample-to-freq",
        type=str,
        default=None,
        help="Optional frequency to resample the input data to (e.g., 'H', 'D', 'W').",
    )
    parser.add_argument(
        "--training-window-duration",
        type=str,
        default="730 days",
        help="Duration of the training window (e.g., '365 days', '2Y').",
    )


def handle_train_and_forecast(args, service: ForecastingService):
    """Handles the 'train-and-forecast' subcommand."""
    # setup_logging(args.log_level) # Handled by main() before service init
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting training and forecasting for series: {args.series_name}")
    logger.info(f"Output will be saved to: {args.output_dir}")

    try:
        regressors = [r.strip() for r in args.regressors.split(',') if r.strip()]

        results = service.get_forecast(
            table_name=args.table,
            ts_column=args.ts_column,
            y_column=args.y_column,
            freq=args.freq,
            horizon=args.horizon,
            series_name=args.series_name,
            regressors=regressors,
            auto_tune=args.auto_tune,
            n_jobs=args.n_jobs, # Pass n_jobs from CLI args
            resample_to_freq=args.resample_to_freq, # New parameter
            training_window_duration=args.training_window_duration, # New parameter
            output_dir=args.output_dir # Pass output_dir for CLI to save plots
        )
        
        # Service saves plots, just need to save forecast_df
        forecast_output_path = args.output_dir / f"{args.series_name}_forecast.csv"
        results["forecast_df"].to_csv(forecast_output_path, index=False)
        logger.info(f"Forecast saved to {forecast_output_path}")
        logger.info(f"Forecast plot saved to {results.get('forecast_plot_path')}")
        logger.info(f"Forecast components plot saved to {results.get('components_plot_path')}")

    except Exception as e:
        logger.exception(f"An error occurred during training and forecasting: {e}")

def handle_evaluate(args, service: ForecastingService):
    """Handles the 'evaluate' subcommand."""
    # setup_logging(args.log_level) # Handled by main() before service init
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting evaluation for series: {args.series_name}")
    logger.info(f"Output will be saved to: {args.output_dir}")

    try:
        metrics_list = args.metrics.split(',')
        regressors = [r.strip() for r in args.regressors.split(',') if r.strip()]

        calculated_metrics = service.evaluate_model(
            table_name=args.table,
            ts_column=args.ts_column,
            y_column=args.y_column,
            freq=args.freq,
            metrics=metrics_list,
            initial=args.initial,
            period=args.period,
            horizon=args.horizon,
            regressors=regressors,
            n_jobs=args.n_jobs, # Pass n_jobs from CLI args
            resample_to_freq=args.resample_to_freq, # New parameter
            training_window_duration=args.training_window_duration, # New parameter
            output_dir=args.output_dir,
            support_uk_holidays=True # Still passed to model, if needed
        )

        logger.info("Cross-Validation Metrics:")
        for metric, value in calculated_metrics.items():
            logger.info(f"  {metric.upper()}: {value:.4f}")

    except Exception as e:
        logger.exception(f"An error occurred during evaluation: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Time series forecasting utility using Prophet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Global arguments (e.g., --log-level could be here or per-subcommand)
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level.",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train and Forecast subcommand
    train_parser = subparsers.add_parser(
        "train-and-forecast",
        help="Train a Prophet model and generate a forecast.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    common_arguments(train_parser)
    train_parser.add_argument(
        "--horizon",
        type=str,
        default="365D",
        help="Forecast horizon (e.g., '90D', '180D', '365D').",
    )
    train_parser.set_defaults(func=lambda args: handle_train_and_forecast(args, forecasting_service))

    # Evaluate subcommand
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Backtest on historical data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    common_arguments(eval_parser)
    eval_parser.add_argument(
        "--metrics",
        type=str,
        default="mae,mape,rmse",
        help="Comma-separated list of metrics to calculate (mae, mape, rmse).",
    )
    eval_parser.add_argument(
        "--initial",
        type=str,
        default="730 days",
        help="Initial training period for Cross-Validation (e.g., '730 days').",
    )
    eval_parser.add_argument(
        "--period",
        type=str,
        default="180 days",
        help="Period between cutoffs for Cross-Validation (e.g., '180 days').",
    )
    eval_parser.add_argument(
        "--horizon",
        type=str,
        default="365 days",
        help="Forecast horizon for Cross-Validation (e.g., '365 days').",
    )
    eval_parser.set_defaults(func=lambda args: handle_evaluate(args, forecasting_service))

    args = parser.parse_args()

    # Re-setup logging with the potentially overridden log-level from subcommand, if global was used
    # or if the subcommand specifies its own. For simplicity, we pass args.log_level to setup_logging.
    setup_logging(args.log_level) 
    
    # Initialize settings, engine and service after args are parsed
    _settings = Settings()
    _engine = create_engine(_settings.DATABASE_URL)
    _models_dir = args.output_dir / "models"
    _models_dir.mkdir(parents=True, exist_ok=True)
    forecasting_service = ForecastingService(_engine, _settings, _models_dir)


    if args.command:
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
