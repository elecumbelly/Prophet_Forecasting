import pandas as pd
import logging
import pathlib
import hashlib
from typing import Dict, List, Any, Optional

from sqlalchemy.engine import Engine

from prophet_forecasting_tool.config import Settings
from prophet_forecasting_tool.data_loader import load_time_series, get_max_date_for_table
from prophet_forecasting_tool.model import (
    train_prophet_model, 
    forecast_with_prophet, 
    get_uk_bank_holidays, 
    save_model, 
    load_model, 
    tune_hyperparameters
)
from prophet_forecasting_tool.evaluate import evaluate_with_cross_validation

logger = logging.getLogger(__name__)

class ForecastingService:
    def __init__(self, engine: Engine, settings: Settings, models_dir: pathlib.Path):
        self.engine = engine
        self.settings = settings
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True) # Ensure models directory exists

    def _generate_data_fingerprint(self, df: pd.DataFrame) -> str:
        """Generates a hash based on key data characteristics to detect changes."""
        if df.empty:
            return "empty_data"
        max_date_str = str(df['ds'].max())
        row_count = len(df)
        return hashlib.md5(f"{max_date_str}_{row_count}".encode('utf-8')).hexdigest()

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
        resample_to_freq: Optional[str] = None, # New parameter
        training_window_duration: str = "730 days", # New parameter
        output_dir: Optional[pathlib.Path] = None,
    ) -> Dict[str, Any]:
        """
        Orchestrates data loading, model training/loading, forecasting, and plotting.
        Returns a dictionary of results for rendering or saving.
        """
        try:
            # 1. Determine Required Data Range for Loading
            latest_db_date = get_max_date_for_table(self.engine, table_name, ts_column)
            if latest_db_date is None:
                raise ValueError(f"No data found in table '{table_name}' or ts_column '{ts_column}' is empty.")

            # Calculate the effective start and end dates for loading data
            td_training_window = pd.to_timedelta(training_window_duration)
            td_horizon = pd.to_timedelta(horizon)
            
            # The data must start at least 'training_window_duration' before latest_db_date
            # Add a buffer (e.g., another horizon) to ensure enough historical data for any pre-processing,
            # and to allow Prophet to identify changepoints properly.
            data_load_start = latest_db_date - td_training_window - td_horizon # Buffer for model needs
            data_load_end = latest_db_date + td_horizon

            logger.info(f"Loading data from {data_load_start} to {latest_db_date}...")

            df = load_time_series(
                self.engine,
                table=table_name,
                ts_column=ts_column,
                y_column=y_column,
                regressors=regressors,
                resample_to_freq=resample_to_freq,
                start=data_load_start, # Pass calculated start
                end=latest_db_date,    # Only load up to latest actual data
            )
            if df.empty:
                raise ValueError(f"No data found in table '{table_name}' for forecasting within the calculated range.")

            # 2. Prepare Training Data - use ALL available data for training
            train_df = df.copy()

            if train_df.empty:
                raise ValueError("Not enough data for training.")

            # Remove outliers using IQR method
            Q1 = train_df['y'].quantile(0.25)
            Q3 = train_df['y'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers_count = len(train_df[(train_df['y'] < lower_bound) | (train_df['y'] > upper_bound)])
            train_df = train_df[(train_df['y'] >= lower_bound) & (train_df['y'] <= upper_bound)]
            logger.info(f"Removed {outliers_count} outliers (IQR method). Bounds: [{lower_bound:.0f}, {upper_bound:.0f}]")

            logger.info(f"Training data range: {train_df['ds'].min()} to {train_df['ds'].max()} ({len(train_df)} rows)")

            # 3. Generate Model ID & Handle Caching
            data_fingerprint = self._generate_data_fingerprint(train_df)
            regressors_str = ",".join(sorted(regressors))
            tune_str = "_tuned" if auto_tune else ""
            model_id_str = f"{table_name}_{ts_column}_{y_column}_{series_name}_{regressors_str}_{data_fingerprint}{tune_str}_{training_window_duration}_{resample_to_freq}"
            model_hash = hashlib.md5(model_id_str.encode('utf-8')).hexdigest()
            model_filename = f"model_{model_hash}.json"
            model_path = self.models_dir / model_filename

            model = None
            if model_path.exists():
                logger.info(f"Attempting to load existing model from {model_path}...")
                try:
                    model = load_model(str(model_path))
                    logger.info(f"Loaded cached model for {series_name} (Tuned: {auto_tune}).")
                except Exception as e:
                    logger.warning(f"Failed to load cached model: {e}. Retraining...")
                    model = None
            
            # 4. Train Model (if not loaded from cache)
            if model is None:
                prophet_kwargs = {
                    "n_changepoints": 25,
                    "changepoint_range": 0.8,
                }
                holidays_df = get_uk_bank_holidays()
                
                if auto_tune:
                    logger.info("Auto-tuning hyperparameters... this may take a while.")
                    best_params = tune_hyperparameters(
                        train_df,
                        freq=freq,
                        holidays_df=holidays_df,
                        regressors=regressors,
                        n_jobs=n_jobs, # Pass n_jobs to tune_hyperparameters
                        **prophet_kwargs
                    )
                    prophet_kwargs.update(best_params)
                    logger.info(f"Tuning complete. Best params: {best_params}")

                model = train_prophet_model(
                    train_df, 
                    freq=freq, 
                    seasonality_yearly="auto",
                    seasonality_weekly="auto",
                    seasonality_daily=False,
                    holidays_df=holidays_df,
                    regressors=regressors,
                    **prophet_kwargs
                )
                
                
                # Save the trained model
                try:
                    save_model(model, str(model_path))
                    logger.info(f"Model saved to {model_path}")
                except Exception as e:
                    logger.error(f"Failed to save model: {e}")
            
            # 5. Generate Forecast
            periods = int(td_horizon / pd.to_timedelta(1, unit=freq)) # Use td_horizon directly
            forecast_df = forecast_with_prophet(
                model, 
                periods=periods, 
                freq=freq,
                future_regressors_df=df # Pass the full df containing potentially future regressor values
            )
            
            # 6. Generate Plots
            # For web, return base64. For CLI, save to file.
            plot_results = {}
            if output_dir: # CLI output
                import matplotlib.pyplot as plt # Moved local for web output before
                import io # Moved local for web output before
                import base64 # Moved local for web output before
                output_dir.mkdir(parents=True, exist_ok=True)
                fig = model.plot(forecast_df)
                plot_path = output_dir / f"{series_name}_forecast_plot.png"
                fig.savefig(plot_path)
                plt.close(fig) # Close figure to free memory
                plot_results['forecast_plot_path'] = str(plot_path)

                fig_components = model.plot_components(forecast_df)
                plot_components_path = output_dir / f"{series_name}_forecast_components_plot.png"
                fig_components.savefig(plot_components_path)
                plt.close(fig_components)
                plot_results['components_plot_path'] = str(plot_components_path)
            else: # Web output
                # Using an internal function to avoid matplotlib import outside of this block
                import matplotlib.pyplot as plt
                import io
                import base64

                def fig_to_base64(fig):
                    img = io.BytesIO()
                    fig.savefig(img, format='png', bbox_inches='tight')
                    img.seek(0)
                    return "data:image/png;base64," + base64.b64encode(img.getvalue()).decode('utf-8')

                fig = model.plot(forecast_df)
                ax = fig.gca()

                # Add black line for historical data
                ax.plot(train_df['ds'], train_df['y'], 'k-', linewidth=1, label='Historical')

                # Load and plot actual data in green
                actual_df_full = None
                try:
                    actual_df_full = load_time_series(
                        self.engine,
                        table='actual_calls',
                        ts_column=ts_column,
                        y_column=y_column,
                        columns_to_load=['ds', 'day', 'y', 'answered_calls', 'abandoned_calls'],
                    )
                    if not actual_df_full.empty:
                        ax.plot(actual_df_full['ds'], actual_df_full['y'], 'g-', linewidth=1.5, label='Actual')
                        ax.legend()
                except Exception as e:
                    logger.warning(f"Could not load actual_calls for plotting: {e}")

                plot_results['forecast_plot'] = fig_to_base64(fig)
                plt.close(fig)

                fig_components = model.plot_components(forecast_df)
                plot_results['components_plot'] = fig_to_base64(fig_components)
                plt.close(fig_components)

                # Create stacked bar chart for actual calls by day of week
                if actual_df_full is not None and not actual_df_full.empty:
                    try:
                        # Map abbreviated days to full names
                        day_map = {'mon': 'Monday', 'tue': 'Tuesday', 'wed': 'Wednesday',
                                   'thu': 'Thursday', 'fri': 'Friday', 'sat': 'Saturday', 'sun': 'Sunday'}
                        actual_df_full['day_full'] = actual_df_full['day'].str.lower().map(day_map)

                        # Order days correctly
                        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

                        # Group by day and sum
                        day_stats = actual_df_full.groupby('day_full').agg({
                            'answered_calls': 'sum',
                            'abandoned_calls': 'sum'
                        }).reindex(day_order).fillna(0)

                        fig_bar, ax_bar = plt.subplots(figsize=(10, 6))

                        x = range(len(day_stats))
                        width = 0.6

                        # Stacked bar: green for answered, red for abandoned
                        ax_bar.bar(x, day_stats['answered_calls'], width, label='Answered', color='green')
                        ax_bar.bar(x, day_stats['abandoned_calls'], width, bottom=day_stats['answered_calls'], label='Abandoned', color='red')

                        ax_bar.set_xlabel('Day of Week')
                        ax_bar.set_ylabel('Number of Calls')
                        ax_bar.set_title('Actual Calls by Day of Week (Answered vs Abandoned)')
                        ax_bar.set_xticks(x)
                        ax_bar.set_xticklabels(day_stats.index, rotation=45, ha='right')
                        ax_bar.legend()

                        plt.tight_layout()
                        plot_results['day_breakdown_plot'] = fig_to_base64(fig_bar)
                        plt.close(fig_bar)
                    except Exception as e:
                        logger.warning(f"Could not create day breakdown plot: {e}")
            
            return {
                "forecast_df": forecast_df,
                "series_name": series_name,
                **plot_results
            }

        except Exception as e:
            logger.exception(f"Error in ForecastingService.get_forecast: {e}")
            raise

    def evaluate_model(
        self,
        table_name: str,
        ts_column: str,
        y_column: str,
        freq: str,
        metrics: List[str],
        initial: str, # For CV
        period: str, # For CV
        horizon: str, # For CV
        regressors: List[str],
        n_jobs: int = -1,
        resample_to_freq: Optional[str] = None, # New parameter
        output_dir: Optional[pathlib.Path] = None,
        **prophet_kwargs,
    ) -> Dict[str, Any]:
        """
        Orchestrates data loading and model evaluation using cross-validation.
        """
        try:
            # 1. Determine Required Data Range for Loading (for Cross-Validation)
            latest_db_date = get_max_date_for_table(self.engine, table_name, ts_column)
            if latest_db_date is None:
                raise ValueError(f"No data found in table '{table_name}' or ts_column '{ts_column}' is empty for evaluation.")
            
            # Cross-validation needs data from latest_db_date back to (initial + (num_cutoffs * period) + horizon)
            # A rough estimate for required start date: latest_db_date - initial - period - horizon (with buffer)
            td_initial = pd.to_timedelta(initial)
            td_period = pd.to_timedelta(period)
            td_horizon = pd.to_timedelta(horizon)
            
            # Load enough data to cover all CV folds. This is a conservative estimate.
            # Start = latest_db_date - (number of folds * period) - initial - horizon
            # Assuming at least 3 folds for reasonable CV: 3 * td_period
            data_load_start = latest_db_date - td_initial - (3 * td_period) - td_horizon
            data_load_end = latest_db_date # CV uses historical data only

            logger.info(f"Loading data for evaluation from {data_load_start} to {data_load_end}...")

            df = load_time_series(
                self.engine, 
                table=table_name, 
                ts_column=ts_column, 
                y_column=y_column,
                regressors=regressors,
                resample_to_freq=resample_to_freq, # Pass to data_loader
                start=data_load_start, # Pass calculated start
                end=data_load_end,     # Pass calculated end
            )
            if df.empty:
                raise ValueError(f"No data found in table '{table_name}' for evaluation within the calculated range.")

            calculated_metrics = evaluate_with_cross_validation(
                df,
                freq=freq,
                metrics=metrics,
                initial=initial,
                period=period,
                horizon=horizon,
                n_jobs=n_jobs,
                regressors=regressors, # Pass regressors
                save_path=None, # Explicitly pass save_path=None as internal models shouldn't be saved
                **prophet_kwargs,
            )

            metrics_output_path = output_dir / f"{table_name}_{ts_column}_{y_column}_cv_metrics.json"
            metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_output_path, "w") as f:
                import json
                json.dump(calculated_metrics, f, indent=4)
            logger.info(f"Cross-Validation metrics saved to {metrics_output_path}")

            logger.info("Cross-Validation Metrics:")
            for metric, value in calculated_metrics.items():
                logger.info(f"  {metric.upper()}: {value:.4f}")
            
            return calculated_metrics

        except Exception as e:
            logger.exception(f"An error occurred in ForecastingService.evaluate_model: {e}")
            raise
