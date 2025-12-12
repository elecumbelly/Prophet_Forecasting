# Set matplotlib backend before any imports that might use it
import matplotlib
matplotlib.use('Agg')

from flask import Flask, render_template, request, send_from_directory, flash, jsonify
import logging
import os
import pandas as pd
from sqlalchemy import create_engine, inspect
import datetime
import pathlib

from prophet_forecasting_tool.config import Settings
from prophet_forecasting_tool.logging_config import setup_logging
from prophet_forecasting_tool.data_loader import load_time_series
from prophet_forecasting_tool.services import ForecastingService

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'a_very_secret_key_that_should_be_changed')

# Setup logging for the Flask app
setup_logging()
logger = logging.getLogger(__name__)

# Load settings and create engine once
settings = Settings()
engine = create_engine(settings.DATABASE_URL)

OUTPUT_DIR = pathlib.Path("./outputs_web")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Directory for saved models
MODELS_DIR = OUTPUT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Instantiate the Forecasting Service
forecasting_service = ForecastingService(engine, settings, MODELS_DIR)

@app.route('/')
def index():
    """Renders the home page with an option to load data and trigger a forecast."""
    # This page could display current config or a form to enter parameters
    return render_template('index.html', title="Prophet Forecasting Tool")

@app.route('/get_columns/<table_name>')
def get_columns(table_name):
    """Returns a JSON list of columns for the specified table."""
    try:
        inspector = inspect(engine)
        if not inspector.has_table(table_name):
            return jsonify({'error': f'Table {table_name} not found'}), 404
        
        columns = [col['name'] for col in inspector.get_columns(table_name)]
        return jsonify({'columns': columns})
    except Exception as e:
        logger.error(f"Error fetching columns for table {table_name}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/historical_data', methods=['GET', 'POST'])
def historical_data():
    """Displays historical data from the database."""
    if request.method == 'POST':
        table_name = request.form.get('table', 'real_call_metrics')
        ts_column = request.form.get('ts_column', 'ds')
        y_column = request.form.get('y_column', 'y')
        
        logger.info(f"Historical Data Request: table='{table_name}', ts_column='{ts_column}', y_column='{y_column}'")
        
        try:
            inspector = inspect(engine)
            if not inspector.has_table(table_name):
                flash(f"Table '{table_name}' not found.", 'danger')
                return render_template('historical_data.html', table_name=table_name, ts_column=ts_column, y_column=y_column, data_exists=False)
            
            all_columns = [col['name'] for col in inspector.get_columns(table_name)]

            df = load_time_series(engine, table=table_name, ts_column=ts_column, y_column=y_column, columns_to_load=all_columns)
            if df.empty:
                flash(f"No data found in table '{table_name}'.", 'warning')
                return render_template('historical_data.html', table_name=table_name, ts_column=ts_column, y_column=y_column, data_exists=False)
            
            # Convert DataFrame to HTML table for display
            historical_table_html = df.to_html(classes='table table-striped table-hover', header=True, index=False)
            flash(f"Successfully loaded {len(df)} rows from '{table_name}'.", 'success')
            return render_template('historical_data.html', table_name=table_name, ts_column=ts_column, y_column=y_column, historical_table_html=historical_table_html, data_exists=True)
        except Exception as e:
            flash(f"Error loading historical data: {e}", 'danger')
            logger.exception("Error loading historical data")
            return render_template('historical_data.html', data_exists=False)
    
    return render_template('historical_data.html', data_exists=False)

@app.route('/forecast', methods=['GET', 'POST'])
def forecast():
    """Triggers a forecast and displays results using the ForecastingService."""
    if request.method == 'POST':
        try:
            results = forecasting_service.get_forecast(
                table_name=request.form.get('table', 'real_call_metrics'),
                ts_column=request.form.get('ts_column', 'ds'),
                y_column=request.form.get('y_column', 'y'),
                freq=request.form.get('freq', 'D'),
                horizon=request.form.get('horizon', '365D'),
                series_name=request.form.get('series_name', 'default_series'),
                regressors=request.form.getlist('regressors'),
                auto_tune=request.form.get('auto_tune') == 'on',
                n_jobs=-1, # Pass n_jobs for parallel tuning
                resample_to_freq=request.form.get('resample_to_freq'), # New parameter
                training_window_duration=request.form.get('training_window_duration', "730 days"), # New parameter
            )
            
            # Convert forecast_df to HTML table here for rendering
            forecast_table_html = results['forecast_df'].tail().to_html(
                classes='table table-striped table-hover', header=True, index=False
            )
            
            flash("Forecast generated successfully!", 'success')
            return render_template(
                'forecast.html',
                series_name=results['series_name'],
                forecast_plot=results['forecast_plot'],
                components_plot=results['components_plot'],
                day_breakdown_plot=results.get('day_breakdown_plot'),
                forecast_table_html=forecast_table_html,
                data_exists=True
            )

        except ValueError as e: # Specific error for data issues
            flash(f"Input Error: {e}", 'warning')
            logger.warning(f"Input Error in forecast route: {e}")
            return render_template('forecast.html', data_exists=False)
        except Exception as e:
            flash(f"Error generating forecast: {e}", 'danger')
            logger.exception("Unexpected error in forecast route.")
            return render_template('forecast.html', data_exists=False)
            
    return render_template('forecast.html', data_exists=False)

@app.route('/outputs_web/<filename>')
def serve_output_file(filename):
    """Serves files from the outputs_web directory."""
    return send_from_directory(OUTPUT_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)