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


@app.after_request
def add_cors_headers(response):
    """Add basic CORS headers so a separate front end (e.g., Next.js) can call the API."""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return response


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


def _parse_optional_date(value):
    """Safely parse a date/time value from JSON inputs."""
    if not value:
        return None
    try:
        return pd.to_datetime(value)
    except Exception as exc:
        logger.warning(f"Could not parse date '{value}': {exc}")
        return None


@app.route('/api/historical_data', methods=['POST', 'OPTIONS'])
def api_historical_data():
    """JSON API: return recent historical data for a given table/column set."""
    if request.method == 'OPTIONS':
        return ('', 204)

    payload = request.get_json(silent=True) or {}
    table_name = payload.get('table', 'real_call_metrics')
    ts_column = payload.get('ts_column', 'ds')
    y_column = payload.get('y_column', 'y')
    columns = payload.get('columns')
    resample_to_freq = payload.get('resample_to_freq')
    start = _parse_optional_date(payload.get('start'))
    end = _parse_optional_date(payload.get('end'))

    try:
        df = load_time_series(
            engine,
            table=table_name,
            ts_column=ts_column,
            y_column=y_column,
            columns_to_load=columns,
            resample_to_freq=resample_to_freq,
            start=start,
            end=end,
        )
        max_rows = int(payload.get('max_rows', 500))
        data_preview = df.tail(max_rows)
        return jsonify({
            'table': table_name,
            'row_count': len(df),
            'columns': list(df.columns),
            'preview': data_preview.to_dict(orient='records'),
        })
    except Exception as e:
        logger.exception("Error loading historical data (API)")
        return jsonify({'error': str(e)}), 400


@app.route('/api/forecast', methods=['POST', 'OPTIONS'])
def api_forecast():
    """JSON API: run a forecast and return plots + tail of the forecast dataframe."""
    if request.method == 'OPTIONS':
        return ('', 204)

    payload = request.get_json(silent=True) or {}
    regressors = payload.get('regressors') or []
    if isinstance(regressors, str):
        regressors = [r.strip() for r in regressors.split(',') if r.strip()]

    try:
        results = forecasting_service.get_forecast(
            table_name=payload.get('table', 'real_call_metrics'),
            ts_column=payload.get('ts_column', 'ds'),
            y_column=payload.get('y_column', 'y'),
            freq=payload.get('freq', 'D'),
            horizon=payload.get('horizon', '365D'),
            series_name=payload.get('series_name', 'default_series'),
            regressors=regressors,
            auto_tune=bool(payload.get('auto_tune', False)),
            n_jobs=int(payload.get('n_jobs', -1)),
            resample_to_freq=payload.get('resample_to_freq'),
            training_window_duration=payload.get('training_window_duration', "730 days"),
        )

        preview_rows = int(payload.get('preview_rows', 50))
        forecast_df = results['forecast_df']
        preview_df = forecast_df.tail(preview_rows)

        return jsonify({
            'series_name': results['series_name'],
            'row_count': len(forecast_df),
            'preview': preview_df.to_dict(orient='records'),
            'forecast_plot': results.get('forecast_plot'),
            'components_plot': results.get('components_plot'),
            'day_breakdown_plot': results.get('day_breakdown_plot'),
        })
    except ValueError as e:
        logger.warning(f"Input error in API forecast: {e}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.exception("Unexpected error in API forecast")
        return jsonify({'error': str(e)}), 500


@app.route('/outputs_web/<filename>')
def serve_output_file(filename):
    """Serves files from the outputs_web directory."""
    return send_from_directory(OUTPUT_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
