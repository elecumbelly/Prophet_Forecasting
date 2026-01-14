# Prophet Forecasting Tool

## Description
This is a comprehensive time series forecasting utility that leverages Facebook Prophet. It's designed to read historical data from PostgreSQL, train robust forecasting models, and generate predictions for future periods. The tool now features a flexible data loading mechanism, supports custom regressors, includes automated hyperparameter tuning, and uses robust cross-validation for evaluation. It offers a powerful CLI, a reusable Python API, and an interactive Flask-based web front end.

## Features
-   Reusable Python module wrapping Prophet for time series forecasting.
-   **Service-Oriented Architecture:** Core logic is encapsulated in a service layer for cleaner separation of concerns.
-   Loads historical call centre data from a PostgreSQL table with **flexible start/end dates and optional resampling**.
-   **Secure database interactions:** Uses SQLAlchemy expressions to prevent SQL injection.
-   **Model Caching:** Automatically saves and loads trained Prophet models based on configuration for faster predictions.
-   **Regressor Support:** Allows inclusion of additional regressor columns from the dataset.
-   **Automated Hyperparameter Tuning:** Optimizes model parameters using cross-validation (with parallel processing).
-   **Dynamic UK holiday support:** Automatically calculates UK bank holidays for any given year.
-   **Configurable Training Window:** Uses a user-defined training window duration, rather than fixed "two years".
-   Supports forecasting daily, weekly, or monthly aggregates via configuration.
-   **Robust Cross-Validation:** Evaluation now uses Prophet's built-in, rolling-window cross-validation.
-   Provides plots and summary metrics for forecast evaluation.
-   Includes a web interface for interactive forecasting and data visualization.
-   **Outlier Removal:** Automatically removes outliers from training data using the IQR (Interquartile Range) method.
-   **Forecast vs Actual Comparison:** Plots overlay historical data (black), forecast (blue), and actual data (green) for visual comparison.
-   **Day-of-Week Breakdown:** Generates stacked bar charts showing answered (green) vs abandoned (red) calls by day of week.

## Tech Stack
-   **Language**: Python 3.11+
-   **Main Libraries**: `prophet`, `pandas`, `sqlalchemy`, `psycopg2-binary`, `pydantic`, `matplotlib`, `python-dotenv`, `pydantic-settings`, `flask`, `scikit-learn`, `holidays`
-   **Database**: PostgreSQL (using SQLAlchemy Core)

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd prophet_forecasting_tool
    ```

2.  **Create and activate a virtual environment (using `uv` is recommended):**
    ```bash
    uv venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    uv pip install -e .
    ```

## Configuration

The tool requires PostgreSQL connection details.

1.  **Create a `.env` file** in the project root. You can use the following template:

    ```ini
    PGHOST=localhost
    PGPORT=5432
    PGUSER=your_postgres_username
    PGPASSWORD=your_postgres_password
    PGDATABASE=your_database_name
    LOG_LEVEL=INFO
    ```

2.  **Edit the `.env` file** with your actual credentials.

    *   **Note:** If you don't have a specific database user, try `PGUSER=postgres` or your system username.
    *   **Note:** If you don't have a specific database, try `PGDATABASE=postgres`.

## Database Setup and Data Import

### 1. Quick Start with Dummy Data

If you don't have existing call center data, you can use the provided helper script to create the `call_center_metrics` table and populate it with 3 years of dummy data.

1.  **Run the setup script:**
    ```bash
    uv run python setup_db.py
    ```
    This will connect to your database (using `.env` credentials), create the table, and insert sample data.

### 2. Importing Real Data (CSV)

If you have your own data in CSV format, you can use the `import_real_data.py` script. This script expects a CSV file with specific column names (like 'date', 'Incoming Calls', etc.) and imports it into a table named `real_call_metrics`.

1.  **Place your CSV file:** Ensure your CSV file (e.g., `Call Center Data.csv`) is located at:
    `../raw_data/Call Center Data.csv` (relative to this project directory)

2.  **Run the import script:**
    ```bash
    uv run python import_real_data.py
    ```
    This will connect to your database, create the `real_call_metrics` table, and import your data.

    *   **Note:** The application defaults (`--table real_call_metrics`, `--ts-column ds`, `--y-column y`) are configured to work with this imported data.


## Web Front End (Flask)

A Flask web application is provided for interactive use.

### Running the Web Application

1.  **Start the Flask application:**
    ```bash
    uv run python src/prophet_forecasting_tool/app.py
    ```
    The application runs on port **5001**.

2.  **Access the application:**
    Open your web browser and navigate to:
    [http://127.0.0.1:5001](http://127.0.0.1:5001)

3.  **Features & Parameters:**
    *   **Defaults:** The UI now defaults to `real_call_metrics` as table, `ds` as timestamp column, and `y` as value column.
    *   **Historical Data:** View all columns of your raw data from the database.
    *   **Forecast:** Configure parameters and generate forecasts with plots. Parameters include:
        *   **Additional Regressors:** Select columns to use as external regressors.
        *   **Auto-tune Hyperparameters:** Automatically optimize Prophet's tuning parameters.
        *   **Resample To:** Resample your input data to a different frequency (e.g., 'H', 'D', 'W') before forecasting.
        *   **Training Window Duration:** Define the duration of the historical data used for training (e.g., '365 days', '2Y').
    *   **High Performance:** Plots are generated in-memory, avoiding disk I/O bottlenecks. Models are cached to avoid retraining identical configurations.
    *   **Visualizations:** The forecast page displays:
        *   **Forecast Plot:** Shows historical data (black line), forecast (blue line with uncertainty interval), and actual data (green line) if available in the `actual_calls` table.
        *   **Components Plot:** Breaks down the forecast into trend, holidays, and weekly seasonality.
        *   **Day-of-Week Breakdown:** Stacked bar chart showing answered (green) vs abandoned (red) calls by day of week from actual data.

### Troubleshooting the Web Server

-   **Port in use:** If port 5001 is busy, the server won't start. Check for other running processes.
-   **Database Errors:** Check `app.log` in the project root if you encounter connection issues. Ensure your `.env` is correct.

## React Front End (shadcn/ui)

A Next.js + Tailwind + shadcn/ui front end lives in `frontend/` for a richer UI that calls the Flask API.

1.  Start the Flask API as above (defaults to http://127.0.0.1:5001).
2.  Run the React app:
    ```bash
    cd frontend
    NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:5001 npm run dev
    ```
    Visit http://localhost:3000 and use `call_center_metrics` with `ts`/`y` for the seeded dummy data.

## CLI Usage

The main entry point for the CLI is `prophet_forecaster`.

### Common Options (for both subcommands)
-   `--table`: PostgreSQL table name (default: `real_call_metrics`).
-   `--ts-column`: Timestamp column name (default: `ds`).
-   `--y-column`: Target value column name (default: `y`).
-   `--freq`: Frequency of the time series (D/W/M, default: `D`).
-   `--output-dir`: Directory to save output files (forecasts, plots, metrics).
-   `--log-level`: Set the logging level.
-   `--regressors`: Comma-separated list of additional regressor column names (e.g., `answer_rate,day`).
-   `--auto-tune`: Enable automatic hyperparameter tuning (flag).
-   `--n-jobs`: Number of parallel jobs for tuning/cross-validation (-1 for all CPUs, default: -1).
-   `--resample-to-freq`: Optional frequency to resample input data to (e.g., 'H', 'D', 'W').
-   `--training-window-duration`: Duration of the training window (e.g., '365 days', '2Y', default: '730 days').

### 1. Train and Forecast

Trains a Prophet model and generates a forecast.

```bash
prophet_forecaster train-and-forecast \
  --series-name "my_call_forecast" \
  --table real_call_metrics \
  --ts-column ds \
  --y-column y \
  --output-dir "./outputs" \
  --horizon "90D" \
  --freq "D" \
  --auto-tune \
  --regressors "answered_calls,abandoned_calls"
```

### 2. Evaluate

Performs a robust cross-validation on historical data.

```bash
prophet_forecaster evaluate \
  --series-name "my_call_evaluation" \
  --table real_call_metrics \
  --ts-column ds \
  --y-column y \
  --output-dir "./outputs" \
  --metrics "mae,rmse,mape" \
  --initial "730 days" \
  --period "180 days" \
  --horizon "365 days" \
  --n-jobs 4
```

## Python API Usage

You can use the core functionalities as a library:

```python
from sqlalchemy import create_engine
from prophet_forecasting_tool.config import Settings
from prophet_forecasting_tool.data_loader import load_time_series, get_max_date_for_table
from prophet_forecasting_tool.model import train_prophet_model, forecast_with_prophet

settings = Settings()
engine = create_engine(settings.DATABASE_URL)

# Load Data (e.g., for the last 3 years to cover training and forecasting)
# You would typically query get_max_date_for_table and calculate start/end
latest_db_date = get_max_date_for_table(engine, "real_call_metrics", "ds")
if latest_db_date:
    data_load_start = latest_db_date - pd.to_timedelta("3Y")
    data_load_end = latest_db_date + pd.to_timedelta("1Y") # Cover some future for regressors

df = load_time_series(
    engine, 
    table="real_call_metrics", 
    ts_column="ds", 
    y_column="y",
    start=data_load_start,
    end=data_load_end,
    resample_to_freq="D" # Example: resample to daily
)

# Prepare training data (e.g., last 2 years of loaded data)
train_end_date = df['ds'].min() + pd.to_timedelta("2Y")
train_df = df[df['ds'] < train_end_date]

# Train Model
model = train_prophet_model(train_df)

# Forecast (e.g., 365 days)
periods = int(pd.to_timedelta("365D") / pd.to_timedelta(1, unit="D"))
forecast = forecast_with_prophet(model, periods=periods, freq="D")
print(forecast.head())
```
