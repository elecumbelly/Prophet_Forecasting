# Prophet Forecasting Tool

A time-series forecasting service built on top of [Facebook Prophet](https://facebook.github.io/prophet/). It reads
historical metrics from PostgreSQL, trains a Prophet model (with optional UK
holiday support, custom regressors, and hyperparameter tuning), and serves
forecasts through both a CLI and a JSON HTTP API. A Next.js + shadcn/ui front
end lives in `frontend/` and consumes the API.

## Highlights

- **JSON API** at `/api/historical_data` and `/api/forecast` (Flask).
- **Next.js UI** with inline duration validation, loading skeletons, and
  downloadable forecast plots.
- **CLI** `prophet_forecaster {train-and-forecast,evaluate}` for batch use.
- **Calendar-aware durations** (`"90D"`, `"12M"`, `"2Y"`, `"730 days"`) — no
  more silent breakage on monthly horizons.
- **Holiday-aware outlier removal** (opt-in) that preserves known UK bank
  holidays so the model can still learn their effect.
- **Tuned cross-validation** that mirrors the production training spec
  (holidays + regressors), with a safe MAPE that doesn't blow up on weekends.
- **Strict identifier validation** on every table/column input — no SQL
  injection through the `table`/`ts_column`/`y_column` knobs.
- **Scoped CORS** (`localhost:3001` by default; configurable via
  `CORS_ALLOWED_ORIGINS`) and a required `FLASK_SECRET_KEY` in production.

## Tech Stack

- **Python** 3.11+ with `prophet`, `pandas`, `sqlalchemy`, `psycopg2-binary`,
  `pydantic-settings`, `matplotlib`, `scikit-learn`, `holidays`, `joblib`.
- **PostgreSQL** as the source of truth.
- **Frontend**: Next.js 16 (App Router) + Tailwind v4 + shadcn/ui + pnpm.

## Setup

```bash
uv venv && source .venv/bin/activate    # or: python -m venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"               # installs runtime + test deps
```

Create `.env` in the project root:

```ini
PGHOST=localhost
PGPORT=5432
PGUSER=your_postgres_username
PGPASSWORD=your_postgres_password
PGDATABASE=your_database_name
LOG_LEVEL=INFO

# Required in production; leave unset locally to auto-generate per-process.
# FLASK_SECRET_KEY=<openssl rand -hex 32>

# Optional: comma-separated list of frontends allowed to call the API.
# CORS_ALLOWED_ORIGINS=http://localhost:3001
```

### Seed dummy data (optional)

```bash
uv run python setup_db.py        # idempotent; use --force to re-seed
```

This creates `call_center_metrics(ts, y, queue, region)` with three years of
synthetic data — the table the UI defaults to.

### Import a CSV

```bash
uv run python import_real_data.py --csv ../raw_data/Call\ Center\ Data.csv --force
```

This writes to `real_call_metrics`. `--force` is required because it `REPLACE`s
the table.

## Running the API

```bash
uv run python -m prophet_forecasting_tool.app
```

The API listens on `http://127.0.0.1:5001` by default (override with `FLASK_HOST`
and `FLASK_PORT`). Useful endpoints:

| Method | Path                       | Notes                                       |
| ------ | -------------------------- | ------------------------------------------- |
| GET    | `/healthz`                 | Liveness check.                             |
| GET    | `/get_columns/<table>`     | Column list for the given table.            |
| POST   | `/api/historical_data`     | Returns a preview of the requested range.   |
| POST   | `/api/forecast`            | Runs a forecast and returns plots + tail.   |
| GET    | `/outputs_web/<file>`      | Serves saved PNG/CSV/JSON artifacts.        |

Model JSONs are cached on disk in `models_cache/` (sibling of `outputs_web/`)
and are **not** served by the HTTP route — only `.png`, `.csv`, and `.json`
artifacts inside `outputs_web/` are reachable.

## Running the React UI

```bash
cd frontend
pnpm install
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:5001 pnpm dev
```

Visit <http://localhost:3001>. Defaults match the seeded dummy data:
`call_center_metrics` with `ts`/`y`.

## CLI Usage

```bash
prophet_forecaster train-and-forecast \
  --series-name my_forecast \
  --table call_center_metrics \
  --ts-column ts \
  --y-column y \
  --output-dir ./outputs \
  --horizon 90D \
  --freq D \
  --auto-tune \
  --remove-outliers \
  --regressors answer_rate
```

```bash
prophet_forecaster evaluate \
  --series-name my_forecast \
  --table call_center_metrics \
  --ts-column ts \
  --y-column y \
  --output-dir ./outputs \
  --metrics mae,rmse,mape \
  --initial "730 days" \
  --period "180 days" \
  --horizon "365 days" \
  --n-jobs 4
```

`--n-jobs` accepts integers; CV parallelism is auto-mapped to Prophet's
`parallel="processes"` mode.

## Python API

```python
from sqlalchemy import create_engine

from prophet_forecasting_tool.config import Settings
from prophet_forecasting_tool.services import ForecastingService

settings = Settings()
engine = create_engine(settings.DATABASE_URL)
service = ForecastingService(engine, settings, models_dir="./models_cache")

result = service.get_forecast(
    table_name="call_center_metrics",
    ts_column="ts",
    y_column="y",
    freq="D",
    horizon="90D",
    series_name="example",
    regressors=[],
    auto_tune=False,
    remove_outliers=False,
    training_window_duration="730 days",
)
print(result["forecast_df"].tail())
```

## Tests

```bash
uv run pytest
```

The suite covers:

- duration parsing and identifier validation (`tests/test_utils.py`),
- the SQLAlchemy data loader with a SQLite integration test
  (`tests/test_data_loader.py`),
- the Prophet wrapper and holiday windows (`tests/test_model.py`),
- the forecasting service orchestration (`tests/test_services.py`),
- the JSON API end-to-end (`tests/test_api.py`),
- the CV evaluator and safe MAPE (`tests/test_evaluate.py`).

## Project layout

```
src/prophet_forecasting_tool/
  app.py             # Flask app factory + JSON API
  cli.py             # prophet_forecaster CLI
  config.py          # pydantic-settings (.env)
  data_loader.py     # PostgreSQL → DataFrame (SQLAlchemy Core, validated identifiers)
  evaluate.py        # Rolling-window CV with safe MAPE
  logging_config.py  # Rotating file + console logging
  model.py           # Prophet wrapper, tuning, UK holidays
  services.py        # Pipeline: load → tune → train → forecast → plot
  utils.py           # Duration parsing, identifier validation, tz normalization

frontend/            # Next.js 16 + shadcn/ui UI
tests/               # Pytest suite
```
