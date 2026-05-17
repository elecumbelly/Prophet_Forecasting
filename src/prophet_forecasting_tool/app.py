"""Flask JSON API for the Prophet forecasting tool.

The Flask layer exposes only JSON endpoints. The browser UI lives in
``frontend/`` (Next.js). CORS is scoped to a configurable allowlist and
applied only on the ``/api/*`` and ``/get_columns/*`` routes.
"""
from __future__ import annotations

# Set matplotlib backend before any imports that might use it.
import matplotlib

matplotlib.use("Agg")

import logging
import os
import pathlib
import secrets
from typing import Any, Iterable, Optional

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from sqlalchemy import create_engine, inspect

from prophet_forecasting_tool.config import Settings
from prophet_forecasting_tool.data_loader import load_time_series
from prophet_forecasting_tool.logging_config import setup_logging
from prophet_forecasting_tool.services import ForecastingService
from prophet_forecasting_tool.utils import parse_duration, validate_identifier


logger = logging.getLogger(__name__)


# Plot artifacts (PNG files) are served publicly. Model JSONs are stored in a
# separate directory that is NOT served by the static route.
PLOTS_DIR_NAME = "outputs_web"
MODELS_DIR_NAME = "models_cache"


def _parse_origins(value: Optional[str]) -> list[str]:
    if not value:
        return ["http://localhost:3001", "http://127.0.0.1:3001"]
    return [o.strip() for o in value.split(",") if o.strip()]


def _origin_allowed(origin: Optional[str], allowed: Iterable[str]) -> bool:
    if not origin:
        return False
    return origin in set(allowed) or "*" in set(allowed)


def _parse_optional_date(value):
    if not value:
        return None
    try:
        ts = pd.to_datetime(value)
        if pd.isna(ts):
            return None
        return ts
    except Exception as exc:
        logger.warning(f"Could not parse date '{value}': {exc}")
        return None


def _bool_from_payload(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"true", "1", "yes", "on"}


def _int_from_payload(value: Any, default: int, *, lo: int, hi: int) -> int:
    try:
        n = int(value) if value is not None else default
    except (TypeError, ValueError):
        n = default
    return max(lo, min(hi, n))


def _regressors_from_payload(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [r.strip() for r in value.split(",") if r.strip()]
    if isinstance(value, list):
        return [str(r).strip() for r in value if str(r).strip()]
    raise ValueError("'regressors' must be a list or comma-separated string.")


def _validate_columns_exist(engine, table: str, expected: Iterable[str]) -> None:
    """Raise ValueError if any of ``expected`` is not a real column on ``table``."""
    inspector = inspect(engine)
    if not inspector.has_table(table):
        raise ValueError(f"Table '{table}' not found.")
    existing = {col["name"] for col in inspector.get_columns(table)}
    missing = [c for c in expected if c and c not in existing]
    if missing:
        raise ValueError(
            f"Column(s) {missing} not found in table '{table}'. Available: {sorted(existing)}"
        )


def create_app(
    settings: Optional[Settings] = None,
    engine=None,
    forecasting_service: Optional[ForecastingService] = None,
    *,
    output_root: Optional[pathlib.Path] = None,
) -> Flask:
    """Build a Flask app. Allows tests to inject mocked engine/service."""
    setup_logging()

    settings = settings or Settings()
    engine = engine if engine is not None else create_engine(settings.DATABASE_URL)

    output_root = pathlib.Path(output_root or "./outputs_web")
    plots_dir = output_root
    models_dir = output_root.parent / MODELS_DIR_NAME if output_root.name == PLOTS_DIR_NAME else output_root / "models"
    plots_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    service = forecasting_service or ForecastingService(engine, settings, models_dir)

    app = Flask(__name__)

    secret_key = os.getenv("FLASK_SECRET_KEY")
    if not secret_key:
        if os.getenv("FLASK_ENV") == "production":
            raise RuntimeError(
                "FLASK_SECRET_KEY must be set in production. Refusing to start."
            )
        secret_key = secrets.token_hex(32)
        logger.warning("FLASK_SECRET_KEY not set; generated an ephemeral key for this process.")
    app.config["SECRET_KEY"] = secret_key

    allowed_origins = _parse_origins(os.getenv("CORS_ALLOWED_ORIGINS"))
    app.config["ALLOWED_ORIGINS"] = allowed_origins
    app.config["PLOTS_DIR"] = plots_dir
    app.config["MODELS_DIR"] = models_dir
    app.config["SERVICE"] = service
    app.config["ENGINE"] = engine

    @app.after_request
    def add_cors_headers(response):
        if request.path.startswith("/api/") or request.path.startswith("/get_columns/"):
            origin = request.headers.get("Origin")
            if _origin_allowed(origin, allowed_origins):
                response.headers["Access-Control-Allow-Origin"] = origin
                response.headers["Vary"] = "Origin"
                response.headers["Access-Control-Allow-Headers"] = "Content-Type"
                response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
        return response

    @app.route("/healthz", methods=["GET"])
    def healthz():
        return jsonify({"status": "ok"})

    @app.route("/get_columns/<table_name>", methods=["GET", "OPTIONS"])
    def get_columns(table_name):
        if request.method == "OPTIONS":
            return ("", 204)
        try:
            validate_identifier(table_name, "table")
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        try:
            inspector = inspect(engine)
            if not inspector.has_table(table_name):
                return jsonify({"error": f"Table {table_name} not found"}), 404
            columns = [col["name"] for col in inspector.get_columns(table_name)]
            return jsonify({"columns": columns})
        except Exception as e:
            logger.exception(f"Error fetching columns for table {table_name}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/historical_data", methods=["POST", "OPTIONS"])
    def api_historical_data():
        if request.method == "OPTIONS":
            return ("", 204)

        payload = request.get_json(silent=True) or {}
        try:
            table_name = payload.get("table", "real_call_metrics")
            ts_column = payload.get("ts_column", "ds")
            y_column = payload.get("y_column", "y")
            columns = payload.get("columns")
            resample_to_freq = payload.get("resample_to_freq")

            for ident, kind in [(table_name, "table"), (ts_column, "column"), (y_column, "column")]:
                validate_identifier(ident, kind)
            if columns:
                if not isinstance(columns, list):
                    raise ValueError("'columns' must be a list of column names.")
                for c in columns:
                    validate_identifier(c, "column")

            engine_ref = app.config["ENGINE"]
            _validate_columns_exist(engine_ref, table_name, [ts_column, y_column])

            start = _parse_optional_date(payload.get("start"))
            end = _parse_optional_date(payload.get("end"))
            max_rows = _int_from_payload(payload.get("max_rows"), default=500, lo=1, hi=10_000)

            df = load_time_series(
                engine_ref,
                table=table_name,
                ts_column=ts_column,
                y_column=y_column,
                columns_to_load=columns,
                resample_to_freq=resample_to_freq,
                start=start,
                end=end,
            )
            data_preview = df.tail(max_rows)
            return jsonify(
                {
                    "table": table_name,
                    "row_count": len(df),
                    "columns": list(df.columns),
                    "preview": data_preview.to_dict(orient="records"),
                }
            )
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception:
            logger.exception("Error loading historical data (API)")
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/api/forecast", methods=["POST", "OPTIONS"])
    def api_forecast():
        if request.method == "OPTIONS":
            return ("", 204)

        payload = request.get_json(silent=True) or {}
        try:
            table_name = payload.get("table", "real_call_metrics")
            ts_column = payload.get("ts_column", "ds")
            y_column = payload.get("y_column", "y")
            for ident, kind in [(table_name, "table"), (ts_column, "column"), (y_column, "column")]:
                validate_identifier(ident, kind)

            regressors = _regressors_from_payload(payload.get("regressors"))
            freq = payload.get("freq", "D")
            if freq not in ("D", "W", "M", "H"):
                raise ValueError(f"Unsupported freq '{freq}'. Use one of D, W, M, H.")
            horizon = payload.get("horizon", "365D")
            training_window = payload.get("training_window_duration", "730 days")
            parse_duration(horizon)
            parse_duration(training_window)

            engine_ref = app.config["ENGINE"]
            _validate_columns_exist(engine_ref, table_name, [ts_column, y_column, *regressors])

            results = app.config["SERVICE"].get_forecast(
                table_name=table_name,
                ts_column=ts_column,
                y_column=y_column,
                freq=freq,
                horizon=horizon,
                series_name=str(payload.get("series_name") or "default_series"),
                regressors=regressors,
                auto_tune=_bool_from_payload(payload.get("auto_tune"), default=False),
                n_jobs=_int_from_payload(payload.get("n_jobs"), default=-1, lo=-1, hi=64),
                resample_to_freq=payload.get("resample_to_freq") or None,
                training_window_duration=training_window,
                remove_outliers=_bool_from_payload(payload.get("remove_outliers"), default=False),
            )

            preview_rows = _int_from_payload(payload.get("preview_rows"), default=50, lo=1, hi=5000)
            forecast_df = results["forecast_df"]
            preview_df = forecast_df.tail(preview_rows)

            return jsonify(
                {
                    "series_name": results["series_name"],
                    "row_count": len(forecast_df),
                    "preview": preview_df.to_dict(orient="records"),
                    "forecast_plot": results.get("forecast_plot"),
                    "components_plot": results.get("components_plot"),
                    "day_breakdown_plot": results.get("day_breakdown_plot"),
                    "training_rows": results.get("training_rows"),
                    "training_start": results.get("training_start"),
                    "training_end": results.get("training_end"),
                    "outliers_removed": results.get("outliers_removed", 0),
                }
            )
        except ValueError as e:
            logger.info(f"Forecast input error: {e}")
            return jsonify({"error": str(e)}), 400
        except Exception:
            logger.exception("Unexpected error in API forecast")
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/outputs_web/<path:filename>")
    def serve_output_file(filename):
        # Only serve files inside the plots dir; whitelist by extension.
        safe = pathlib.PurePosixPath(filename)
        if any(part in ("..", "") for part in safe.parts):
            return jsonify({"error": "Invalid path"}), 400
        if safe.suffix.lower() not in {".png", ".csv", ".json"}:
            return jsonify({"error": "File type not served"}), 404
        return send_from_directory(app.config["PLOTS_DIR"], str(safe))

    return app


# Default app for `flask run` / `python -m prophet_forecasting_tool.app`.
app = create_app()


if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", "5001"))
    debug = os.getenv("FLASK_DEBUG", "false").lower() in {"1", "true", "yes"}
    app.run(host=host, port=port, debug=debug)
