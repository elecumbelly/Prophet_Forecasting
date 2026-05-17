"""End-to-end smoke test: SQLite-backed engine + real ForecastingService.

This is a single realistic round-trip through the Flask API that exercises
the load -> outlier check -> train -> forecast -> plot pipeline against a
SQLite database seeded with synthetic data. Prophet training takes a few
seconds, so this test is intentionally compact (one happy path, one error
path).
"""
from datetime import datetime, timedelta

import numpy as np
import pytest
from sqlalchemy import (
    Column,
    Float,
    MetaData,
    Table,
    create_engine,
    insert,
)
from sqlalchemy.types import DateTime

from prophet_forecasting_tool.app import create_app
from prophet_forecasting_tool.config import Settings
from prophet_forecasting_tool.services import ForecastingService


@pytest.fixture(scope="module")
def seeded_sqlite(tmp_path_factory):
    """A SQLite DB with ~400 days of synthetic call-centre data."""
    db_path = tmp_path_factory.mktemp("e2e") / "test.db"
    engine = create_engine(f"sqlite:///{db_path}")

    metadata = MetaData()
    tbl = Table(
        "demo_metrics",
        metadata,
        Column("ts", DateTime, primary_key=True),
        Column("y", Float),
    )
    metadata.create_all(engine)

    rng = np.random.default_rng(seed=42)
    rows = []
    start = datetime(2023, 1, 1)
    for i in range(400):
        # Trend + weekly seasonality + noise — realistic enough for Prophet.
        ts = start + timedelta(days=i)
        trend = 100 + 0.1 * i
        weekly = 20 * np.sin(2 * np.pi * ts.weekday() / 7)
        noise = rng.normal(0, 5)
        rows.append({"ts": ts, "y": float(trend + weekly + noise)})

    with engine.begin() as conn:
        conn.execute(insert(tbl), rows)

    yield engine, db_path
    engine.dispose()


@pytest.fixture(scope="module")
def e2e_app(seeded_sqlite, tmp_path_factory):
    engine, _ = seeded_sqlite
    settings = Settings()
    output_root = tmp_path_factory.mktemp("e2e_outputs")
    models_dir = output_root / "models_cache"
    service = ForecastingService(engine, settings, models_dir)
    return create_app(
        settings=settings,
        engine=engine,
        forecasting_service=service,
        output_root=output_root / "outputs_web",
    )


@pytest.fixture
def e2e_client(e2e_app):
    e2e_app.config["TESTING"] = True
    with e2e_app.test_client() as client:
        yield client


class TestEndToEndForecast:
    def test_full_pipeline_returns_forecast_and_plot(self, e2e_client):
        resp = e2e_client.post(
            "/api/forecast",
            json={
                "table": "demo_metrics",
                "ts_column": "ts",
                "y_column": "y",
                "freq": "D",
                "horizon": "14D",
                "series_name": "e2e_test",
                "regressors": [],
                "training_window_duration": "200 days",
                "auto_tune": False,
                "remove_outliers": False,
                "preview_rows": 10,
            },
        )
        assert resp.status_code == 200, resp.json
        body = resp.json
        assert body["series_name"] == "e2e_test"
        # Training data + 14 future periods.
        assert body["row_count"] >= 200
        assert body["training_rows"] >= 100
        assert len(body["preview"]) == 10
        # Preview must contain Prophet's standard output columns.
        row = body["preview"][0]
        assert "ds" in row
        assert "yhat" in row
        # ds must be ISO-8601 string, not a Timestamp blob.
        assert isinstance(row["ds"], str)
        assert "T" in row["ds"]
        # At least the forecast plot must be a base64 data URL.
        assert body["forecast_plot"].startswith("data:image/png;base64,")
        assert body["components_plot"].startswith("data:image/png;base64,")

    def test_historical_endpoint_returns_jsonable_rows(self, e2e_client):
        resp = e2e_client.post(
            "/api/historical_data",
            json={
                "table": "demo_metrics",
                "ts_column": "ts",
                "y_column": "y",
                "max_rows": 5,
            },
        )
        assert resp.status_code == 200, resp.json
        body = resp.json
        assert body["row_count"] >= 5
        assert isinstance(body["preview"][0]["ds"], str)

    def test_get_columns_against_real_db(self, e2e_client):
        resp = e2e_client.get("/get_columns/demo_metrics")
        assert resp.status_code == 200
        assert set(resp.json["columns"]) == {"ts", "y"}

    def test_readyz_against_real_db(self, e2e_client):
        resp = e2e_client.get("/readyz")
        assert resp.status_code == 200
        assert resp.json["database"] == "reachable"

    def test_unknown_table_returns_400(self, e2e_client):
        resp = e2e_client.post(
            "/api/forecast",
            json={
                "table": "does_not_exist",
                "ts_column": "ts",
                "y_column": "y",
                "freq": "D",
                "horizon": "14D",
                "series_name": "missing",
                "regressors": [],
            },
        )
        assert resp.status_code == 400
        assert "does_not_exist" in resp.json["error"]
