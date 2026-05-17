"""Optional PostgreSQL integration test.

Runs only when ``PG_TEST_URL`` is set. Example::

    PG_TEST_URL=postgresql+psycopg2://postgres@localhost/test \\
        pytest tests/test_postgres.py

Each test cleans up the table it created so the suite is re-runnable.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import (
    Column,
    Float,
    MetaData,
    Table,
    create_engine,
    insert,
    text,
)
from sqlalchemy.types import DateTime, TIMESTAMP

from prophet_forecasting_tool.data_loader import (
    get_max_date_for_table,
    load_time_series,
)

PG_TEST_URL = os.getenv("PG_TEST_URL")

pytestmark = pytest.mark.skipif(
    not PG_TEST_URL,
    reason="PG_TEST_URL not set; skipping Postgres integration tests.",
)


@pytest.fixture(scope="module")
def pg_engine():
    engine = create_engine(PG_TEST_URL)
    yield engine
    engine.dispose()


@pytest.fixture
def seeded_table(pg_engine):
    """Create a unique table, seed it, drop it on teardown."""
    table_name = f"pf_test_{uuid.uuid4().hex[:8]}"
    metadata = MetaData()
    tbl = Table(
        table_name,
        metadata,
        Column("ts", DateTime, primary_key=True),
        Column("y", Float),
    )
    metadata.create_all(pg_engine)

    rng = np.random.default_rng(seed=7)
    rows = []
    start = datetime(2024, 1, 1)
    for i in range(30):
        rows.append({"ts": start + timedelta(days=i), "y": float(100 + i + rng.normal(0, 1))})
    with pg_engine.begin() as conn:
        conn.execute(insert(tbl), rows)

    yield table_name

    with pg_engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))


@pytest.fixture
def seeded_tz_table(pg_engine):
    """Same as ``seeded_table`` but with TIMESTAMPTZ to exercise tz handling."""
    table_name = f"pf_test_tz_{uuid.uuid4().hex[:8]}"
    metadata = MetaData()
    tbl = Table(
        table_name,
        metadata,
        Column("ts", TIMESTAMP(timezone=True), primary_key=True),
        Column("y", Float),
    )
    metadata.create_all(pg_engine)

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows = [{"ts": start + timedelta(days=i), "y": float(50 + i)} for i in range(10)]
    with pg_engine.begin() as conn:
        conn.execute(insert(tbl), rows)

    yield table_name

    with pg_engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))


class TestPostgresIntegration:
    def test_load_returns_rows(self, pg_engine, seeded_table):
        df = load_time_series(pg_engine, table=seeded_table, ts_column="ts", y_column="y")
        assert len(df) == 30
        assert list(df.columns)[:2] == ["ds", "y"]
        assert pd.api.types.is_datetime64_any_dtype(df["ds"])

    def test_get_max_date(self, pg_engine, seeded_table):
        max_ts = get_max_date_for_table(pg_engine, seeded_table, "ts")
        assert max_ts is not None
        assert max_ts.year == 2024

    def test_date_range_filter(self, pg_engine, seeded_table):
        df = load_time_series(
            pg_engine,
            table=seeded_table,
            ts_column="ts",
            y_column="y",
            start=pd.Timestamp("2024-01-10"),
            end=pd.Timestamp("2024-01-20"),
        )
        assert len(df) == 11  # inclusive both ends
        assert df["ds"].min() == pd.Timestamp("2024-01-10")
        assert df["ds"].max() == pd.Timestamp("2024-01-20")

    def test_timestamptz_normalized_to_naive(self, pg_engine, seeded_tz_table):
        df = load_time_series(pg_engine, table=seeded_tz_table, ts_column="ts", y_column="y")
        assert len(df) == 10
        assert df["ds"].dt.tz is None, "ds must be tz-naive for Prophet compatibility"

    def test_resample_weekly(self, pg_engine, seeded_table):
        df = load_time_series(
            pg_engine,
            table=seeded_table,
            ts_column="ts",
            y_column="y",
            resample_to_freq="W",
        )
        # 30 days resampled weekly should yield 4-5 buckets.
        assert 4 <= len(df) <= 6
