"""Data-loader tests, including a SQLite integration smoke test."""
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from sqlalchemy import (
    Column,
    Float,
    MetaData,
    String,
    Table,
    create_engine,
    insert,
)
from sqlalchemy.types import DateTime

from prophet_forecasting_tool.data_loader import load_time_series


@pytest.fixture
def memory_engine():
    """SQLite in-memory engine seeded with a tiny call_center table."""
    engine = create_engine("sqlite:///:memory:")
    metadata = MetaData()
    tbl = Table(
        "demo_metrics",
        metadata,
        Column("ts", DateTime, primary_key=True),
        Column("y", Float),
        Column("answer_rate", Float),
        Column("region", String),
    )
    metadata.create_all(engine)
    with engine.begin() as conn:
        for i in range(14):
            conn.execute(
                insert(tbl).values(
                    ts=datetime(2024, 1, 1 + i),
                    y=100 + i,
                    answer_rate=0.9 - 0.01 * i,
                    region="uk",
                )
            )
    yield engine
    engine.dispose()


class TestLoadTimeSeriesUnit:
    @patch("prophet_forecasting_tool.data_loader.pd.read_sql")
    def test_success(self, mock_read_sql):
        sample = {
            "ds": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "y": [10, 20],
        }
        mock_read_sql.return_value = pd.DataFrame(sample)
        mock_engine = MagicMock(spec=create_engine("sqlite:///:memory:"))
        df = load_time_series(mock_engine, table="test_table")
        assert not df.empty
        assert "ds" in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df["ds"])

    def test_rejects_bad_table(self):
        engine = create_engine("sqlite:///:memory:")
        with pytest.raises(ValueError):
            load_time_series(engine, table="drop;tables")

    def test_rejects_bad_column(self):
        engine = create_engine("sqlite:///:memory:")
        with pytest.raises(ValueError):
            load_time_series(engine, table="t", ts_column="ts; --")

    def test_rejects_non_engine(self):
        with pytest.raises(TypeError):
            load_time_series("not-an-engine", table="t")  # type: ignore[arg-type]


class TestLoadTimeSeriesIntegration:
    def test_basic(self, memory_engine):
        df = load_time_series(memory_engine, table="demo_metrics", ts_column="ts", y_column="y")
        assert len(df) == 14
        assert list(df.columns)[:2] == ["ds", "y"]

    def test_regressors(self, memory_engine):
        df = load_time_series(
            memory_engine,
            table="demo_metrics",
            ts_column="ts",
            y_column="y",
            regressors=["answer_rate"],
        )
        assert "answer_rate" in df.columns

    def test_resample_uses_mean_for_rate_columns(self, memory_engine):
        df = load_time_series(
            memory_engine,
            table="demo_metrics",
            ts_column="ts",
            y_column="y",
            regressors=["answer_rate"],
            resample_to_freq="W",
        )
        # answer_rate is mean-aggregated; values must stay in [0, 1].
        assert df["answer_rate"].between(0, 1).all()
