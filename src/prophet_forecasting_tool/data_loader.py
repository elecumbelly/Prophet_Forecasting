"""Time-series data loading from PostgreSQL with identifier validation."""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import Column, MetaData, Table, select
from sqlalchemy.engine import Engine

from prophet_forecasting_tool.utils import validate_identifier, normalize_tz_naive

logger = logging.getLogger(__name__)

# Columns whose summed value is meaningless when resampling; aggregate as mean.
_RATE_NAME_RE = re.compile(r"(rate|pct|percent|ratio|level|sla)$", re.IGNORECASE)


def _aggregation_for(column_name: str, dtype) -> str:
    """Pick a resample aggregation: sum for counts, mean for rates, first for text."""
    if not pd.api.types.is_numeric_dtype(dtype):
        return "first"
    if _RATE_NAME_RE.search(column_name):
        return "mean"
    return "sum"


def get_max_date_for_table(engine: Engine, table: str, ts_column: str) -> Optional[pd.Timestamp]:
    """Return the maximum timestamp from ``table.ts_column`` or ``None``."""
    validate_identifier(table, "table")
    validate_identifier(ts_column, "column")
    try:
        metadata = MetaData()
        tbl = Table(table, metadata, Column(ts_column))
        stmt = select(tbl.c[ts_column]).order_by(tbl.c[ts_column].desc()).limit(1)
        with engine.connect() as connection:
            result = connection.execute(stmt).scalar_one_or_none()
            if result is None:
                return None
            return normalize_tz_naive(pd.to_datetime(result))
    except Exception as e:
        logger.error(f"Error getting max date for table {table}: {e}")
        return None


def load_time_series(
    engine: Engine,
    table: str = "call_center_metrics",
    ts_column: str = "ts",
    y_column: str = "y",
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    filters: Optional[Dict[str, Any]] = None,
    regressors: Optional[List[str]] = None,
    resample_to_freq: Optional[str] = None,
    columns_to_load: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load a time series, renaming ``ts_column``→``ds`` and ``y_column``→``y``.

    Identifiers are validated against a strict regex. On resample, ``y`` is
    summed by default; columns whose names match ``*_rate``/``*_pct``/etc. are
    averaged; non-numeric columns take the first value in the bucket.
    """
    if not isinstance(engine, Engine):
        raise TypeError("The 'engine' argument must be an SQLAlchemy Engine instance.")

    validate_identifier(table, "table")
    validate_identifier(ts_column, "column")
    validate_identifier(y_column, "column")
    if regressors:
        for r in regressors:
            validate_identifier(r, "regressor column")
    if columns_to_load:
        for c in columns_to_load:
            validate_identifier(c, "column")
    if filters:
        for k in filters.keys():
            validate_identifier(k, "filter column")

    try:
        metadata = MetaData()
        if columns_to_load:
            tbl_cols = [Column(col_name) for col_name in columns_to_load]
            tbl = Table(table, metadata, *tbl_cols)

            select_cols = []
            for col_name in columns_to_load:
                if col_name == ts_column:
                    select_cols.append(tbl.c[col_name].label("ds"))
                elif col_name == y_column:
                    select_cols.append(tbl.c[col_name].label("y"))
                else:
                    select_cols.append(tbl.c[col_name])
            stmt = select(*select_cols).select_from(tbl)
        else:
            columns = [Column(ts_column), Column(y_column)]
            known_cols = {ts_column, y_column}
            if filters:
                for col_name in filters.keys():
                    if col_name not in known_cols:
                        columns.append(Column(col_name))
                        known_cols.add(col_name)
            if regressors:
                for col_name in regressors:
                    if col_name not in known_cols:
                        columns.append(Column(col_name))
                        known_cols.add(col_name)

            tbl = Table(table, metadata, *columns)
            selection = [tbl.c[ts_column].label("ds"), tbl.c[y_column].label("y")]
            if regressors:
                for col_name in regressors:
                    selection.append(tbl.c[col_name])
            stmt = select(*selection)

        if start is not None and pd.notna(start):
            stmt = stmt.where(tbl.c[ts_column] >= start)
        if end is not None and pd.notna(end):
            stmt = stmt.where(tbl.c[ts_column] <= end)
        if filters:
            for key, value in filters.items():
                stmt = stmt.where(tbl.c[key] == value)

        stmt = stmt.order_by(tbl.c[ts_column])

        logger.debug(f"Executing load_time_series query against table {table}")
        df = pd.read_sql(stmt, engine)

        if ts_column in df.columns and ts_column != "ds":
            df = df.rename(columns={ts_column: "ds"})
        df["ds"] = pd.to_datetime(df["ds"])
        if pd.api.types.is_datetime64tz_dtype(df["ds"]):
            df["ds"] = df["ds"].dt.tz_localize(None)

        if resample_to_freq:
            logger.info(f"Resampling data to {resample_to_freq}...")
            df = df.set_index("ds")
            agg_dict: Dict[str, str] = {}
            for col in df.columns:
                if col == "y":
                    agg_dict[col] = "sum"
                else:
                    agg_dict[col] = _aggregation_for(col, df[col].dtype)
            df = df.resample(resample_to_freq).agg(agg_dict).reset_index()
            df = df.dropna(subset=["y"]).reset_index(drop=True)
            logger.info(f"Resampled to {resample_to_freq}: {len(df)} rows.")

        logger.info(f"Loaded {len(df)} rows from {table}.")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {table}: {e}")
        raise
