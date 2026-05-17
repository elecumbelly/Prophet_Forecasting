"""Seed PostgreSQL with three years of dummy call-center data for demos.

Idempotent: if the table already has rows, this script leaves it alone unless
``--force`` is passed.
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from prophet_forecasting_tool.config import Settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TABLE_NAME = "call_center_metrics"


def create_dummy_data(force: bool = False) -> None:
    settings = Settings()
    engine = create_engine(settings.DATABASE_URL)
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                        ts TIMESTAMP WITHOUT TIME ZONE NOT NULL,
                        y NUMERIC NOT NULL,
                        queue TEXT,
                        region TEXT
                    )
                    """
                )
            )
        logger.info(f"Table '{TABLE_NAME}' ready.")

        with engine.connect() as conn:
            existing = conn.execute(text(f"SELECT COUNT(*) FROM {TABLE_NAME}")).scalar() or 0
        if existing > 0 and not force:
            logger.info(f"{TABLE_NAME} already has {existing} rows; skipping insert. Use --force to re-seed.")
            return
        if force and existing > 0:
            with engine.begin() as conn:
                conn.execute(text(f"DELETE FROM {TABLE_NAME}"))
            logger.warning(f"--force: cleared {existing} existing rows.")

        rng = np.random.default_rng(seed=42)
        dates = pd.date_range(start="2020-01-01", end="2023-01-01", freq="D")
        n = len(dates)
        trend = np.linspace(100, 150, n)
        weekly = 20 * np.sin(2 * np.pi * dates.dayofweek / 7)
        yearly = 50 * np.sin(2 * np.pi * dates.dayofyear / 365)
        noise = rng.normal(0, 10, n)
        df = pd.DataFrame({
            "ts": dates,
            "y": trend + weekly + yearly + noise,
            "queue": "general",
            "region": "uk",
        })

        df.to_sql(TABLE_NAME, engine, if_exists="append", index=False)
        logger.info(f"Inserted {len(df)} rows of dummy data.")
    finally:
        engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed dummy data into PostgreSQL.")
    parser.add_argument("--force", action="store_true", help="Clear and re-seed existing rows.")
    args = parser.parse_args()
    create_dummy_data(force=args.force)


if __name__ == "__main__":
    main()
