"""Import call-center CSV data into PostgreSQL.

The destination table is replaced only when ``--force`` is passed, so accidental
re-runs cannot drop a live dataset.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, inspect

from prophet_forecasting_tool.config import Settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CSV = Path(__file__).parent.parent / "raw_data" / "Call Center Data.csv"
TABLE_NAME = "real_call_metrics"


def clean_column_name(col_name: str) -> str:
    return (
        col_name.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("%", "pct")
        .lower()
    )


def duration_to_seconds(duration_str: str) -> int:
    if pd.isna(duration_str):
        return 0
    parts = str(duration_str).split(":")
    if len(parts) == 3:
        h, m, s = map(int, parts)
    elif len(parts) == 2:
        h, (m, s) = 0, map(int, parts)
    else:
        return 0
    return h * 3600 + m * 60 + s


def percentage_to_float(percent_str: str) -> float:
    if pd.isna(percent_str):
        return 0.0
    return float(str(percent_str).replace("%", "")) / 100.0


def import_data(csv_path: Path, force: bool) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found at {csv_path}")

    settings = Settings()
    engine = create_engine(settings.DATABASE_URL)
    try:
        if inspect(engine).has_table(TABLE_NAME) and not force:
            logger.error(
                f"Table '{TABLE_NAME}' already exists. Re-running this script will "
                f"REPLACE it. Pass --force to confirm."
            )
            return

        df = pd.read_csv(csv_path)
        df.columns = [clean_column_name(c) for c in df.columns]
        if "index" in df.columns:
            df = df.drop(columns=["index"])
        df = df.rename(columns={"date": "ds", "incoming_calls": "y"})
        df["ds"] = pd.to_datetime(df["ds"], format="%d/%m/%Y")

        for col in ("answer_rate_pct", "service_level_in20_seconds_pct"):
            if col in df.columns:
                df[col] = df[col].apply(percentage_to_float)
        for col in ("answer_speed_avg", "talk_duration_avg"):
            if col in df.columns:
                df[col] = df[col].apply(duration_to_seconds)

        df.to_sql(TABLE_NAME, engine, if_exists="replace", index=False)
        logger.info(f"Imported {len(df)} rows into '{TABLE_NAME}'.")
    finally:
        engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Import call-center CSV into PostgreSQL.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to the input CSV file.")
    parser.add_argument(
        "--force",
        action="store_true",
        help=f"Drop and replace '{TABLE_NAME}' if it already exists.",
    )
    args = parser.parse_args()
    import_data(args.csv, force=args.force)


if __name__ == "__main__":
    main()
