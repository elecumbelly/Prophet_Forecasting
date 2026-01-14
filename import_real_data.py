import pandas as pd
from sqlalchemy import create_engine, text
import os
from pathlib import Path
from dotenv import load_dotenv
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Database connection details from environment variables
PGHOST = os.getenv("PGHOST", "localhost")
PGPORT = os.getenv("PGPORT", "5432")
PGUSER = os.getenv("PGUSER", "user")
PGPASSWORD = os.getenv("PGPASSWORD", "password")
PGDATABASE = os.getenv("PGDATABASE", "prophet_forecast_db")

DATABASE_URL = f"postgresql+psycopg2://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}"

CSV_FILE_PATH = str(Path(__file__).parent.parent / "raw_data/Call Center Data.csv")
TABLE_NAME = "real_call_metrics"

def clean_column_name(col_name: str) -> str:
    """Cleans column names for SQL compatibility."""
    return (
        col_name.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("%", "pct")
        .lower()
    )

def duration_to_seconds(duration_str: str) -> int:
    """Converts HH:MM:SS string to total seconds."""
    if pd.isna(duration_str):
        return 0
    parts = str(duration_str).split(':')
    if len(parts) == 3:
        h, m, s = map(int, parts)
    elif len(parts) == 2: # Handle MM:SS format if it occurs
        h = 0
        m, s = map(int, parts)
    else:
        return 0 # Or raise error/handle
    return h * 3600 + m * 60 + s

def percentage_to_float(percent_str: str) -> float:
    """Converts percentage string (e.g., '94.01%') to float (0.9401)."""
    if pd.isna(percent_str):
        return 0.0
    return float(percent_str.replace('%', '')) / 100.0

def import_data():
    logger.info(f"Starting data import from {CSV_FILE_PATH} to PostgreSQL table {TABLE_NAME}...")
    
    try:
        # Read the CSV file
        df = pd.read_csv(CSV_FILE_PATH)

        # Clean column names
        df.columns = [clean_column_name(col) for col in df.columns]

        # Drop the 'index' column if it exists and is just an artifact
        if 'index' in df.columns:
            df = df.drop(columns=['index'])

        # Rename 'date' to 'ds'
        df = df.rename(columns={'date': 'ds'})
        
        # Rename 'incoming_calls' to 'y'
        df = df.rename(columns={'incoming_calls': 'y'})


        # Convert 'ds' to datetime
        df['ds'] = pd.to_datetime(df['ds'], format='%d/%m/%Y')

        # Convert percentage columns
        for col in ['answer_rate_pct', 'service_level_in20_seconds_pct']:
            if col in df.columns:
                df[col] = df[col].apply(percentage_to_float)

        # Convert duration columns to seconds
        for col in ['answer_speed_avg', 'talk_duration_avg']:
            if col in df.columns:
                df[col] = df[col].apply(duration_to_seconds)

        # Connect to PostgreSQL
        engine = create_engine(DATABASE_URL)

        # Import data to PostgreSQL
        df.to_sql(TABLE_NAME, engine, if_exists='replace', index=False)
        logger.info(f"Successfully imported {len(df)} rows into table '{TABLE_NAME}'.")

    except FileNotFoundError:
        logger.error(f"Error: CSV file not found at {CSV_FILE_PATH}")
    except Exception as e:
        logger.exception(f"An error occurred during data import: {e}")

if __name__ == "__main__":
    import_data()
