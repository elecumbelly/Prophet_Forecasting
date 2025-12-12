import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from prophet_forecasting_tool.config import Settings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_dummy_data():
    """Creates the table and populates it with dummy data."""
    settings = Settings()
    engine = create_engine(settings.DATABASE_URL)

    # 1. Create Table
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS call_center_metrics (
        ts TIMESTAMP WITHOUT TIME ZONE NOT NULL,
        y NUMERIC NOT NULL,
        queue TEXT,
        region TEXT
    );
    """
    try:
        with engine.begin() as conn:
            conn.execute(text(create_table_sql))
        logger.info("Table 'call_center_metrics' created (or already exists).")
    except Exception as e:
        logger.error(f"Error creating table: {e}")
        return

    # 2. Generate Dummy Data (3 years)
    logger.info("Generating 3 years of dummy data...")
    dates = pd.date_range(start="2020-01-01", end="2023-01-01", freq="D")
    
    # Base volume + trend + weekly seasonality + yearly seasonality + noise
    n = len(dates)
    trend = np.linspace(100, 150, n) # Increasing trend
    weekly = 20 * np.sin(2 * np.pi * dates.dayofweek / 7)
    yearly = 50 * np.sin(2 * np.pi * dates.dayofyear / 365)
    noise = np.random.normal(0, 10, n)
    
    y = trend + weekly + yearly + noise
    
    df = pd.DataFrame({
        'ts': dates,
        'y': y,
        'queue': 'general',
        'region': 'uk'
    })

    # 3. Insert Data
    try:
        # Check if data already exists to avoid duplication
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM call_center_metrics"))
            count = result.scalar()
            
        if count > 0:
            logger.info(f"Table already has {count} rows. Skipping insertion to avoid duplicates.")
        else:
            df.to_sql('call_center_metrics', engine, if_exists='append', index=False)
            logger.info(f"Successfully inserted {len(df)} rows of dummy data.")
            
    except Exception as e:
        logger.error(f"Error inserting data: {e}")

if __name__ == "__main__":
    create_dummy_data()
