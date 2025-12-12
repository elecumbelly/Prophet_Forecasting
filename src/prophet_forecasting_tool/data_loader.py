import pandas as pd
from sqlalchemy import create_engine, text, Engine, Table, Column, MetaData, select
from typing import Optional, Any, Dict, List
import logging

logger = logging.getLogger(__name__)

def get_max_date_for_table(engine: Engine, table: str, ts_column: str) -> Optional[pd.Timestamp]:
    """
    Retrieves the maximum timestamp from a specified table and timestamp column.

    Args:
        engine: SQLAlchemy engine for the database connection.
        table: The name of the table to query.
        ts_column: The name of the timestamp column.

    Returns:
        A pandas Timestamp representing the maximum date, or None if the table is empty or an error occurs.
    """
    try:
        metadata = MetaData()
        tbl = Table(table, metadata, Column(ts_column))
        stmt = select(tbl.c[ts_column]).order_by(tbl.c[ts_column].desc()).limit(1)
        
        logger.info(f"Executing query to get max date: {stmt}")
        with engine.connect() as connection:
            result = connection.execute(stmt).scalar_one_or_none()
            if result:
                return pd.to_datetime(result)
            return None
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
    regressors: Optional[list[str]] = None,
    resample_to_freq: Optional[str] = None,
    columns_to_load: Optional[list[str]] = None, # New parameter
) -> pd.DataFrame:
    """
    Loads time series data from a PostgreSQL table.

    Args:
        engine: SQLAlchemy engine for the database connection.
        table: The name of the table to load data from.
        ts_column: The name of the timestamp column in the table.
        y_column: The name of the target value column in the table.
        start: Optional start timestamp to filter data.
        end: Optional end timestamp to filter data.
        filters: Optional dictionary of additional column filters (e.g., {"queue": "sales"}).
        regressors: Optional list of column names to include as regressors.
        resample_to_freq: Optional frequency string (e.g., 'H', 'D', 'W') to resample the data to.
                          'y' is summed, regressors are set to their 'first' value in the interval.
        columns_to_load: Optional list of specific column names to load. If None, loads ts_column, y_column, and regressors.

    Returns:
        A pandas DataFrame with 'ds' (datetime), 'y' (numeric), and any additional specified columns.
    """
    if not isinstance(engine, Engine):
        raise TypeError("The 'engine' argument must be an SQLAlchemy Engine instance.")

    try:
        metadata = MetaData()
        
        # Determine which columns to select
        if columns_to_load:
            # If specific columns are requested, load them all as-is (except for renaming ts_column to ds)
            tbl_cols = [Column(col_name) for col_name in columns_to_load]
            tbl = Table(table, metadata, *tbl_cols)

            # Build selection with proper table column references
            select_cols = []
            for col_name in columns_to_load:
                if col_name == ts_column:
                    select_cols.append(tbl.c[col_name].label('ds'))
                else:
                    select_cols.append(tbl.c[col_name])

            stmt = select(*select_cols).select_from(tbl)
        else:
            # Original logic: select ts_column (as ds), y_column (as y), and any regressors/filter columns
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

            selection = [tbl.c[ts_column].label('ds'), tbl.c[y_column].label('y')]
            
            if regressors:
                for col_name in regressors:
                    selection.append(tbl.c[col_name])

            stmt = select(*selection)

        if start:
            stmt = stmt.where(tbl.c[ts_column] >= start)
        if end:
            stmt = stmt.where(tbl.c[ts_column] <= end)
        
        if filters:
            for key, value in filters.items():
                # For filtering, always use the raw column name
                stmt = stmt.where(tbl.c[key] == value)

        stmt = stmt.order_by(tbl.c[ts_column])
        
        # Compile statement for logging (shows bound params as ?)
        logger.info(f"Executing query: {stmt}")

        df = pd.read_sql(stmt, engine)
        
        # Ensure 'ds' is datetime, handling cases where it might not be directly selected as 'ds'
        # if 'ds' is not already a datetime column, convert it.
        if ts_column in df.columns and ts_column != 'ds':
            df = df.rename(columns={ts_column: 'ds'})
        df['ds'] = pd.to_datetime(df['ds']) # Ensure 'ds' is datetime
        
        if resample_to_freq:
            logger.info(f"Resampling data to {resample_to_freq}...")
            df = df.set_index('ds')
            
            agg_dict = {'y': 'sum'}
            if columns_to_load: # If specific columns were loaded, try to aggregate them
                 for col in columns_to_load:
                     if col != ts_column and col != y_column and col in df.columns:
                         # Default to 'first' for non-y columns if no specific agg is known
                         if pd.api.types.is_numeric_dtype(df[col]):
                             agg_dict[col] = 'sum'
                         else:
                             agg_dict[col] = 'first'
            elif regressors: # If regressors were loaded
                for reg in regressors:
                    if reg in df.columns:
                        if pd.api.types.is_numeric_dtype(df[reg]):
                             agg_dict[reg] = 'sum'
                        else:
                             agg_dict[reg] = 'first'

            df = df.resample(resample_to_freq).agg(agg_dict).reset_index()
            df.dropna(subset=['y'], inplace=True) 
            logger.info(f"Resampled to {resample_to_freq}, resulting in {len(df)} rows.")

        logger.info(f"Loaded {len(df)} rows from {table}.")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {table}: {e}")
        raise

if __name__ == "__main__":
    # Example usage (requires PostgreSQL to be running and connection details in .env)
    from dotenv import load_dotenv
    import os

    load_dotenv()

    # Database connection details from environment variables
    PGHOST = os.getenv("PGHOST", "localhost")
    PGPORT = os.getenv("PGPORT", "5432")
    PGUSER = os.getenv("PGUSER", "user")
    PGPASSWORD = os.getenv("PGPASSWORD", "password")
    PGDATABASE = os.getenv("PGDATABASE", "prophet_forecast_db")

    DATABASE_URL = f"postgresql+psycopg2://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}"
    print(f"Attempting to connect to: {DATABASE_URL}")

    try:
        engine = create_engine(DATABASE_URL)
        # Example: Load data for a specific queue
        df = load_time_series(engine, filters={"queue": "support"})
        print(f"Successfully loaded data. Head:\n{df.head()}")
    except Exception as e:
        print(f"Failed to load data: {e}")
