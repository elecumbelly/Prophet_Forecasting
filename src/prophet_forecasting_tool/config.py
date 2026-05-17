"""Project settings loaded from environment variables (.env supported)."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL


class Settings(BaseSettings):
    PGHOST: str = "localhost"
    PGPORT: int = 5432
    PGUSER: str = "user"
    PGPASSWORD: str = "password"
    PGDATABASE: str = "prophet_forecast_db"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def DATABASE_URL(self) -> str:
        return str(
            URL.create(
                drivername="postgresql+psycopg2",
                username=self.PGUSER,
                password=self.PGPASSWORD,
                host=self.PGHOST,
                port=self.PGPORT,
                database=self.PGDATABASE,
            )
        )
