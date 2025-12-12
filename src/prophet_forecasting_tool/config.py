from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL
import os

class Settings(BaseSettings):
    """
    Project settings loaded from environment variables or .env file.
    """
    PGHOST: str = "localhost"
    PGPORT: int = 5432
    PGUSER: str = "user"
    PGPASSWORD: str = "password"
    PGDATABASE: str = "prophet_forecast_db"

    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    @property
    def DATABASE_URL(self) -> str:
        """
        Constructs the SQLAlchemy database URL from settings.
        """
        return str(URL.create(
            drivername="postgresql+psycopg2",
            username=self.PGUSER,
            password=self.PGPASSWORD,
            host=self.PGHOST,
            port=self.PGPORT,
            database=self.PGDATABASE,
        ))

# Example usage (for testing purposes, not for production direct use)
if __name__ == "__main__":
    # Create a dummy .env file for testing
    with open(".env", "w") as f:
        f.write("PGHOST=test_host\n")
        f.write("PGUSER=test_user\n")
        f.write("PGPASSWORD=test_pass\n")
        f.write("PGDATABASE=test_db\n")

    settings = Settings()
    print(f"PGHOST: {settings.PGHOST}")
    print(f"PGUSER: {settings.PGUSER}")
    print(f"DATABASE_URL: {settings.DATABASE_URL}")

    # Clean up dummy .env file
    os.remove(".env")
