import logging
import os

def setup_logging(level: str = "INFO"):
    """
    Configures basic logging for the application, including a file handler.
    """
    log_level = os.environ.get("LOG_LEVEL", level).upper()

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers to prevent duplicate logs in case of re-calling setup_logging
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Console handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    # File handler
    file_handler = logging.FileHandler('app.log')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Silence chatty loggers if needed
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    logging.getLogger("prophet").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("pydantic_settings").setLevel(logging.WARNING)

if __name__ == "__main__":
    setup_logging("DEBUG")
    logger = logging.getLogger(__name__)
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")