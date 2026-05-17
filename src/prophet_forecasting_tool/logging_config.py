"""Logging setup with rotation and absolute paths."""
from __future__ import annotations

import logging
import os
import pathlib
from logging.handlers import RotatingFileHandler


def setup_logging(level: str = "INFO", log_dir: str | os.PathLike | None = None) -> None:
    """Configure root logging once. Re-callable safely."""
    log_level = os.environ.get("LOG_LEVEL", level).upper()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove any handlers we previously installed (idempotent re-setup).
    for h in list(root_logger.handlers):
        if getattr(h, "_pf_managed", False):
            root_logger.removeHandler(h)

    has_stream = any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers)
    if not has_stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        setattr(stream_handler, "_pf_managed", True)
        root_logger.addHandler(stream_handler)

    log_dir_path = pathlib.Path(log_dir or os.getenv("LOG_DIR") or pathlib.Path.cwd() / "logs")
    try:
        log_dir_path.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir_path / "app.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        setattr(file_handler, "_pf_managed", True)
        root_logger.addHandler(file_handler)
    except OSError as e:
        root_logger.warning(f"Could not create file log handler at {log_dir_path}: {e}")

    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    logging.getLogger("prophet").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("pydantic_settings").setLevel(logging.WARNING)
