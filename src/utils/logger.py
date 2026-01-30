"""Logging configuration using loguru."""

import sys
from pathlib import Path
from loguru import logger

from .constants import LOGS_DIR, APP_NAME

def setup_logger(debug: bool = False):
    """Configure application logging.

    Args:
        debug: Enable debug level logging
    """
    # Remove default handler
    logger.remove()

    # Console handler with colors
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    level = "DEBUG" if debug else "INFO"

    # Console handler (only if stderr is available - not in windowed mode)
    if sys.stderr is not None:
        logger.add(
            sys.stderr,
            format=log_format,
            level=level,
            colorize=True,
        )

    # File handler for errors (only if LOGS_DIR exists)
    try:
        if LOGS_DIR and LOGS_DIR.exists():
            log_file = LOGS_DIR / f"{APP_NAME.lower().replace(' ', '_')}.log"
            logger.add(
                log_file,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
                level="WARNING",
                rotation="10 MB",
                retention="7 days",
                compression="zip",
            )

            # Debug log file (if debug mode)
            if debug:
                debug_log = LOGS_DIR / f"{APP_NAME.lower().replace(' ', '_')}_debug.log"
                logger.add(
                    debug_log,
                    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
                    level="DEBUG",
                    rotation="50 MB",
                    retention="3 days",
                )
    except Exception:
        pass  # Skip file logging if not available

    logger.info(f"{APP_NAME} logger initialized")
    return logger


# Export pre-configured logger
__all__ = ["logger", "setup_logger"]
