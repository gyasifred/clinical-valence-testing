"""
Logging configuration for Clinical Valence Testing.

This module provides centralized logging configuration and utilities
for consistent logging across the project.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output."""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }

    def format(self, record):
        """Format log record with color."""
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{log_color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True,
    log_format: Optional[str] = None,
    colored: bool = True
) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        console: Whether to log to console
        log_format: Custom log format string
        colored: Whether to use colored console output

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("clinical_valence_testing")
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers = []

    # Default format
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))

        if colored and sys.stdout.isatty():
            console_formatter = ColoredFormatter(log_format)
        else:
            console_formatter = logging.Formatter(log_format)

        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the project namespace.

    Args:
        name: Logger name (typically __name__ of the module)

    Returns:
        Logger instance
    """
    return logging.getLogger(f"clinical_valence_testing.{name}")


class LoggerContext:
    """Context manager for temporary logging configuration changes."""

    def __init__(self, logger: logging.Logger, level: str):
        """
        Initialize logger context.

        Args:
            logger: Logger instance to modify
            level: Temporary logging level
        """
        self.logger = logger
        self.level = getattr(logging, level.upper())
        self.original_level = None

    def __enter__(self):
        """Set temporary logging level."""
        self.original_level = self.logger.level
        self.logger.setLevel(self.level)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original logging level."""
        self.logger.setLevel(self.original_level)


def log_experiment_info(
    logger: logging.Logger,
    config: dict,
    dataset_info: Optional[dict] = None
):
    """
    Log experiment configuration and dataset information.

    Args:
        logger: Logger instance
        config: Configuration dictionary
        dataset_info: Dataset information dictionary (optional)
    """
    logger.info("=" * 80)
    logger.info("EXPERIMENT CONFIGURATION")
    logger.info("=" * 80)

    # Log timestamp
    logger.info(f"Timestamp: {datetime.now().isoformat()}")

    # Log configuration
    if config:
        logger.info("\nConfiguration:")
        for key, value in config.items():
            logger.info(f"  {key}: {value}")

    # Log dataset info
    if dataset_info:
        logger.info("\nDataset Information:")
        for key, value in dataset_info.items():
            logger.info(f"  {key}: {value}")

    logger.info("=" * 80)


def log_results_summary(
    logger: logging.Logger,
    results: dict,
    title: str = "RESULTS SUMMARY"
):
    """
    Log results summary in a formatted manner.

    Args:
        logger: Logger instance
        results: Results dictionary
        title: Title for the summary section
    """
    logger.info("=" * 80)
    logger.info(title)
    logger.info("=" * 80)

    for key, value in results.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.6f}")
        else:
            logger.info(f"  {key}: {value}")

    logger.info("=" * 80)


class ProgressLogger:
    """Logger for tracking progress of long-running operations."""

    def __init__(
        self,
        logger: logging.Logger,
        total: int,
        prefix: str = "Progress",
        log_interval: int = 10
    ):
        """
        Initialize progress logger.

        Args:
            logger: Logger instance
            total: Total number of items
            prefix: Prefix for log messages
            log_interval: Log every N percent
        """
        self.logger = logger
        self.total = total
        self.prefix = prefix
        self.log_interval = log_interval
        self.current = 0
        self.last_logged_percent = 0

    def update(self, n: int = 1):
        """
        Update progress.

        Args:
            n: Number of items processed
        """
        self.current += n
        percent = int((self.current / self.total) * 100)

        if percent >= self.last_logged_percent + self.log_interval or percent == 100:
            self.logger.info(
                f"{self.prefix}: {self.current}/{self.total} ({percent}%)"
            )
            self.last_logged_percent = percent

    def finish(self):
        """Log completion message."""
        self.logger.info(f"{self.prefix}: Completed {self.total}/{self.total} (100%)")
