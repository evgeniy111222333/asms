"""Logging Configuration - Structured logging with Loguru.

Implements:
- Loguru configuration with structured logging
- Multiple sinks: console, file, rotating file
- JSON logging for production
- Correlation ID tracking
- Performance logging decorators
"""

import functools
import time
import uuid
import json
import sys
import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Callable


# ============================================================================
# Loguru Configuration
# ============================================================================

def configure_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "/data/acms/logs",
    json_format: bool = False,
    rotation: str = "100 MB",
    retention: str = "30 days",
    correlation_id: Optional[str] = None,
) -> None:
    """Configure Loguru logging with multiple sinks.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional specific log file path.
        log_dir: Directory for log files.
        json_format: Use JSON format for production logging.
        rotation: Log rotation size.
        retention: Log retention period.
        correlation_id: Optional correlation ID for request tracking.
    """
    try:
        from loguru import logger

        # Remove default handler
        logger.remove()

        # Console sink with color
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "{extra[correlation_id]} | "
            "<level>{message}</level>"
        )

        json_console_format = _json_formatter

        logger.add(
            sys.stderr,
            format=console_format if not json_format else json_console_format,
            level=level,
            colorize=not json_format,
        )

        # File sink
        if log_file or log_dir:
            log_path = Path(log_file) if log_file else Path(log_dir) / "acms.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_format = (
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
                "{name}:{function}:{line} | {extra[correlation_id]} | {message}"
            )

            logger.add(
                str(log_path),
                format=file_format if not json_format else json_console_format,
                level=level,
                rotation=rotation,
                retention=retention,
                compression="gz",
            )

            # Error-only log file
            error_path = log_path.parent / "acms_error.log"
            logger.add(
                str(error_path),
                format=file_format,
                level="ERROR",
                rotation=rotation,
                retention=retention,
                compression="gz",
                filter=lambda record: record["level"].name == "ERROR",
            )

        # Set default correlation ID
        default_cid = correlation_id or "-"
        logger.configure(extra={"correlation_id": default_cid})

        # Intercept standard logging
        _intercept_standard_logging(level)

        logger.info("Logging configured: level={}, json={}, file={}",
                     level, json_format, log_file or log_dir)

    except ImportError:
        # Fallback to standard logging
        logging.basicConfig(
            level=getattr(logging, level, logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
        logging.warning("loguru not installed, using standard logging")


def _json_formatter(record: Dict) -> str:
    """Format log record as JSON for production.

    Args:
        record: Loguru record dict.

    Returns:
        JSON formatted log string.
    """
    log_entry = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "message": record["message"],
        "logger": record["name"],
        "function": record["function"],
        "line": record["line"],
        "correlation_id": record["extra"].get("correlation_id", "-"),
    }

    # Include exception info if present
    if record["exception"]:
        log_entry["exception"] = {
            "type": str(record["exception"].type.__name__) if record["exception"].type else None,
            "value": str(record["exception"].value) if record["exception"].value else None,
            "traceback": record["exception"].traceback if record["exception"].traceback else None,
        }

    # Include extra fields
    for key, value in record["extra"].items():
        if key not in ("correlation_id",):
            log_entry[key] = str(value)

    return json.dumps(log_entry, default=str) + "\n"


def _intercept_standard_logging(level: str = "INFO") -> None:
    """Intercept standard logging messages and route through Loguru.

    Args:
        level: Minimum logging level to intercept.
    """
    try:
        from loguru import logger

        class InterceptHandler(logging.Handler):
            def emit(self, record):
                try:
                    level = logger.level(record.levelname).name
                except ValueError:
                    level = record.levelno
                frame, depth = logging.currentframe(), 2
                while frame and frame.f_code.co_filename == logging.__file__:
                    frame = frame.f_back
                    depth += 1
                logger.opt(depth=depth, exception=record.exc_info).log(
                    level, record.getMessage()
                )

        logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
        for name in logging.root.manager.loggerDict.keys():
            logging.getLogger(name).handlers = []
            logging.getLogger(name).propagate = True

    except ImportError:
        logger = logging.getLogger(__name__)
        logger.debug("loguru not available, using standard logging")


# ============================================================================
# Correlation ID Tracking
# ============================================================================

class CorrelationID:
    """Thread-local correlation ID tracking for request tracing."""

    _current_id: Optional[str] = None

    @classmethod
    def get(cls) -> str:
        """Get the current correlation ID."""
        return cls._current_id or "-"

    @classmethod
    def set(cls, cid: Optional[str] = None) -> str:
        """Set a new correlation ID.

        Args:
            cid: Optional specific ID. Auto-generated if not provided.

        Returns:
            The correlation ID that was set.
        """
        cls._current_id = cid or str(uuid.uuid4())[:8]
        return cls._current_id

    @classmethod
    def clear(cls) -> None:
        """Clear the current correlation ID."""
        cls._current_id = None


@contextmanager
def correlation_context(cid: Optional[str] = None):
    """Context manager for correlation ID scoping.

    Args:
        cid: Optional specific correlation ID.

    Yields:
        The correlation ID for this context.

    Example::

        with correlation_context("req-123") as cid:
            logger.info("Processing request")  # Includes correlation ID
    """
    old_id = CorrelationID._current_id
    CorrelationID.set(cid)
    try:
        yield CorrelationID.get()
    finally:
        CorrelationID._current_id = old_id


# ============================================================================
# Performance Logging Decorators
# ============================================================================

def log_performance(func: Optional[Callable] = None, *, threshold_ms: float = 100.0,
                     log_args: bool = False) -> Callable:
    """Decorator for logging function execution performance.

    Args:
        func: Function to decorate (when used without arguments).
        threshold_ms: Minimum duration to log (in milliseconds).
        log_args: Whether to log function arguments.

    Returns:
        Decorated function.

    Example::

        @log_performance(threshold_ms=50.0)
        def compute_indicators(data):
            ...
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                duration_ms = (time.perf_counter() - start) * 1000
                if duration_ms >= threshold_ms:
                    _log_performance(fn.__name__, duration_ms, log_args, args, kwargs, None)
                return result
            except Exception as e:
                duration_ms = (time.perf_counter() - start) * 1000
                _log_performance(fn.__name__, duration_ms, log_args, args, kwargs, e)
                raise

        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = await fn(*args, **kwargs)
                duration_ms = (time.perf_counter() - start) * 1000
                if duration_ms >= threshold_ms:
                    _log_performance(fn.__name__, duration_ms, log_args, args, kwargs, None)
                return result
            except Exception as e:
                duration_ms = (time.perf_counter() - start) * 1000
                _log_performance(fn.__name__, duration_ms, log_args, args, kwargs, e)
                raise

        if asyncio.iscoroutinefunction(fn):
            return async_wrapper
        return sync_wrapper

    if func is not None:
        return decorator(func)
    return decorator


def _log_performance(func_name: str, duration_ms: float, log_args: bool,
                     args: tuple, kwargs: dict, error: Optional[Exception]) -> None:
    """Log a performance measurement.

    Args:
        func_name: Function name.
        duration_ms: Execution duration in milliseconds.
        log_args: Whether to log arguments.
        args: Positional arguments.
        kwargs: Keyword arguments.
        error: Optional exception that occurred.
    """
    try:
        from loguru import logger
    except ImportError:
        logger = logging.getLogger(__name__)

    msg = f"PERF: {func_name} took {duration_ms:.2f}ms"
    if log_args:
        msg += f" | args={args[:3]}... kwargs={list(kwargs.keys())}"
    if error:
        msg += f" | error={type(error).__name__}"

    if error:
        logger.warning(msg)
    else:
        logger.info(msg)


# Need asyncio import for iscoroutinefunction check
import asyncio
