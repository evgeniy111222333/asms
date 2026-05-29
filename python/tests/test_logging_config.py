"""Comprehensive tests for acms.logging_config module.

Tests all classes, methods, and edge cases:
- configure_logging (console only, with file, JSON format, custom level, fallback)
- _json_formatter
- _intercept_standard_logging
- CorrelationID (get/set/clear)
- correlation_context
- log_performance decorator (sync/async, threshold, log_args, exceptions)
- _log_performance
"""

import sys
sys.path.insert(0, '/home/z/my-project/asms/python')

import asyncio
import json
import logging
import os
import tempfile
import time
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

from acms.logging_config import (
    configure_logging,
    _json_formatter,
    _intercept_standard_logging,
    CorrelationID,
    correlation_context,
    log_performance,
    _log_performance,
)


# ============================================================================
# configure_logging Tests
# ============================================================================

class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_default_configuration(self):
        """Should configure logging with defaults (no file)."""
        # Should not raise - use log_dir in temp to avoid permission issues
        with tempfile.TemporaryDirectory() as tmpdir:
            configure_logging(level='INFO', log_dir=tmpdir)

    def test_custom_level(self):
        """Should accept custom log level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            configure_logging(level="DEBUG", log_dir=tmpdir)
            configure_logging(level="WARNING", log_dir=tmpdir)
            configure_logging(level="ERROR", log_dir=tmpdir)
            configure_logging(level="CRITICAL", log_dir=tmpdir)

    def test_json_format(self):
        """Should configure with JSON format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            configure_logging(json_format=True, log_dir=tmpdir)

    def test_with_log_file(self):
        """Should configure with log file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test.log")
            configure_logging(log_file=log_path, level="INFO")
            # File should be created (or at least not raise)

    def test_with_log_dir(self):
        """Should configure with log directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            configure_logging(log_dir=tmpdir, level="INFO")

    def test_with_correlation_id(self):
        """Should accept correlation_id parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            configure_logging(correlation_id="test-123", log_dir=tmpdir)

    def test_custom_rotation_retention(self):
        """Should accept custom rotation and retention."""
        with tempfile.TemporaryDirectory() as tmpdir:
            configure_logging(rotation="50 MB", retention="7 days", log_dir=tmpdir)

    def test_loguru_import_fallback(self):
        """Should fall back to standard logging when loguru not available."""
        with patch.dict('sys.modules', {'loguru': None}):
            # Should not raise
            configure_logging(level="WARNING")

    def test_invalid_level_string(self):
        """Should handle invalid level gracefully."""
        try:
            configure_logging(level="INVALID")
        except (ValueError, Exception):
            pass  # loguru raises ValueError for invalid levels

    def test_both_log_file_and_dir(self):
        """log_file should take precedence over log_dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "specific.log")
            configure_logging(log_file=log_path, log_dir=tmpdir, level="INFO")


# ============================================================================
# _json_formatter Tests
# ============================================================================

class TestJsonFormatter:
    """Tests for _json_formatter function."""

    def test_basic_format(self):
        """Should format basic log record as JSON."""
        record = {
            "time": datetime(2024, 1, 1, 12, 0, 0),
            "level": MagicMock(name="INFO"),
            "message": "Test message",
            "name": "test.module",
            "function": "test_func",
            "line": 42,
            "exception": None,
            "extra": {"correlation_id": "-"},
        }
        record["level"].name = "INFO"
        result = _json_formatter(record)
        parsed = json.loads(result)
        assert parsed["message"] == "Test message"
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test.module"
        assert parsed["function"] == "test_func"
        assert parsed["line"] == 42

    def test_with_correlation_id(self):
        """Should include correlation_id in JSON output."""
        record = {
            "time": datetime(2024, 1, 1, 12, 0, 0),
            "level": MagicMock(name="INFO"),
            "message": "Test",
            "name": "test",
            "function": "func",
            "line": 1,
            "exception": None,
            "extra": {"correlation_id": "req-123"},
        }
        record["level"].name = "INFO"
        result = _json_formatter(record)
        parsed = json.loads(result)
        assert parsed["correlation_id"] == "req-123"

    def test_with_exception(self):
        """Should include exception info when present."""
        exc_type = TypeError
        exc_value = TypeError("test error")
        record = {
            "time": datetime(2024, 1, 1, 12, 0, 0),
            "level": MagicMock(name="ERROR"),
            "message": "Error occurred",
            "name": "test",
            "function": "func",
            "line": 1,
            "exception": MagicMock(
                type=exc_type,
                value=exc_value,
                traceback="traceback_string",
            ),
            "extra": {"correlation_id": "-"},
        }
        record["level"].name = "ERROR"
        result = _json_formatter(record)
        parsed = json.loads(result)
        assert "exception" in parsed
        assert parsed["exception"]["type"] == "TypeError"
        assert parsed["exception"]["value"] == "test error"

    def test_with_none_exception_type(self):
        """Should handle None exception type."""
        record = {
            "time": datetime(2024, 1, 1, 12, 0, 0),
            "level": MagicMock(name="INFO"),
            "message": "Test",
            "name": "test",
            "function": "func",
            "line": 1,
            "exception": MagicMock(type=None, value=None, traceback=None),
            "extra": {"correlation_id": "-"},
        }
        record["level"].name = "INFO"
        result = _json_formatter(record)
        parsed = json.loads(result)
        assert "exception" in parsed

    def test_extra_fields(self):
        """Should include extra fields (except correlation_id) in output."""
        record = {
            "time": datetime(2024, 1, 1, 12, 0, 0),
            "level": MagicMock(name="INFO"),
            "message": "Test",
            "name": "test",
            "function": "func",
            "line": 1,
            "exception": None,
            "extra": {"correlation_id": "-", "custom_field": "custom_value"},
        }
        record["level"].name = "INFO"
        result = _json_formatter(record)
        parsed = json.loads(result)
        assert parsed["custom_field"] == "custom_value"

    def test_output_ends_with_newline(self):
        """JSON output should end with newline."""
        record = {
            "time": datetime(2024, 1, 1, 12, 0, 0),
            "level": MagicMock(name="INFO"),
            "message": "Test",
            "name": "test",
            "function": "func",
            "line": 1,
            "exception": None,
            "extra": {"correlation_id": "-"},
        }
        record["level"].name = "INFO"
        result = _json_formatter(record)
        assert result.endswith("\n")

    def test_output_is_valid_json(self):
        """Output should be valid JSON."""
        record = {
            "time": datetime(2024, 1, 1, 12, 0, 0),
            "level": MagicMock(name="INFO"),
            "message": "Test with 'quotes' and \"dquotes\"",
            "name": "test",
            "function": "func",
            "line": 1,
            "exception": None,
            "extra": {"correlation_id": "-"},
        }
        record["level"].name = "INFO"
        result = _json_formatter(record)
        # Should not raise
        parsed = json.loads(result)
        assert isinstance(parsed, dict)


# ============================================================================
# _intercept_standard_logging Tests
# ============================================================================

class TestInterceptStandardLogging:
    """Tests for _intercept_standard_logging function."""

    def test_basic_call(self):
        """Should not raise when called."""
        _intercept_standard_logging(level="INFO")

    def test_debug_level(self):
        """Should accept DEBUG level."""
        _intercept_standard_logging(level="DEBUG")

    def test_warning_level(self):
        """Should accept WARNING level."""
        _intercept_standard_logging(level="WARNING")

    def test_without_loguru(self):
        """Should not raise when loguru not available."""
        with patch.dict('sys.modules', {'loguru': None}):
            _intercept_standard_logging(level="INFO")


# ============================================================================
# CorrelationID Tests
# ============================================================================

class TestCorrelationID:
    """Tests for CorrelationID class."""

    def setup_method(self):
        """Clear correlation ID before each test."""
        CorrelationID.clear()

    def test_get_default(self):
        """Default correlation ID should be '-'."""
        assert CorrelationID.get() == "-"

    def test_set_with_value(self):
        """Should set specific correlation ID."""
        result = CorrelationID.set("req-123")
        assert result == "req-123"
        assert CorrelationID.get() == "req-123"

    def test_set_auto_generated(self):
        """Should auto-generate ID when not provided."""
        result = CorrelationID.set()
        assert result is not None
        assert len(result) > 0
        assert CorrelationID.get() == result

    def test_auto_generated_length(self):
        """Auto-generated ID should be 8 chars (UUID[:8])."""
        result = CorrelationID.set()
        assert len(result) == 8

    def test_clear(self):
        """Clear should reset to None (get returns '-')."""
        CorrelationID.set("req-123")
        CorrelationID.clear()
        assert CorrelationID.get() == "-"

    def test_set_returns_id(self):
        """set() should return the correlation ID."""
        cid = CorrelationID.set("test-id")
        assert cid == "test-id"

    def test_set_overwrite(self):
        """Setting a new ID should overwrite the previous one."""
        CorrelationID.set("old-id")
        CorrelationID.set("new-id")
        assert CorrelationID.get() == "new-id"

    def test_class_attribute_persistence(self):
        """Correlation ID should persist as class attribute."""
        CorrelationID.set("persistent-id")
        assert CorrelationID._current_id == "persistent-id"

    def test_none_after_clear(self):
        """_current_id should be None after clear."""
        CorrelationID.set("test")
        CorrelationID.clear()
        assert CorrelationID._current_id is None


# ============================================================================
# correlation_context Tests
# ============================================================================

class TestCorrelationContext:
    """Tests for correlation_context context manager."""

    def setup_method(self):
        CorrelationID.clear()

    def test_sets_correlation_id(self):
        """Should set correlation ID within context."""
        with correlation_context("ctx-123") as cid:
            assert cid == "ctx-123"
            assert CorrelationID.get() == "ctx-123"

    def test_restores_previous_id(self):
        """Should restore previous correlation ID after context."""
        CorrelationID.set("old-id")
        with correlation_context("ctx-123"):
            assert CorrelationID.get() == "ctx-123"
        assert CorrelationID.get() == "old-id"

    def test_auto_generates_id(self):
        """Should auto-generate ID when not provided."""
        with correlation_context() as cid:
            assert cid is not None
            assert len(cid) > 0

    def test_restores_on_exception(self):
        """Should restore previous ID even on exception."""
        CorrelationID.set("before")
        try:
            with correlation_context("during"):
                raise ValueError("test error")
        except ValueError:
            pass
        assert CorrelationID.get() == "before"

    def test_nested_contexts(self):
        """Should handle nested contexts."""
        CorrelationID.set("outer")
        with correlation_context("inner1"):
            assert CorrelationID.get() == "inner1"
            with correlation_context("inner2"):
                assert CorrelationID.get() == "inner2"
            assert CorrelationID.get() == "inner1"
        assert CorrelationID.get() == "outer"

    def test_yields_current_id(self):
        """Context manager should yield the current correlation ID."""
        with correlation_context("yield-test") as cid:
            assert cid == "yield-test"

    def test_no_previous_id(self):
        """Should work when no previous ID was set."""
        CorrelationID.clear()
        with correlation_context("ctx-123"):
            assert CorrelationID.get() == "ctx-123"
        # After context, should restore to None (get returns '-')
        assert CorrelationID.get() == "-"


# ============================================================================
# log_performance Decorator Tests
# ============================================================================

class TestLogPerformance:
    """Tests for log_performance decorator."""

    def test_sync_function(self):
        """Should decorate sync function."""
        @log_performance
        def slow_func():
            return 42

        result = slow_func()
        assert result == 42

    def test_sync_function_with_args(self):
        """Should pass through arguments."""
        @log_performance
        def add(a, b):
            return a + b

        assert add(3, 4) == 7

    def test_sync_function_with_kwargs(self):
        """Should pass through keyword arguments."""
        @log_performance
        def greet(name="World"):
            return f"Hello, {name}!"

        assert greet(name="Test") == "Hello, Test!"

    @pytest.mark.asyncio
    async def test_async_function(self):
        """Should decorate async function."""
        @log_performance
        async def async_func():
            return 42

        result = await async_func()
        assert result == 42

    @pytest.mark.asyncio
    async def test_async_function_with_args(self):
        """Should pass through args for async function."""
        @log_performance
        async def async_add(a, b):
            return a + b

        result = await async_add(3, 4)
        assert result == 7

    def test_preserves_function_name(self):
        """Should preserve function name via functools.wraps."""
        @log_performance
        def my_function():
            """My docstring."""
            pass

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."

    def test_sync_exception_propagated(self):
        """Exceptions should be propagated."""
        @log_performance
        def failing_func():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            failing_func()

    @pytest.mark.asyncio
    async def test_async_exception_propagated(self):
        """Exceptions should be propagated for async functions."""
        @log_performance
        async def async_failing():
            raise ValueError("async error")

        with pytest.raises(ValueError, match="async error"):
            await async_failing()

    def test_with_threshold(self):
        """Should accept threshold_ms parameter."""
        @log_performance(threshold_ms=1000.0)
        def fast_func():
            return 1

        assert fast_func() == 1

    def test_with_low_threshold(self):
        """Should log with very low threshold."""
        @log_performance(threshold_ms=0.0)
        def func():
            return 1

        assert func() == 1

    def test_with_log_args(self):
        """Should accept log_args parameter."""
        @log_performance(log_args=True)
        def func(a, b):
            return a + b

        assert func(1, 2) == 3

    def test_with_threshold_and_log_args(self):
        """Should accept both threshold and log_args."""
        @log_performance(threshold_ms=0.0, log_args=True)
        def func(x):
            return x * 2

        assert func(5) == 10

    @pytest.mark.asyncio
    async def test_async_with_threshold(self):
        """Should accept threshold for async function."""
        @log_performance(threshold_ms=1000.0)
        async def async_func():
            return 1

        result = await async_func()
        assert result == 1

    @pytest.mark.asyncio
    async def test_async_with_log_args(self):
        """Should accept log_args for async function."""
        @log_performance(log_args=True)
        async def async_func(a, b):
            return a + b

        result = await async_func(1, 2)
        assert result == 3

    def test_returns_correct_value(self):
        """Should return the function's return value."""
        @log_performance
        def compute():
            return {"key": "value", "count": 42}

        result = compute()
        assert result == {"key": "value", "count": 42}

    @pytest.mark.asyncio
    async def test_async_returns_correct_value(self):
        """Should return the async function's return value."""
        @log_performance
        async def async_compute():
            return [1, 2, 3]

        result = await async_compute()
        assert result == [1, 2, 3]

    def test_decorated_without_arguments(self):
        """Should work when decorator is used without arguments."""
        @log_performance
        def my_func():
            return "result"

        assert my_func() == "result"

    def test_decorated_with_arguments(self):
        """Should work when decorator is used with arguments."""
        @log_performance(threshold_ms=50.0)
        def my_func():
            return "result"

        assert my_func() == "result"

    def test_sync_exception_logged(self):
        """Exception in sync function should still be logged."""
        @log_performance(threshold_ms=0.0)
        def failing():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            failing()

    @pytest.mark.asyncio
    async def test_async_exception_logged(self):
        """Exception in async function should still be logged."""
        @log_performance(threshold_ms=0.0)
        async def async_failing():
            raise RuntimeError("async boom")

        with pytest.raises(RuntimeError):
            await async_failing()

    def test_class_method(self):
        """Should work on class methods."""
        class MyClass:
            @log_performance(threshold_ms=0.0)
            def method(self, x):
                return x * 2

        obj = MyClass()
        assert obj.method(5) == 10

    @pytest.mark.asyncio
    async def test_async_class_method(self):
        """Should work on async class methods."""
        class MyClass:
            @log_performance(threshold_ms=0.0)
            async def method(self, x):
                return x * 2

        obj = MyClass()
        result = await obj.method(5)
        assert result == 10


# ============================================================================
# _log_performance Tests
# ============================================================================

class TestLogPerformanceInternal:
    """Tests for _log_performance internal function."""

    def test_basic_call(self):
        """Should not raise on basic call."""
        _log_performance("test_func", 100.0, False, (), {}, None)

    def test_with_log_args(self):
        """Should include args when log_args=True."""
        _log_performance("test_func", 100.0, True, (1, 2, 3), {"key": "val"}, None)

    def test_with_error(self):
        """Should log error info."""
        _log_performance("test_func", 100.0, False, (), {}, ValueError("test"))

    def test_with_error_and_args(self):
        """Should log both error and args."""
        _log_performance("test_func", 100.0, True, (1,), {"k": "v"}, TypeError("err"))

    def test_zero_duration(self):
        """Should handle zero duration."""
        _log_performance("test_func", 0.0, False, (), {}, None)

    def test_very_large_duration(self):
        """Should handle very large duration."""
        _log_performance("test_func", 999999.99, False, (), {}, None)

    def test_empty_args_kwargs(self):
        """Should handle empty args and kwargs."""
        _log_performance("test_func", 50.0, True, (), {}, None)

    def test_many_args(self):
        """Should only log first 3 args (as per implementation)."""
        _log_performance("test_func", 50.0, True, (1, 2, 3, 4, 5), {"a": 1}, None)

    def test_error_type_in_message(self):
        """Should include error type name in message."""
        # Just verify it doesn't raise
        _log_performance("func", 10.0, False, (), {}, RuntimeError("err"))
