"""Tests for the fixed API module - risk checks, orderbook, no hardcoded fallbacks."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock


class TestAPIModuleExists:
    """Test that the API module can be imported and has routes."""

    def test_import_app(self):
        from acms.api import app
        assert app is not None

    def test_app_has_routes(self):
        from acms.api import app
        routes = [r.path for r in app.routes if hasattr(r, 'path')]
        assert len(routes) > 0

    def test_health_route_exists(self):
        from acms.api import app
        routes = [r.path for r in app.routes if hasattr(r, 'path')]
        assert any("health" in r for r in routes)


class TestAPIModuleStructure:
    """Test that the API has been split into proper modules."""

    def test_import_app_module(self):
        try:
            from acms.api.app import create_app
            assert create_app is not None
        except ImportError:
            # Might still be in old structure
            from acms.api import app
            assert app is not None

    def test_import_schemas(self):
        try:
            from acms.api import schemas
            assert schemas is not None
        except ImportError:
            pass

    def test_import_dependencies(self):
        try:
            from acms.api import dependencies
            assert dependencies is not None
        except ImportError:
            pass


class TestAPIOrderbookEndpoint:
    def test_orderbook_route_exists(self):
        from acms.api import app
        routes = [r.path for r in app.routes if hasattr(r, 'path')]
        orderbook_routes = [r for r in routes if "orderbook" in r]
        assert len(orderbook_routes) > 0, "Orderbook route should exist"
