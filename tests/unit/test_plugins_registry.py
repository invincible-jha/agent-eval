"""Unit tests for agent_eval.plugins.registry.

Tests PluginRegistry registration, lookup, deregistration,
entry-point loading, and error cases.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from unittest.mock import MagicMock, patch

import pytest

from agent_eval.plugins.registry import (
    PluginAlreadyRegisteredError,
    PluginNotFoundError,
    PluginRegistry,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class BasePlugin(ABC):
    @abstractmethod
    def process(self) -> str: ...


class ConcretePlugin(BasePlugin):
    def process(self) -> str:
        return "concrete"


class AnotherPlugin(BasePlugin):
    def process(self) -> str:
        return "another"


def _fresh_registry(name: str = "test-registry") -> PluginRegistry[BasePlugin]:
    return PluginRegistry(BasePlugin, name)


# ---------------------------------------------------------------------------
# PluginNotFoundError and PluginAlreadyRegisteredError
# ---------------------------------------------------------------------------


class TestPluginErrors:
    def test_not_found_error_attributes(self) -> None:
        err = PluginNotFoundError("my-plugin", "my-registry")
        assert err.plugin_name == "my-plugin"
        assert err.registry_name == "my-registry"
        assert "my-plugin" in str(err)

    def test_already_registered_error_attributes(self) -> None:
        err = PluginAlreadyRegisteredError("my-plugin", "my-registry")
        assert err.plugin_name == "my-plugin"
        assert err.registry_name == "my-registry"
        assert "my-plugin" in str(err)


# ---------------------------------------------------------------------------
# PluginRegistry — registration
# ---------------------------------------------------------------------------


class TestPluginRegistryRegister:
    def test_register_decorator_returns_class_unchanged(self) -> None:
        registry = _fresh_registry()

        @registry.register("concrete")
        class MyPlugin(BasePlugin):
            def process(self) -> str:
                return "ok"

        assert MyPlugin is not None
        assert MyPlugin().process() == "ok"

    def test_registered_plugin_retrievable(self) -> None:
        registry = _fresh_registry()

        @registry.register("my-plugin")
        class MyPlugin(BasePlugin):
            def process(self) -> str:
                return "result"

        cls = registry.get("my-plugin")
        assert cls is MyPlugin

    def test_duplicate_name_raises_already_registered(self) -> None:
        registry = _fresh_registry()
        registry.register("dup")(ConcretePlugin)
        with pytest.raises(PluginAlreadyRegisteredError):
            registry.register("dup")(AnotherPlugin)

    def test_non_subclass_raises_type_error(self) -> None:
        registry = _fresh_registry()
        with pytest.raises(TypeError, match="subclass"):
            registry.register("bad")(int)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# PluginRegistry — register_class
# ---------------------------------------------------------------------------


class TestPluginRegistryRegisterClass:
    def test_register_class_directly(self) -> None:
        registry = _fresh_registry()
        registry.register_class("direct", ConcretePlugin)
        assert registry.get("direct") is ConcretePlugin

    def test_duplicate_register_class_raises_error(self) -> None:
        registry = _fresh_registry()
        registry.register_class("dup", ConcretePlugin)
        with pytest.raises(PluginAlreadyRegisteredError):
            registry.register_class("dup", AnotherPlugin)

    def test_non_subclass_register_class_raises_type_error(self) -> None:
        registry = _fresh_registry()
        with pytest.raises(TypeError):
            registry.register_class("bad", str)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# PluginRegistry — deregister
# ---------------------------------------------------------------------------


class TestPluginRegistryDeregister:
    def test_deregister_removes_plugin(self) -> None:
        registry = _fresh_registry()
        registry.register_class("to-remove", ConcretePlugin)
        registry.deregister("to-remove")
        assert "to-remove" not in registry

    def test_deregister_missing_raises_not_found(self) -> None:
        registry = _fresh_registry()
        with pytest.raises(PluginNotFoundError):
            registry.deregister("nonexistent")

    def test_deregistered_plugin_not_in_list(self) -> None:
        registry = _fresh_registry()
        registry.register_class("p1", ConcretePlugin)
        registry.deregister("p1")
        assert "p1" not in registry.list_plugins()


# ---------------------------------------------------------------------------
# PluginRegistry — lookup
# ---------------------------------------------------------------------------


class TestPluginRegistryGet:
    def test_get_existing_plugin(self) -> None:
        registry = _fresh_registry()
        registry.register_class("p", ConcretePlugin)
        assert registry.get("p") is ConcretePlugin

    def test_get_missing_raises_not_found(self) -> None:
        registry = _fresh_registry()
        with pytest.raises(PluginNotFoundError, match="my-plugin"):
            registry.get("my-plugin")

    def test_list_plugins_returns_sorted(self) -> None:
        registry = _fresh_registry()
        registry.register_class("zebra", ConcretePlugin)
        registry.register_class("alpha", AnotherPlugin)
        assert registry.list_plugins() == ["alpha", "zebra"]

    def test_contains_operator(self) -> None:
        registry = _fresh_registry()
        registry.register_class("p", ConcretePlugin)
        assert "p" in registry
        assert "missing" not in registry

    def test_len_returns_count(self) -> None:
        registry = _fresh_registry()
        assert len(registry) == 0
        registry.register_class("p1", ConcretePlugin)
        assert len(registry) == 1
        registry.register_class("p2", AnotherPlugin)
        assert len(registry) == 2

    def test_repr_includes_name_and_plugins(self) -> None:
        registry = _fresh_registry("test-reg")
        registry.register_class("plug", ConcretePlugin)
        r = repr(registry)
        assert "test-reg" in r
        assert "plug" in r


# ---------------------------------------------------------------------------
# PluginRegistry — load_entrypoints
# ---------------------------------------------------------------------------


class TestPluginRegistryLoadEntrypoints:
    def test_load_entrypoints_registers_valid_plugin(self) -> None:
        registry = _fresh_registry()
        mock_ep = MagicMock()
        mock_ep.name = "ep-plugin"
        mock_ep.load.return_value = ConcretePlugin

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            registry.load_entrypoints("agent_eval.plugins")

        assert "ep-plugin" in registry

    def test_load_entrypoints_skips_already_registered(self) -> None:
        registry = _fresh_registry()
        registry.register_class("existing", ConcretePlugin)

        mock_ep = MagicMock()
        mock_ep.name = "existing"
        mock_ep.load.return_value = ConcretePlugin

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            registry.load_entrypoints("agent_eval.plugins")

        # Should not raise, and existing should still be there
        assert registry.get("existing") is ConcretePlugin

    def test_load_entrypoints_skips_failed_load(self) -> None:
        registry = _fresh_registry()
        mock_ep = MagicMock()
        mock_ep.name = "broken-plugin"
        mock_ep.load.side_effect = ImportError("cannot import")

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            registry.load_entrypoints("agent_eval.plugins")

        # Should not raise, broken plugin not registered
        assert "broken-plugin" not in registry

    def test_load_entrypoints_skips_non_subclass(self) -> None:
        registry = _fresh_registry()
        mock_ep = MagicMock()
        mock_ep.name = "wrong-type"
        mock_ep.load.return_value = str  # Not a subclass of BasePlugin

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            registry.load_entrypoints("agent_eval.plugins")

        assert "wrong-type" not in registry

    def test_load_entrypoints_idempotent(self) -> None:
        registry = _fresh_registry()
        mock_ep = MagicMock()
        mock_ep.name = "idempotent"
        mock_ep.load.return_value = ConcretePlugin

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            registry.load_entrypoints("agent_eval.plugins")
            registry.load_entrypoints("agent_eval.plugins")

        assert len(registry) == 1
