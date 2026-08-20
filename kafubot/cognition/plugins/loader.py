from __future__ import annotations

from importlib import import_module
from importlib.metadata import EntryPoint, entry_points
from pkgutil import iter_modules
from types import ModuleType

from .base import PluginCatalog, PluginDefinition

PLUGIN_PACKAGE = "kafubot.cognition.plugins"
PLUGIN_ENTRY_POINT_GROUP = "kafubot.plugins"


def _module_plugin(module: ModuleType, source: str) -> PluginDefinition:
    try:
        definition = module.plugin
    except AttributeError as exc:
        raise TypeError(
            f"plugin module {source} must export `plugin: PluginDefinition`"
        ) from exc
    if not isinstance(definition, PluginDefinition):
        raise TypeError(
            f"plugin module {source} must export `plugin: PluginDefinition`"
        )
    return definition


def _entry_point_plugin(entry_point: EntryPoint) -> PluginDefinition:
    loaded = entry_point.load()
    if not isinstance(loaded, ModuleType):
        raise TypeError(
            f"entry point {entry_point.name} must resolve to a plugin module or package"
        )
    return _module_plugin(loaded, entry_point.value)


def discover_plugins(
    package_name: str = PLUGIN_PACKAGE,
    *,
    include_entry_points: bool = True,
) -> PluginCatalog:
    """Discover direct child files and packages that explicitly export ``plugin``.

    A single-file plugin owns its definition in that file. A package plugin owns it
    in ``__init__.py``; its private implementation modules are never scanned. Direct
    children without a ``plugin`` export are ordinary framework modules and skipped.
    """
    package = import_module(package_name)
    package_path = getattr(package, "__path__", None)
    if package_path is None:
        raise TypeError(f"plugin package has no discovery path: {package_name}")

    catalog = PluginCatalog()
    modules = sorted(
        (
            item
            for item in iter_modules(package_path, f"{package_name}.")
            if not item.name.rsplit(".", 1)[-1].startswith("_")
        ),
        key=lambda item: item.name,
    )
    for module_info in modules:
        module = import_module(module_info.name)
        if not hasattr(module, "plugin"):
            continue
        catalog.register(_module_plugin(module, module_info.name))

    if include_entry_points:
        discovered = entry_points(group=PLUGIN_ENTRY_POINT_GROUP)
        for entry_point in sorted(discovered, key=lambda item: (item.name, item.value)):
            catalog.register(_entry_point_plugin(entry_point))
    return catalog


__all__ = [
    "PLUGIN_ENTRY_POINT_GROUP",
    "PLUGIN_PACKAGE",
    "discover_plugins",
]
