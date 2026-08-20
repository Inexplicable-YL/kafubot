from __future__ import annotations

from typing import Any

from kafubot.cognition.plugins.base import PluginContext, PluginDefinition


def apply(_context: PluginContext, _config: Any) -> None:
    """Expose the configured display clock to the Main Executive."""


plugin = PluginDefinition(name="clock", apply=apply)
