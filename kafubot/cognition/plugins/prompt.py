from __future__ import annotations

from typing import Any

from kafubot.cognition.plugins.base import PluginContext, PluginDefinition


def apply(_context: PluginContext, _config: Any) -> None:
    """Enable the Main Executive system prompt."""


plugin = PluginDefinition(name="prompt", apply=apply)
