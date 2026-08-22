from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from kafubot.cognition.plugins.base import (
    PluginContext,
    PluginDefinition,
    PluginHost,
)

from .attention import AttentionScheduler
from .providers import (
    ConversationWorldProvider,
    ProviderCommit,
    ProviderQuery,
    WorldModelHub,
    WorldModelProvider,
)
from .state import SelfStateStore

if TYPE_CHECKING:
    from kafubot.cognition.plugins.environment import SocialEnvironment
    from kafubot.config import AgentConfig


def apply(context: PluginContext, _config: Any) -> None:
    config = cast("AgentConfig", context.service("config"))
    environment = cast("SocialEnvironment", context.service("environment"))
    if context.optional_service("state_store") is None:
        context.provide(
            "state_store",
            SelfStateStore(
                config.executive.state_file,
                recent_action_limit=config.executive.recent_action_limit,
            ),
        )
    context.provide("attention", AttentionScheduler(config.attention))
    context.provider(ConversationWorldProvider(environment))

    def assemble() -> None:
        host = cast("PluginHost", context.service("plugin_host"))
        context.provide("world_model", WorldModelHub([*context.providers(), host]))

    context.ready(assemble)


plugin = PluginDefinition(
    name="world_model",
    apply=apply,
    requires=("environment",),
)


__all__ = [
    "AttentionScheduler",
    "ConversationWorldProvider",
    "plugin",
    "ProviderCommit",
    "ProviderQuery",
    "SelfStateStore",
    "WorldModelHub",
    "WorldModelProvider",
]
