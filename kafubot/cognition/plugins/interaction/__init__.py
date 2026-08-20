from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from kafubot.cognition.plugins.base import (
    PluginContext,
    PluginDefinition,
    PluginHost,
    ToolScope,
)

from .compiler import ContextCompiler
from .executive import ExecutiveMiddleware

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain.agents.middleware import AgentMiddleware

    from kafubot.cognition.plugins.environment import SocialEnvironment
    from kafubot.cognition.plugins.replyer import Replyer
    from kafubot.cognition.plugins.self_state import SelfStateStore
    from kafubot.cognition.plugins.world_model import WorldModelHub
    from kafubot.config import AgentConfig


def apply(context: PluginContext, _config: Any) -> None:
    def assemble() -> None:
        custom = cast(
            "Sequence[AgentMiddleware[Any, Any]] | None",
            context.optional_service("custom_middleware"),
        )
        if custom is not None:
            return
        config = cast("AgentConfig", context.service("config"))
        environment = cast("SocialEnvironment", context.service("environment"))
        providers = cast("WorldModelHub", context.service("world_model"))
        replyer = cast("Replyer", context.service("replyer"))
        state_store = cast("SelfStateStore", context.service("state_store"))
        host = cast("PluginHost", context.service("plugin_host"))
        compiler = ContextCompiler(environment, providers, config.executive)
        context.provide("context_compiler", compiler)
        context.middleware(
            ExecutiveMiddleware(
                environment,
                compiler,
                replyer,
                providers,
                state_store,
                config.executive,
                host,
                home_tools=context.tools_for(ToolScope.HOME),
                open_tools=context.tools_for(ToolScope.OPEN),
                prompt_enabled=context.enabled("prompt"),
                clock_enabled=context.enabled("clock"),
            )
        )

    context.ready(assemble)


plugin = PluginDefinition(
    name="interaction",
    apply=apply,
    requires=("environment", "model", "replyer", "self_state", "world_model"),
)


__all__ = ["ContextCompiler", "ExecutiveMiddleware", "plugin"]
