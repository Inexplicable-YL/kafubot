from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

from pydantic import BaseModel, ConfigDict

from kafubot.cognition.models import get_thinking_model
from kafubot.cognition.plugins.base import (
    PluginContext,
    PluginDefinition,
    PluginHost,
    ToolScope,
)

from .compiler import ContextCompiler
from .executive import ExecutiveAgent
from .replyer import Replyer

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.language_models.chat_models import BaseChatModel

    from kafubot.cognition.plugins.environment import SocialEnvironment
    from kafubot.cognition.plugins.world_model import SelfStateStore, WorldModelHub
    from kafubot.config import AgentConfig


class InteractionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning_effort: Literal["high", "max"] = "max"
    prompt_enabled: bool = True
    clock_enabled: bool = True


def apply(context: PluginContext, config: InteractionConfig) -> None:
    factory = cast(
        "Callable[[], BaseChatModel] | None",
        context.optional_service("model_factory_override"),
    )
    context.provide(
        "model_factory",
        factory or (lambda: get_thinking_model(config.reasoning_effort)),
    )
    context.provide("replyer", Replyer())

    def assemble() -> None:
        agent_config = cast("AgentConfig", context.service("config"))
        environment = cast("SocialEnvironment", context.service("environment"))
        providers = cast("WorldModelHub", context.service("world_model"))
        replyer = cast("Replyer", context.service("replyer"))
        state_store = cast("SelfStateStore", context.service("state_store"))
        host = cast("PluginHost", context.service("plugin_host"))
        compiler = ContextCompiler(environment, providers, agent_config.executive)
        context.provide("context_compiler", compiler)
        context.provide(
            "executive_agent",
            ExecutiveAgent(
                environment,
                compiler,
                replyer,
                providers,
                state_store,
                agent_config.executive,
                host,
                home_tools=context.tools_for(ToolScope.HOME),
                open_tools=context.tools_for(ToolScope.OPEN),
                prompt_enabled=config.prompt_enabled,
                clock_enabled=config.clock_enabled,
            ),
        )

    context.ready(assemble)


plugin = PluginDefinition(
    name="interaction",
    apply=apply,
    requires=("environment", "world_model"),
    config_model=InteractionConfig,
)


__all__ = [
    "ContextCompiler",
    "ExecutiveAgent",
    "InteractionConfig",
    "plugin",
    "Replyer",
]
