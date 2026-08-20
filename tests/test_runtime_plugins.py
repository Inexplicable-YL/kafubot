from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import anyio
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain.tools import tool
from langchain_core.messages import ToolMessage

from kafubot.agency.models import SelfState, SocialHome
from kafubot.cognition.plugins.base import (
    PluginCatalog,
    PluginContext,
    PluginDefinition,
    ToolScope,
)
from kafubot.cognition.plugins.environment import (
    InMemoryHistoryRepository,
    SocialEnvironment,
)
from kafubot.cognition.plugins.environment import plugin as environment_plugin
from kafubot.cognition.plugins.interaction import ExecutiveMiddleware
from kafubot.cognition.plugins.interaction import plugin as interaction_plugin
from kafubot.cognition.plugins.model import plugin as model_plugin
from kafubot.cognition.plugins.replyer import plugin as replyer_plugin
from kafubot.cognition.plugins.self_state import SelfStateStore
from kafubot.cognition.plugins.self_state import plugin as self_state_plugin
from kafubot.cognition.plugins.world_model import plugin as world_model_plugin
from kafubot.config import AgentConfig, PluginSettings
from kafubot.social import SocialAgentContext, SocialAgentRuntime


class FakeImageAnalyzer:
    async def ainvoke(
        self,
        input: dict[str, Any],  # noqa: A002
        config: Any = None,
        **kwargs: Any,
    ) -> str:
        _ = input, config, kwargs
        return ""


@tool
def home_plugin_tool() -> str:
    """Test-only HOME plugin tool."""
    return "home"


@tool
def open_plugin_tool() -> str:
    """Test-only OPEN plugin tool."""
    return "open"


@tool
def shared_plugin_tool() -> str:
    """Test-only plugin tool available in both states."""
    return "both"


class OwnedMiddleware(AgentMiddleware[Any, Any, Any]):
    def __init__(self, closed: list[str]) -> None:
        self.closed = closed

    async def aclose(self) -> None:
        self.closed.append("owned")


def test_runtime_mounts_native_plugins_into_main_executive(monkeypatch: Any) -> None:
    closed: list[str] = []
    owned_middleware = OwnedMiddleware(closed)

    async def close_recording() -> None:
        closed.append("recording")

    def recording(
        context: PluginContext,
        _options: dict[str, Any],
    ) -> None:
        context.on_prepare_reply(lambda _preparation: "recorded")
        context.effect(close_recording)

    def owned(
        context: PluginContext,
        _options: dict[str, Any],
    ) -> None:
        context.middleware(owned_middleware)
        context.tool(home_plugin_tool, ToolScope.HOME)
        context.tool(open_plugin_tool, ToolScope.OPEN)
        context.tool(shared_plugin_tool, ToolScope.BOTH)

    catalog = PluginCatalog(
        [
            environment_plugin,
            model_plugin,
            self_state_plugin,
            world_model_plugin,
            replyer_plugin,
            interaction_plugin,
            PluginDefinition("recording", recording),
            PluginDefinition("owned", owned),
        ]
    )
    monkeypatch.setattr(
        "kafubot.social.discover_plugins",
        lambda: catalog,
    )

    async def scenario() -> None:
        runtime = SocialAgentRuntime(
            AgentConfig(plugins={"interaction": PluginSettings()}),
            environment=SocialEnvironment(history=InMemoryHistoryRepository()),
            state_store=SelfStateStore(None),
            image_analyzer_factory=FakeImageAnalyzer,
        )
        await runtime._ensure_plugins()

        assert len(runtime.middleware) == 2
        executive = runtime.middleware[0]
        assert isinstance(executive, ExecutiveMiddleware)
        assert executive.plugins is runtime._plugin_host
        assert runtime.middleware[1] is owned_middleware
        assert runtime.providers is not None
        assert len(runtime.providers.providers) == 2
        assert {tool.name for tool in executive._home_tools} == {
            "open_chat",
            "finish",
            "home_plugin_tool",
            "shared_plugin_tool",
        }
        assert {tool.name for tool in executive._open_tools} == {
            "read_more",
            "inspect_person",
            "inspect_media",
            "search_memory",
            "quit",
            "reply",
            "open_plugin_tool",
            "shared_plugin_tool",
        }
        assert {tool.name for tool in executive.tools} == (
            {tool.name for tool in executive._home_tools}
            | {tool.name for tool in executive._open_tools}
        )

        context = SocialAgentContext(
            home=SocialHome(),
            state=SelfState(),
            provider_peek=[],
            budget=4,
        )
        request = ToolCallRequest(
            tool_call={
                "name": "open_plugin_tool",
                "args": {},
                "id": "tool-call",
                "type": "tool_call",
            },
            tool=open_plugin_tool,
            state={},
            runtime=cast("Any", SimpleNamespace(context=context)),
        )
        handled: list[str] = []

        async def handler(_request: ToolCallRequest) -> ToolMessage:
            handled.append("called")
            return ToolMessage(
                content="ok",
                tool_call_id="tool-call",
                name="open_plugin_tool",
            )

        rejected = await executive.awrap_tool_call(request, handler)
        assert isinstance(rejected, ToolMessage)
        assert rejected.status == "error"
        assert handled == []

        context.open_session_id = "session"
        accepted = await executive.awrap_tool_call(request, handler)
        assert isinstance(accepted, ToolMessage)
        assert accepted.status == "success"
        assert handled == ["called"]

        await runtime.aclose()

    anyio.run(scenario)
    assert closed == ["owned", "recording"]
