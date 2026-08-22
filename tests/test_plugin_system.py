from __future__ import annotations

import sys
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import anyio
import pytest
from langchain.tools import tool
from pydantic import BaseModel, ConfigDict, ValidationError

from kafubot.agency.models import TimelineEntry
from kafubot.cognition.plugins.base import (
    PluginCatalog,
    PluginContext,
    PluginDefinition,
    PluginHost,
    ToolScope,
)
from kafubot.cognition.plugins.lifecycle import ContextWindowEvicted
from kafubot.cognition.plugins.loader import discover_plugins
from kafubot.config import PluginSettings

if TYPE_CHECKING:
    from pathlib import Path


@tool
def home_capability() -> str:
    """Test-only HOME capability."""
    return "home"


@tool
def open_capability() -> str:
    """Test-only OPEN capability."""
    return "open"


@tool
def shared_capability() -> str:
    """Test-only capability available in both states."""
    return "both"


def noop(_context: PluginContext, _config: Any) -> None:
    pass


def test_discovery_only_accepts_explicit_direct_file_or_package_plugins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "sample_plugins"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "framework.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package / "single.py").write_text(
        "from kafubot.cognition.plugins import PluginDefinition\n"
        "def apply(context, config): pass\n"
        "plugin = PluginDefinition(name='single', apply=apply)\n",
        encoding="utf-8",
    )
    packaged = package / "packaged"
    packaged.mkdir()
    (packaged / "__init__.py").write_text(
        "from kafubot.cognition.plugins import PluginDefinition\n"
        "def apply(context, config): pass\n"
        "plugin = PluginDefinition(name='packaged', apply=apply)\n",
        encoding="utf-8",
    )
    (packaged / "hidden.py").write_text(
        "from kafubot.cognition.plugins import PluginDefinition\n"
        "def apply(context, config): pass\n"
        "plugin = PluginDefinition(name='must_not_be_scanned', apply=apply)\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    catalog = discover_plugins("sample_plugins", include_entry_points=False)

    assert {definition.name for definition in catalog.definitions} == {
        "packaged",
        "single",
    }
    assert "sample_plugins.packaged.hidden" not in sys.modules


def test_catalog_orders_dependencies_and_soft_predecessors() -> None:
    catalog = PluginCatalog(
        [
            PluginDefinition("consumer", noop, requires=("store",), after=("audit",)),
            PluginDefinition("audit", noop),
            PluginDefinition("store", noop),
        ]
    )

    resolved = catalog.resolve({})
    assert [definition.name for definition in resolved] == [
        "store",
        "audit",
        "consumer",
    ]

    with pytest.raises(ValueError, match="requires enabled plugin"):
        catalog.resolve({"store": PluginSettings(enabled=False)})

    resolved_without_optional = catalog.resolve(
        {"audit": PluginSettings(enabled=False)}
    )
    assert [definition.name for definition in resolved_without_optional] == [
        "store",
        "consumer",
    ]


def test_catalog_rejects_unknown_plugin_configuration() -> None:
    with pytest.raises(ValueError, match="unknown plugins: typo"):
        PluginCatalog().resolve({"typo": PluginSettings()})


def test_host_is_instance_scoped_and_honors_toggles() -> None:
    contexts: list[PluginContext] = []

    def apply(context: PluginContext, options: dict[str, Any]) -> None:
        contexts.append(context)
        context.provide("value", options.get("value"))

    async def scenario() -> None:
        catalog = PluginCatalog([PluginDefinition("component", apply)])
        disabled = await PluginHost.build(
            catalog,
            {"component": PluginSettings(enabled=False)},
        )
        first = await PluginHost.build(
            catalog,
            {"component": PluginSettings(options={"value": 1})},
        )
        second = await PluginHost.build(
            catalog,
            {"component": PluginSettings(options={"value": 2})},
        )

        assert disabled.enabled == ()
        assert first.enabled == ("component",)
        assert first.context.service("value") == 1
        assert second.context.service("value") == 2
        assert contexts[0] is not contexts[1]

        await disabled.aclose()
        await first.aclose()
        await second.aclose()

    anyio.run(scenario)


def test_plugin_api_dispatches_native_context_eviction_hooks() -> None:
    received: list[tuple[str, tuple[int, ...]]] = []

    def apply(context: PluginContext, _config: Any) -> None:
        def evicted(event: ContextWindowEvicted) -> None:
            received.append(
                (event.session_id, tuple(entry.sequence for entry in event.entries))
            )

        context.on_context_evicted(evicted)

    async def scenario() -> None:
        host = await PluginHost.build(
            PluginCatalog([PluginDefinition("learner", apply)]),
            {},
        )
        event = ContextWindowEvicted(
            session_id="session",
            entries=(
                TimelineEntry(
                    sequence=4,
                    role="assistant",
                    timestamp=datetime.now(UTC),
                    content="outside",
                ),
            ),
        )
        await host.context_evicted(event)
        await host.aclose()

    anyio.run(scenario)
    assert received == [("session", (4,))]
    assert not hasattr(PluginContext, "middleware")
    assert not hasattr(PluginHost, "middleware")


def test_plugin_consumes_dependency_service_through_its_context() -> None:
    def provide_store(context: PluginContext, _config: Any) -> None:
        context.provide("store", "ready")

    def consume_store(context: PluginContext, _config: Any) -> None:
        context.provide("result", f"{context.service('store')}:used")

    async def scenario() -> None:
        host = await PluginHost.build(
            PluginCatalog(
                [
                    PluginDefinition(
                        "consumer",
                        consume_store,
                        requires=("store",),
                    ),
                    PluginDefinition("store", provide_store),
                ]
            ),
            {},
        )
        assert host.context.service("result") == "ready:used"
        await host.aclose()

    anyio.run(scenario)


def test_ready_hook_composes_later_tools_services_and_lifecycle_hooks() -> None:
    assembled: list[tuple[str, tuple[str, ...]]] = []

    def shell(context: PluginContext, _config: Any) -> None:
        async def prepare(_preparation: Any) -> str:
            return "native context"

        def ready() -> None:
            assembled.append(
                (
                    cast("str", context.service("later")),
                    tuple(tool.name for tool in context.tools_for(ToolScope.OPEN)),
                )
            )

        context.on_prepare_reply(prepare)
        context.ready(ready)

    def later(context: PluginContext, _config: Any) -> None:
        context.provide("later", "available")
        context.tool(open_capability, ToolScope.OPEN)

    async def scenario() -> None:
        host = await PluginHost.build(
            PluginCatalog(
                [
                    PluginDefinition("shell", shell),
                    PluginDefinition("later", later),
                ]
            ),
            {},
        )
        assert assembled == [("available", ("open_capability",))]
        assert await host.prepare_reply(cast("Any", object())) == ["native context"]
        await host.aclose()

    anyio.run(scenario)


def test_host_registers_tools_by_executive_scope() -> None:
    def apply(context: PluginContext, _config: Any) -> None:
        context.tool(home_capability, ToolScope.HOME)
        context.tool(open_capability, ToolScope.OPEN)
        context.tool(shared_capability, ToolScope.BOTH)

    async def scenario() -> None:
        host = await PluginHost.build(
            PluginCatalog([PluginDefinition("scoped", apply)]),
            {},
        )
        assert [tool.name for tool in host.tools_for(ToolScope.HOME)] == [
            "home_capability",
            "shared_capability",
        ]
        assert [tool.name for tool in host.tools_for(ToolScope.OPEN)] == [
            "open_capability",
            "shared_capability",
        ]
        await host.aclose()

    anyio.run(scenario)


def test_host_rejects_duplicate_tool_names() -> None:
    def first(context: PluginContext, _config: Any) -> None:
        context.tool(home_capability, ToolScope.HOME)

    def second(context: PluginContext, _config: Any) -> None:
        context.tool(home_capability, ToolScope.OPEN)

    async def scenario() -> None:
        with pytest.raises(ValueError, match="already registered"):
            await PluginHost.build(
                PluginCatalog(
                    [
                        PluginDefinition("first", first),
                        PluginDefinition("second", second),
                    ]
                ),
                {},
            )

    anyio.run(scenario)


def test_host_rolls_back_all_started_effects_in_reverse_order() -> None:
    events: list[str] = []

    async def close(name: str) -> None:
        events.append(f"close:{name}")

    def first(context: PluginContext, _config: Any) -> None:
        events.append("start:first")
        context.effect(lambda: close("first:one"))
        context.effect(lambda: close("first:two"))

    def broken(context: PluginContext, _config: Any) -> None:
        events.append("start:broken")
        context.effect(lambda: close("broken"))
        raise RuntimeError("boom")

    async def scenario() -> None:
        catalog = PluginCatalog(
            [PluginDefinition("first", first), PluginDefinition("broken", broken)]
        )
        with pytest.raises(RuntimeError, match="boom"):
            await PluginHost.build(catalog, {})

    anyio.run(scenario)
    assert events == [
        "start:first",
        "start:broken",
        "close:broken",
        "close:first:two",
        "close:first:one",
    ]


def test_host_clears_and_closes_every_effect_despite_failure() -> None:
    events: list[str] = []

    def apply(name: str, *, fail: bool = False) -> Any:
        def register(context: PluginContext, _config: Any) -> None:
            async def clear(session_id: str) -> None:
                events.append(f"clear:{name}:{session_id}")
                if fail:
                    raise RuntimeError("clear failed")

            async def close() -> None:
                events.append(f"close:{name}")
                if fail:
                    raise RuntimeError("close failed")

            context.clear_session(clear)
            context.effect(close)

        return register

    async def scenario() -> None:
        host = await PluginHost.build(
            PluginCatalog(
                [
                    PluginDefinition("first", apply("first")),
                    PluginDefinition("second", apply("second", fail=True)),
                ]
            ),
            {},
        )
        await host.clear_session("session")
        await host.aclose()

    anyio.run(scenario)
    assert events == [
        "clear:second:session",
        "clear:first:session",
        "close:second",
        "close:first",
    ]


def test_plugin_configuration_is_validated_before_apply() -> None:
    applied: list[int] = []

    class Options(BaseModel):
        model_config = ConfigDict(extra="forbid")

        limit: int

    def apply(_context: PluginContext, options: Options) -> None:
        applied.append(options.limit)

    async def scenario() -> None:
        catalog = PluginCatalog(
            [PluginDefinition("validated", apply, config_model=Options)]
        )
        with pytest.raises(ValidationError):
            await PluginHost.build(
                catalog,
                {"validated": PluginSettings(options={"limit": "invalid"})},
            )
        assert applied == []

        host = await PluginHost.build(
            catalog,
            {"validated": PluginSettings(options={"limit": 3})},
        )
        assert applied == [3]
        await host.aclose()

    anyio.run(scenario)
