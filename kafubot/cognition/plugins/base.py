from __future__ import annotations

import logging
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from enum import StrEnum
from inspect import isawaitable
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar, cast
from typing_extensions import override

import aiosqlite
from langchain_core.tools import BaseTool
from langchain_openai import OpenAIEmbeddings
from langgraph.store.sqlite import AsyncSqliteStore
from pydantic import BaseModel, ConfigDict, Field

from kafubot.config import PluginSettings

if TYPE_CHECKING:
    from kafubot.agency.models import SelfState, SocialHome
    from kafubot.cognition.plugins.world_model import ProviderQuery, WorldModelProvider

    from .lifecycle import (
        ContextWindowEvicted,
        ObservationEvent,
        ReplyCommitted,
        ReplyPreparation,
        SkipCommitted,
    )

logger = logging.getLogger(__name__)

T = TypeVar("T")
MaybeAwaitable: TypeAlias = Awaitable[T] | T
EffectDisposer: TypeAlias = Callable[[], MaybeAwaitable[None]]
SessionClearer: TypeAlias = Callable[[str], MaybeAwaitable[None]]
ReadyHook: TypeAlias = Callable[[], MaybeAwaitable[None]]
ObserveHook: TypeAlias = Callable[["ObservationEvent"], MaybeAwaitable[None]]
ContextEvictionHook: TypeAlias = Callable[
    ["ContextWindowEvicted"], MaybeAwaitable[None]
]
PeekHook: TypeAlias = Callable[
    ["SocialHome", "SelfState"],
    MaybeAwaitable[str | None],
]
QueryHook: TypeAlias = Callable[["ProviderQuery"], MaybeAwaitable[str | None]]
PreparationHook: TypeAlias = Callable[
    ["ReplyPreparation"],
    MaybeAwaitable[str | None],
]
ReplyHook: TypeAlias = Callable[["ReplyCommitted"], MaybeAwaitable[None]]
SkipHook: TypeAlias = Callable[["SkipCommitted"], MaybeAwaitable[None]]


class ToolScope(StrEnum):
    """Executive states in which a registered plugin tool is visible and callable."""

    HOME = "home"
    OPEN = "open"
    BOTH = "both"


class PluginTool(BaseModel):
    """One executable tool plus its Main Executive availability contract."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    tool: BaseTool
    scope: ToolScope

    def is_available(self, scope: ToolScope) -> bool:
        return self.scope in {scope, ToolScope.BOTH}


class _PluginContribution(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tools: list[PluginTool] = Field(default_factory=list)
    providers: list[Any] = Field(default_factory=list)
    services: dict[str, Any] = Field(default_factory=dict)
    effects: list[EffectDisposer] = Field(default_factory=list)
    session_clearers: list[SessionClearer] = Field(default_factory=list)
    ready_hooks: list[ReadyHook] = Field(default_factory=list)
    observe_hooks: list[Any] = Field(default_factory=list)
    context_eviction_hooks: list[Any] = Field(default_factory=list)
    peek_hooks: list[Any] = Field(default_factory=list)
    query_hooks: list[Any] = Field(default_factory=list)
    preparation_hooks: list[Any] = Field(default_factory=list)
    guard_hooks: list[Any] = Field(default_factory=list)
    reply_hooks: list[Any] = Field(default_factory=list)
    skip_hooks: list[Any] = Field(default_factory=list)
    owned_resources: set[int] = Field(default_factory=set)


class PluginBuildContext:
    """Shared capability graph behind all isolated plugin contexts."""

    def __init__(self, services: Mapping[str, Any] | None = None) -> None:
        self.services: dict[str, Any] = dict(services or {})
        self.components: dict[str, _PluginContribution] = {}
        self.enabled: frozenset[str] = frozenset()

    def tools_for(self, scope: ToolScope) -> tuple[BaseTool, ...]:
        return tuple(
            registration.tool
            for contribution in self.components.values()
            for registration in contribution.tools
            if registration.is_available(scope)
        )

    def providers(self) -> tuple[WorldModelProvider, ...]:
        return tuple(
            provider
            for contribution in self.components.values()
            for provider in contribution.providers
        )

    def service(self, name: str) -> Any:
        try:
            return self.services[name]
        except KeyError as exc:
            raise LookupError(f"plugin service is unavailable: {name}") from exc


class PluginContext:
    """The only registration surface available to an autonomous plugin."""

    def __init__(
        self,
        plugin_name: str,
        root: PluginBuildContext,
        contribution: _PluginContribution,
    ) -> None:
        self.plugin_name = plugin_name
        self._root = root
        self._contribution = contribution

    def enabled(self, plugin_name: str) -> bool:
        return plugin_name in self._root.enabled

    def service(self, name: str) -> Any:
        return self._root.service(name)

    def optional_service(self, name: str) -> Any | None:
        return self._root.services.get(name)

    def tools_for(self, scope: ToolScope) -> tuple[BaseTool, ...]:
        return self._root.tools_for(scope)

    def providers(self) -> tuple[WorldModelProvider, ...]:
        return self._root.providers()

    def provide(self, name: str, service: T) -> T:
        if name in self._root.services:
            raise ValueError(f"plugin service is already registered: {name}")
        self._root.services[name] = service
        self._contribution.services[name] = service
        return service

    def resource(self, name: str, resource: T) -> T:
        """Publish and own a closeable service in one registration."""
        self.provide(name, resource)
        self.own(resource)
        return resource

    def own(self, resource: T) -> T:
        """Bind a closeable resource to this plugin's reversible lifetime."""
        identity = id(resource)
        if identity in self._contribution.owned_resources:
            return resource
        closer = getattr(resource, "aclose", None) or getattr(resource, "close", None)
        if not callable(closer):
            raise TypeError(
                f"plugin {self.plugin_name} resource has no close/aclose method"
            )
        self._contribution.owned_resources.add(identity)
        self.effect(cast("EffectDisposer", closer))
        return resource

    def tool(self, tool: BaseTool, scope: ToolScope = ToolScope.BOTH) -> BaseTool:
        if not isinstance(tool, BaseTool):
            raise TypeError(
                f"plugin {self.plugin_name} tool registration must contain BaseTool"
            )
        self._contribution.tools.append(PluginTool(tool=tool, scope=scope))
        return tool

    def provider(self, provider: WorldModelProvider, *, owned: bool = True) -> Any:
        self._contribution.providers.append(provider)
        if owned and callable(
            getattr(provider, "aclose", None) or getattr(provider, "close", None)
        ):
            self.own(provider)
        return provider

    def effect(self, disposer: EffectDisposer) -> EffectDisposer:
        self._contribution.effects.append(disposer)
        return disposer

    def clear_session(self, clearer: SessionClearer) -> SessionClearer:
        self._contribution.session_clearers.append(clearer)
        return clearer

    def ready(self, callback: ReadyHook) -> ReadyHook:
        """Run after every enabled plugin has applied and published its services."""
        self._contribution.ready_hooks.append(callback)
        return callback

    async def sqlite_store(
        self,
        path: str,
        *,
        indexed: bool = False,
        embedding_model: str = "text-embedding-3-large",
        dimensions: int = 3072,
    ) -> AsyncSqliteStore:
        """Open a plugin-owned SQLite store with automatic rollback and shutdown."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        connection = await aiosqlite.connect(path, isolation_level=None)
        try:
            index = None
            if indexed:
                index = {
                    "embed": OpenAIEmbeddings(
                        model=embedding_model,
                        base_url=os.getenv("OPENAI_BASE_URL"),
                    ),
                    "dims": dimensions,
                }
            store = AsyncSqliteStore(conn=connection, index=cast("Any", index))
        except BaseException:
            await connection.close()
            raise
        self.effect(connection.close)
        return store

    def on_observe(self, callback: ObserveHook) -> ObserveHook:
        self._contribution.observe_hooks.append(callback)
        return callback

    def on_context_evicted(
        self,
        callback: ContextEvictionHook,
    ) -> ContextEvictionHook:
        """Consume messages only after they leave the live context window."""
        self._contribution.context_eviction_hooks.append(callback)
        return callback

    def on_peek(self, callback: PeekHook) -> PeekHook:
        self._contribution.peek_hooks.append(callback)
        return callback

    def on_query(self, callback: QueryHook) -> QueryHook:
        self._contribution.query_hooks.append(callback)
        return callback

    def on_prepare_reply(self, callback: PreparationHook) -> PreparationHook:
        self._contribution.preparation_hooks.append(callback)
        return callback

    def on_guard_reply(self, callback: PreparationHook) -> PreparationHook:
        self._contribution.guard_hooks.append(callback)
        return callback

    def on_reply_committed(self, callback: ReplyHook) -> ReplyHook:
        self._contribution.reply_hooks.append(callback)
        return callback

    def on_skip_committed(self, callback: SkipHook) -> SkipHook:
        self._contribution.skip_hooks.append(callback)
        return callback


PluginApply: TypeAlias = Callable[[PluginContext, Any], MaybeAwaitable[None]]


class PluginDefinition(BaseModel):
    """Self-description exported by one autonomous plugin module."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    name: str
    apply: PluginApply
    default_enabled: bool = True
    requires: tuple[str, ...] = ()
    after: tuple[str, ...] = ()
    config_model: type[BaseModel] | None = None

    def __init__(
        self,
        name: str,
        apply: PluginApply,
        *,
        default_enabled: bool = True,
        requires: tuple[str, ...] = (),
        after: tuple[str, ...] = (),
        config_model: type[BaseModel] | None = None,
    ) -> None:
        super().__init__(
            name=name,
            apply=apply,
            default_enabled=default_enabled,
            requires=requires,
            after=after,
            config_model=config_model,
        )

    @override
    def model_post_init(self, _context: Any, /) -> None:
        if not self.name.strip():
            raise ValueError("plugin name cannot be empty")
        if self.name in self.requires:
            raise ValueError(f"plugin cannot require itself: {self.name}")
        if self.name in self.after:
            raise ValueError(f"plugin cannot order itself after itself: {self.name}")

    def parse_options(self, options: Mapping[str, Any]) -> Any:
        if self.config_model is None:
            return dict(options)
        return self.config_model.model_validate(options)


class PluginCatalog:
    """Deterministic set of discovered, self-describing plugin definitions."""

    def __init__(self, definitions: Sequence[PluginDefinition] = ()) -> None:
        self._definitions: dict[str, PluginDefinition] = {}
        for definition in definitions:
            self.register(definition)

    def register(self, definition: PluginDefinition) -> None:
        if definition.name in self._definitions:
            raise ValueError(f"plugin is already registered: {definition.name}")
        self._definitions[definition.name] = definition

    @property
    def definitions(self) -> tuple[PluginDefinition, ...]:
        return tuple(self._definitions.values())

    def resolve(
        self,
        settings: Mapping[str, PluginSettings],
    ) -> tuple[PluginDefinition, ...]:
        unknown = set(settings).difference(self._definitions)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown plugins: {names}")

        enabled = {
            definition.name
            for definition in self._definitions.values()
            if settings.get(
                definition.name,
                PluginSettings(enabled=definition.default_enabled),
            ).enabled
        }
        ordered: list[PluginDefinition] = []
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(name: str) -> None:
            if name in visited:
                return
            if name in visiting:
                raise ValueError(f"cyclic plugin dependency at {name}")
            visiting.add(name)
            definition = self._definitions[name]
            missing = [item for item in definition.requires if item not in enabled]
            if missing:
                required = ", ".join(missing)
                raise ValueError(
                    f"plugin {name} requires enabled plugin(s): {required}"
                )
            for dependency in definition.requires:
                visit(dependency)
            for predecessor in definition.after:
                if predecessor in enabled:
                    visit(predecessor)
            visiting.remove(name)
            visited.add(name)
            ordered.append(definition)

        for definition in self._definitions.values():
            if definition.name in enabled:
                visit(definition.name)
        return tuple(ordered)


class PluginHost:
    """Own, compose and dispatch every capability in one plugin graph."""

    name = "cognitive_plugins"

    def __init__(self, context: PluginBuildContext) -> None:
        self.context = context
        self._ordered: list[tuple[str, _PluginContribution]] = []
        self._tool_owners: dict[str, str] = {}

    @classmethod
    async def build(
        cls,
        catalog: PluginCatalog,
        settings: Mapping[str, PluginSettings],
        *,
        services: Mapping[str, Any] | None = None,
    ) -> PluginHost:
        root = PluginBuildContext(services)
        host = cls(root)
        if "plugin_host" in root.services:
            raise ValueError("plugin service is already registered: plugin_host")
        root.services["plugin_host"] = host
        definitions = catalog.resolve(settings)
        root.enabled = frozenset(definition.name for definition in definitions)
        try:
            for definition in definitions:
                contribution = _PluginContribution()
                host._ordered.append((definition.name, contribution))
                root.components[definition.name] = contribution
                context = PluginContext(definition.name, root, contribution)
                options = settings.get(definition.name, PluginSettings()).options
                await host._complete(
                    definition.apply(context, definition.parse_options(options)),
                    error=(f"plugin {definition.name} apply() must not return a value"),
                )
            for name, contribution in host._ordered:
                for callback in contribution.ready_hooks:
                    await host._complete(
                        callback(),
                        error=f"plugin {name} ready hook must not return a value",
                    )
            for name, contribution in host._ordered:
                host._register_tools(name, contribution.tools)
        except BaseException:
            await host.aclose()
            raise
        return host

    async def _complete(self, result: MaybeAwaitable[T], *, error: str) -> T:
        if isawaitable(result):
            result = await result
        if result is not None:
            raise TypeError(error)
        return result

    @property
    def enabled(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self._ordered)

    def has(self, plugin_name: str) -> bool:
        return plugin_name in self.context.enabled

    def service(self, name: str) -> Any:
        return self.context.service(name)

    def optional_service(self, name: str) -> Any | None:
        return self.context.services.get(name)

    @property
    def providers(self) -> tuple[WorldModelProvider, ...]:
        return tuple(
            provider
            for _, contribution in self._ordered
            for provider in contribution.providers
        )

    def tools_for(self, scope: ToolScope) -> tuple[BaseTool, ...]:
        return tuple(
            registration.tool
            for _, contribution in self._ordered
            for registration in contribution.tools
            if registration.is_available(scope)
        )

    def _register_tools(
        self,
        plugin_name: str,
        registrations: Sequence[PluginTool],
    ) -> None:
        for registration in registrations:
            tool_name = registration.tool.name
            if owner := self._tool_owners.get(tool_name):
                raise ValueError(
                    f"plugin tool name is already registered: {tool_name} "
                    f"({owner}, {plugin_name})"
                )
            self._tool_owners[tool_name] = plugin_name

    async def observe(self, event: ObservationEvent) -> None:
        await self._notify("observe", "observe_hooks", event)

    async def context_evicted(self, event: ContextWindowEvicted) -> None:
        await self._notify(
            "context_evicted",
            "context_eviction_hooks",
            event,
        )

    async def peek(self, home: SocialHome, state: SelfState) -> str | None:
        outputs = await self._collect("peek", "peek_hooks", home, state)
        return "\n\n".join(outputs) or None

    async def query(self, request: ProviderQuery) -> str | None:
        outputs = await self._collect("query", "query_hooks", request)
        return "\n\n".join(outputs) or None

    async def prepare_reply(self, preparation: ReplyPreparation) -> list[str]:
        return await self._collect(
            "prepare_reply",
            "preparation_hooks",
            preparation,
        )

    async def guard_reply(self, preparation: ReplyPreparation) -> str | None:
        for name, contribution in self._ordered:
            for callback in contribution.guard_hooks:
                try:
                    result = callback(preparation)
                    rejection = await result if isawaitable(result) else result
                    if rejection:
                        return rejection
                except Exception:
                    logger.exception("Plugin reply guard failed: %s", name)
                    return "a reply safety plugin failed; no message was sent"
        return None

    async def reply_committed(self, event: ReplyCommitted) -> None:
        await self._notify("reply_committed", "reply_hooks", event)

    async def skip_committed(self, event: SkipCommitted) -> None:
        await self._notify("skip_committed", "skip_hooks", event)

    async def commit(self, change: Any) -> None:
        """Participate directly in the composed world-model provider surface."""
        _ = change

    async def _notify(self, label: str, field_name: str, event: Any) -> None:
        for name, contribution in self._ordered:
            for callback in getattr(contribution, field_name):
                try:
                    result = callback(event)
                    if isawaitable(result):
                        await result
                except Exception:
                    logger.exception("Plugin hook failed: %s.%s", name, label)

    async def _collect(
        self,
        label: str,
        field_name: str,
        *args: Any,
    ) -> list[str]:
        outputs: list[str] = []
        for name, contribution in self._ordered:
            for callback in getattr(contribution, field_name):
                try:
                    result = callback(*args)
                    output = await result if isawaitable(result) else result
                except Exception:
                    logger.exception("Plugin query failed: %s.%s", name, label)
                    continue
                if output:
                    outputs.append(output)
        return outputs

    async def clear_session(self, session_id: str) -> None:
        for name, contribution in reversed(self._ordered):
            for clearer in reversed(contribution.session_clearers):
                try:
                    result = clearer(session_id)
                    if isawaitable(result):
                        await result
                except Exception:
                    logger.exception("Plugin session clear failed: %s", name)

    async def aclose(self) -> None:
        ordered, self._ordered = self._ordered, []
        for name, contribution in reversed(ordered):
            for disposer in reversed(contribution.effects):
                try:
                    result = disposer()
                    if isawaitable(result):
                        await result
                except Exception:
                    logger.exception("Plugin shutdown failed: %s", name)
        self.context.components.clear()
        self.context.services.clear()
        self._tool_owners.clear()


__all__ = [
    "PluginBuildContext",
    "PluginCatalog",
    "PluginContext",
    "PluginDefinition",
    "PluginHost",
    "PluginSettings",
    "PluginTool",
    "ToolScope",
]
