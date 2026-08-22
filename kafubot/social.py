from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any, cast

import anyio
from langchain.agents import AgentState
from langchain_core.messages import AIMessage
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel, Field

from kafubot.actions import QQActions
from kafubot.agency.models import (
    ConversationCandidate,
    RoundResult,
    SelfState,
    SocialHome,
)
from kafubot.cognition.graph import create_agent
from kafubot.cognition.media.image import ImageReadResult, get_analyzer
from kafubot.cognition.plugins.base import PluginHost
from kafubot.cognition.plugins.lifecycle import ContextWindowEvicted, ObservationEvent
from kafubot.cognition.plugins.loader import discover_plugins
from kafubot.cognition.plugins.world_model import ProviderCommit
from kafubot.log import logger

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Protocol

    from langchain_core.language_models.chat_models import BaseChatModel

    from kafubot.adapters.cqhttp.event import (
        GroupMessageEvent,
        PrivateMessageEvent,
    )
    from kafubot.cognition.plugins.environment import (
        ConversationState,
        CQHTTPMessageIngestor,
        ImageAnalyzer,
        SocialEnvironment,
    )
    from kafubot.cognition.plugins.interaction import (
        ContextCompiler,
        ExecutiveAgent,
        Replyer,
    )
    from kafubot.cognition.plugins.world_model import (
        AttentionScheduler,
        SelfStateStore,
        WorldModelHub,
    )
    from kafubot.cognition.types import UserMessage
    from kafubot.config import AgentConfig

    class WakeDecision(Protocol):
        wake: bool
        score: float
        reasons: tuple[str, ...]

    class WakeGate(Protocol):
        async def consider(
            self,
            session_id: str,
            message: UserMessage,
            candidate: ConversationCandidate,
            state: SelfState,
        ) -> WakeDecision: ...

        async def force_wake(self) -> None: ...

        async def wait(self, *, timeout: float | None = None) -> bool: ...


class SocialAgentContext(BaseModel):
    """One create_agent invocation, owned and managed by the social runtime."""

    home: SocialHome
    state: SelfState
    provider_peek: list[str]
    budget: int
    opened: dict[str, list[int]] = Field(default_factory=dict)
    open_session_id: str | None = None
    open_context_summaries: dict[str, str] = Field(default_factory=dict)
    focus_stack: list[str] = Field(default_factory=list)
    claimed: set[str] = Field(default_factory=set)
    handled: set[str] = Field(default_factory=set)
    reply_slots_used: int = 0
    replies: int = 0
    skipped: int = 0
    finished: bool = False
    exhausted_budget: bool = False

    def spend(self, cost: int) -> bool:
        if cost > self.budget:
            self.exhausted_budget = True
            return False
        self.budget -= cost
        return True


class SocialAgentState(AgentState[None]):
    """State carried by the social Executive graph."""


class SocialAgentRuntime:
    """Owns the global create_agent graph and its replaceable plugin stack."""

    def __init__(
        self,
        config: AgentConfig,
        *,
        environment: SocialEnvironment | None = None,
        state_store: SelfStateStore | None = None,
        model_factory: Callable[[], BaseChatModel] | None = None,
        image_analyzer_factory: Callable[[], ImageAnalyzer] | None = None,
    ) -> None:
        self.config = config
        self.environment: SocialEnvironment | None = environment
        self.state_store: SelfStateStore | None = state_store
        self.gate: WakeGate | None = None
        self.attention: AttentionScheduler | None = None
        self.providers: WorldModelHub | None = None
        self.compiler: ContextCompiler | None = None
        self.replyer: Replyer | None = None
        self._agent_stack: tuple[ExecutiveAgent, ...] = ()
        self._plugin_host: PluginHost | None = None
        self._plugins_lock = anyio.Lock()
        self._model_factory_override = model_factory
        self.model_factory: Callable[[], BaseChatModel] | None = model_factory
        self._model: BaseChatModel | None = None
        self._agent: Any | None = None
        self._image_analyzer_factory = image_analyzer_factory or (
            lambda: get_analyzer(True)
        )
        self.ingress: CQHTTPMessageIngestor | None = None
        self._ingest_locks: defaultdict[str, anyio.Lock] = defaultdict(anyio.Lock)
        self._wake_event = anyio.Event()
        self._wake_lock = anyio.Lock()
        self._closed = False

    @property
    def model(self) -> BaseChatModel:
        if self._model is None:
            if self.model_factory is None:
                raise RuntimeError("interaction plugin did not provide a model")
            self._model = self.model_factory()
        return self._model

    async def _get_agent(self) -> Any:
        await self._ensure_plugins()
        if self._agent is None:
            self._agent = create_agent(
                model=self.model,
                tools=[],
                middleware=self._agent_stack,
                state_schema=SocialAgentState,
                context_schema=SocialAgentContext,
                name="social_agent",
            )
        return self._agent

    async def _ensure_plugins(self) -> None:
        if self._plugin_host is not None:
            return
        async with self._plugins_lock:
            if self._plugin_host is not None:
                return
            host = await PluginHost.build(
                discover_plugins(),
                self.config.plugins,
                services={
                    "config": self.config,
                    "image_analyzer_factory": self._image_analyzer_factory,
                    "model_factory_override": self._model_factory_override,
                    **(
                        {"environment": self.environment}
                        if self.environment is not None
                        else {}
                    ),
                    **(
                        {"state_store": self.state_store}
                        if self.state_store is not None
                        else {}
                    ),
                },
            )
            providers = cast(
                "WorldModelHub | None",
                host.optional_service("world_model"),
            )
            executive = cast(
                "ExecutiveAgent | None",
                host.optional_service("executive_agent"),
            )
            middleware = (executive,) if executive is not None else ()
            self.providers = providers
            self.environment = cast(
                "SocialEnvironment | None",
                host.optional_service("environment"),
            )
            self.state_store = cast(
                "SelfStateStore | None",
                host.optional_service("state_store"),
            )
            self.gate = cast("WakeGate | None", host.optional_service("gate"))
            self.attention = cast(
                "AttentionScheduler | None",
                host.optional_service("attention"),
            )
            self.compiler = cast(
                "ContextCompiler | None",
                host.optional_service("context_compiler"),
            )
            self.replyer = cast("Replyer | None", host.optional_service("replyer"))
            self.ingress = cast(
                "CQHTTPMessageIngestor | None",
                host.optional_service("ingress"),
            )
            configured_model_factory = cast(
                "Callable[[], BaseChatModel] | None",
                host.optional_service("model_factory"),
            )
            if configured_model_factory is not None:
                self.model_factory = configured_model_factory
            self._agent_stack = middleware
            self._plugin_host = host

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._force_wake()
        if self._plugin_host is not None:
            await self._plugin_host.aclose()
            self._plugin_host = None

    async def run_round(self) -> RoundResult:
        await self._ensure_plugins()
        if self.environment is None or self.state_store is None:
            return RoundResult()
        state = await self.state_store.cool_fatigue()
        home = (
            self.attention.rank(
                await self.environment.candidates(),
                state,
                limit=self.config.executive.home_session_limit,
            )
            if self.attention is not None
            else SocialHome()
        )
        result = RoundResult(home_size=len(home.items))
        if not home.items:
            return result

        context = SocialAgentContext(
            home=home,
            state=state,
            provider_peek=(
                await self.providers.peek(home, state)
                if self.providers is not None
                else []
            ),
            budget=self.config.executive.focus_budget,
        )
        try:
            if not cast("PluginHost", self._plugin_host).has("interaction"):
                return result
            output: dict[str, Any] = await (await self._get_agent()).ainvoke(
                {"messages": []},
                config={"recursion_limit": self.config.executive.max_steps * 2},
                context=context,
            )
        except GraphRecursionError:
            logger.info(
                "Social agent reached its create_agent step limit",
                max_steps=self.config.executive.max_steps,
            )
            result.steps = self.config.executive.max_steps
        else:
            messages = output.get("messages", [])
            if isinstance(messages, list):
                result.steps = min(
                    self.config.executive.max_steps,
                    sum(isinstance(message, AIMessage) for message in messages),
                )

        result.replies = context.replies
        result.skipped = context.skipped
        result.exhausted_budget = context.exhausted_budget
        return result

    async def run(self) -> None:
        """Run the only cognitive loop. Protocol workers only update the world."""
        await self._ensure_plugins()
        while not self._closed:
            woke = await self._wait_for_wake(self.config.gate.idle_wake_seconds)
            if self._closed:
                return
            if not woke and not await self._has_idle_work():
                continue
            try:
                result = await self.run_round()
                logger.info(
                    "Social agent round completed",
                    home_size=result.home_size,
                    steps=result.steps,
                    replies=result.replies,
                    skipped=result.skipped,
                    exhausted_budget=result.exhausted_budget,
                )
            except anyio.get_cancelled_exc_class():
                raise
            except Exception:
                logger.exception("Social agent round failed")

    async def session(self, session_id: str) -> ConversationState:
        await self._ensure_plugins()
        if self.environment is None:
            raise RuntimeError("environment plugin is disabled")
        return await self.environment.session(session_id)

    async def clear_session(self, session_id: str, actions: QQActions) -> None:
        await self._ensure_plugins()
        if self.environment is None or self.state_store is None:
            raise RuntimeError("environment or self_state plugin is disabled")
        await self.environment.clear(session_id)
        await self.state_store.clear_session(session_id)
        await cast("PluginHost", self._plugin_host).clear_session(session_id)
        if self.providers is not None:
            await self.providers.commit(
                ProviderCommit(operation="clear", session_id=session_id)
            )
        await actions.reply("[SYSTEM] 已清除当前会话历史")

    async def get_image(
        self,
        actions: QQActions,
        file: str,
    ) -> ImageReadResult | None:
        await self._ensure_plugins()
        return (
            await self.ingress.get_image(actions, file)
            if self.ingress is not None
            else None
        )

    async def to_user_message(
        self,
        event: GroupMessageEvent | PrivateMessageEvent,
        session: ConversationState,
        actions: QQActions,
    ) -> UserMessage | None:
        await self._ensure_plugins()
        return (
            await self.ingress.convert(event, list(session.entries), actions)
            if self.ingress is not None
            else None
        )

    async def fill_images(self, message: UserMessage) -> UserMessage | None:
        await self._ensure_plugins()
        return (
            await self.ingress.fill_images(message)
            if self.ingress is not None
            else None
        )

    async def handle(
        self,
        event: GroupMessageEvent | PrivateMessageEvent,
    ) -> None:
        """Commit one accepted chat event; never invoke the executive directly."""
        await self._ensure_plugins()
        environment = self.environment
        state_store = self.state_store
        if environment is None or state_store is None:
            return
        if (
            event.user_id == event.self_id
            or event.user_id in self.config.ignore_user_ids
        ):
            return
        session_id = event.get_conversation_id()
        actions = QQActions(event)
        if event.get_plain_text().strip() in self.config.clear_keywords:
            await self.clear_session(session_id, actions)
            return

        async with self._ingest_locks[session_id]:
            session = await environment.session(session_id)
            message = await self.to_user_message(event, session, actions)
            if message is None:
                return
            candidate = await environment.observe(session_id, message, actions)
            history = tuple(
                await environment.read(
                    session_id,
                    limit=self.config.environment_message_limit,
                )
            )
        await cast("PluginHost", self._plugin_host).observe(
            ObservationEvent(
                session_id=session_id,
                message=message,
                candidate=candidate,
                history=history,
                actions=actions,
            )
        )
        evicted = await environment.take_context_evictions(session_id)
        if evicted:
            await cast("PluginHost", self._plugin_host).context_evicted(
                ContextWindowEvicted(session_id=session_id, entries=tuple(evicted))
            )
        state = await state_store.load()
        if self.gate is not None:
            decision = await self.gate.consider(
                session_id,
                message,
                candidate,
                state,
            )
        else:
            decision = None
            await self._force_wake()
        logger.debug(
            "Global Gate evaluated event",
            session_id=session_id,
            wake=decision.wake if decision is not None else True,
            score=decision.score if decision is not None else 0.0,
            reasons=decision.reasons if decision is not None else ("gate_disabled",),
        )

    async def _force_wake(self) -> None:
        if self.gate is not None:
            await self.gate.force_wake()
            return
        async with self._wake_lock:
            self._wake_event.set()

    async def _wait_for_wake(self, timeout: float) -> bool:
        if self.gate is not None:
            return await self.gate.wait(timeout=timeout)
        async with self._wake_lock:
            event = self._wake_event
        with anyio.move_on_after(timeout) as cancel_scope:
            await event.wait()
        if cancel_scope.cancel_called:
            return False
        async with self._wake_lock:
            if self._wake_event is event:
                self._wake_event = anyio.Event()
        return True

    async def _has_idle_work(self) -> bool:
        if self.state_store is None or self.environment is None:
            return False
        state = await self.state_store.load()
        for candidate in await self.environment.candidates():
            if (
                candidate.directed_count
                or candidate.session_id in self.config.proactive_sessions
                or candidate.session_id in state.active_threads
            ):
                return True
        return False
