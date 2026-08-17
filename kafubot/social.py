from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from inspect import isawaitable
from typing import TYPE_CHECKING, Any, Protocol, cast
from zoneinfo import ZoneInfo

import anyio
from langchain.agents import AgentState
from langchain_core.messages import AIMessage
from langgraph.errors import GraphRecursionError

from agent.base import UserMessage
from agent.builder import create_agent
from agent.message import QQMessage, QQMessageSegment
from agent.models import get_thinking_model
from agent.multimodal.image import ImageReadResult, get_analyzer, read_image
from kafubot.actions import QQActions
from kafubot.agency import (
    AttentionScheduler,
    ContextCompiler,
    ExecutiveMiddleware,
    GlobalGate,
    Replyer,
    SelfStateStore,
    SocialEnvironment,
    WorldModelHub,
)
from kafubot.agency.models import RoundResult, SelfState, SocialHome
from kafubot.agency.providers import ConversationWorldProvider, ProviderCommit
from kafubot.log import logger

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.language_models.chat_models import BaseChatModel

    from kafubot.adapters.cqhttp.event import (
        GroupMessageEvent,
        PrivateMessageEvent,
    )
    from kafubot.agency.environment import ConversationState
    from kafubot.config import AgentConfig

DEFAULT_USERNAME = "陌生用户"


@dataclass(slots=True)
class SocialAgentContext:
    """One create_agent invocation, owned and managed by the social runtime."""

    home: SocialHome
    state: SelfState
    provider_peek: list[str]
    budget: int
    opened: dict[str, list[int]] = field(default_factory=dict)
    open_session_id: str | None = None
    open_context_summaries: dict[str, str] = field(default_factory=dict)
    focus_stack: list[str] = field(default_factory=list)
    claimed: set[str] = field(default_factory=set)
    handled: set[str] = field(default_factory=set)
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
    """State carried by the replaceable social-agent middleware graph."""


class ImageAnalyzer(Protocol):
    async def ainvoke(self, value: dict[str, Any]) -> str: ...


def _has_model_visible_content(message: UserMessage) -> bool:
    return any(
        bool(segment.data.get("text", "").strip())
        if segment.type == "text"
        else segment.type not in {"image", "meme"} or bool(segment.data.get("content"))
        for segment in message.message
    )


class SocialAgentRuntime:
    """Owns the global create_agent graph and its replaceable plugin stack."""

    def __init__(
        self,
        config: AgentConfig,
        *,
        environment: SocialEnvironment | None = None,
        state_store: SelfStateStore | None = None,
        middleware: Sequence[
            AgentMiddleware[SocialAgentState, SocialAgentContext, None]
        ]
        | None = None,
        model_factory: Callable[[], BaseChatModel] | None = None,
        image_analyzer_factory: Callable[[], ImageAnalyzer] | None = None,
    ) -> None:
        self.config = config
        self.environment = environment or SocialEnvironment(
            max_entries=config.environment_message_limit
        )
        self.state_store = state_store or SelfStateStore(
            config.executive.state_file,
            recent_action_limit=config.executive.recent_action_limit,
        )
        self.gate = GlobalGate(
            config.gate,
            proactive_sessions=config.proactive_sessions,
            talk_value=config.talk_value,
            keywords=set(config.reply_keywords),
        )
        self.attention = AttentionScheduler(config.attention)
        self.providers = WorldModelHub([ConversationWorldProvider(self.environment)])
        self.compiler = ContextCompiler(
            self.environment,
            self.providers,
            config.executive,
        )
        self.replyer = Replyer()
        self.middleware = (
            tuple(middleware)
            if middleware is not None
            else (
                ExecutiveMiddleware(
                    self.environment,
                    self.compiler,
                    self.replyer,
                    self.providers,
                    self.state_store,
                    config.executive,
                ),
            )
        )
        self.model_factory = model_factory or (
            lambda: get_thinking_model(reasoning_effort="max")
        )
        self._model: BaseChatModel | None = None
        self._agent: Any | None = None
        self.image_limiter = anyio.CapacityLimiter(config.image_analyzer_workers)
        factory = image_analyzer_factory or (lambda: get_analyzer(True))
        self.image_analyzer = factory()
        self._ingest_locks: defaultdict[str, anyio.Lock] = defaultdict(anyio.Lock)
        self._closed = False

    @property
    def model(self) -> BaseChatModel:
        if self._model is None:
            self._model = self.model_factory()
        return self._model

    @property
    def agent(self) -> Any:
        if self._agent is None:
            self._agent = create_agent(
                model=self.model,
                tools=[],
                middleware=self.middleware,
                state_schema=SocialAgentState,
                context_schema=SocialAgentContext,
                name="social_agent",
            )
        return self._agent

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self.gate.force_wake()
        await self._close_middleware()
        await self.providers.aclose()

    async def run_round(self) -> RoundResult:
        state = await self.state_store.cool_fatigue()
        home = self.attention.rank(
            await self.environment.candidates(),
            state,
            limit=self.config.executive.home_session_limit,
        )
        result = RoundResult(home_size=len(home.items))
        if not home.items:
            return result

        context = SocialAgentContext(
            home=home,
            state=state,
            provider_peek=await self.providers.peek(home, state),
            budget=self.config.executive.focus_budget,
        )
        try:
            output: dict[str, Any] = await self.agent.ainvoke(
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
        while not self._closed:
            woke = await self.gate.wait(
                timeout=self.config.gate.idle_wake_seconds,
            )
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
        return await self.environment.session(session_id)

    async def clear_session(self, session_id: str, actions: QQActions) -> None:
        await self.environment.clear(session_id)
        await self.state_store.clear_session(session_id)
        await self.providers.commit(
            ProviderCommit(operation="clear", session_id=session_id)
        )
        await actions.reply("[SYSTEM] 已清除当前会话历史")

    async def get_image(
        self,
        actions: QQActions,
        file: str,
    ) -> ImageReadResult | None:
        try:
            result: dict[str, str] = await actions.call_api("get_image", file=file)
            if path := result.get("file"):
                return await read_image(path, result.get("url"))
        except Exception:
            logger.exception("Failed to read QQ image", file=file)
        return None

    async def to_user_message(
        self,
        event: GroupMessageEvent | PrivateMessageEvent,
        session: ConversationState,
        actions: QQActions,
    ) -> UserMessage | None:
        message = await QQMessage.from_cqhttp_message(
            event.message,
            list(session.entries),
            lambda file: self.get_image(actions, file),
        )
        if event.reply and event.reply.time:
            time_text = (
                datetime.fromtimestamp(event.reply.time, tz=UTC)
                .astimezone(ZoneInfo("Asia/Shanghai"))
                .strftime("%Y-%m-%d %H:%M:%S")
            )
            message = (
                QQMessageSegment.reply(
                    time_text,
                    str(event.reply.message_id),
                    include={"message_id"},
                )
                + message
            )
        if not message:
            return None

        images: list[tuple[ImageReadResult, bool]] = []
        for segment in message:
            if segment.type in {"image", "meme"} and not segment.data.get("content"):
                image_data = segment.data.get("image")
                if image_data is not None:
                    images.append(
                        (cast("ImageReadResult", image_data), segment.type == "meme")
                    )

        return UserMessage(
            role="user",
            timestamp=datetime.fromtimestamp(event.time, tz=UTC),
            user=event.sender.card or event.sender.nickname or DEFAULT_USERNAME,
            message=message,
            user_id=str(event.user_id),
            message_id=str(event.message_id),
            is_tome=event.is_tome(),
            images=images,
            chat_type=event.message_type,
        )

    async def fill_images(self, message: UserMessage) -> UserMessage | None:
        analyses: list[str | None] = [None] * len(message.images)

        async def analyze_one(
            result_index: int,
            image: ImageReadResult,
            as_meme: bool,
        ) -> None:
            try:
                async with self.image_limiter:
                    analyses[result_index] = await self.image_analyzer.ainvoke(
                        {
                            "image": image.base64,
                            "phash": image.phash,
                            "as_meme": as_meme,
                        }
                    )
            except Exception:
                logger.exception("Failed to analyze incoming image")

        async with anyio.create_task_group() as task_group:
            for index, (image, as_meme) in enumerate(message.images):
                task_group.start_soon(analyze_one, index, image, as_meme)

        results = iter(analyses)
        rebuilt = QQMessage()
        for segment in message.message:
            if segment.type in {"image", "meme"} and not segment.data.get("content"):
                content = next(results, None)
                if content:
                    rebuilt += getattr(QQMessageSegment, segment.type)(content=content)
                continue
            rebuilt += segment
        message.message = rebuilt
        return message if _has_model_visible_content(message) else None

    async def handle(
        self,
        event: GroupMessageEvent | PrivateMessageEvent,
    ) -> None:
        """Commit one accepted chat event; never invoke the executive directly."""
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
            session = await self.environment.session(session_id)
            message = await self.to_user_message(event, session, actions)
            if message is None:
                return
            if message.images:
                message = await self.fill_images(message)
            if message is None:
                return
            candidate = await self.environment.observe(session_id, message, actions)
        state = await self.state_store.load()
        decision = await self.gate.consider(
            session_id,
            message,
            candidate,
            state,
        )
        logger.debug(
            "Global Gate evaluated event",
            session_id=session_id,
            wake=decision.wake,
            score=decision.score,
            reasons=decision.reasons,
        )

    async def _close_middleware(self) -> None:
        closed: set[int] = set()
        for plugin in reversed(self.middleware):
            identity = id(plugin)
            if identity in closed:
                continue
            closed.add(identity)
            close = getattr(plugin, "aclose", None)
            if not callable(close):
                continue
            try:
                result = close()
                if isawaitable(result):
                    await result
            except Exception:
                logger.exception(
                    "Agent middleware shutdown failed",
                    middleware=type(plugin).__name__,
                )

    async def _has_idle_work(self) -> bool:
        state = await self.state_store.load()
        for candidate in await self.environment.candidates():
            if (
                candidate.directed_count
                or candidate.session_id in self.config.proactive_sessions
                or candidate.session_id in state.active_threads
            ):
                return True
        return False
