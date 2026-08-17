import os
from collections.abc import Callable
from datetime import datetime
from typing import Any, cast

import aiosqlite
import anyio
from dotenv import load_dotenv
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    dynamic_prompt,
    wrap_model_call,
)
from langchain.agents.middleware.types import _CallableReturningSystemMessage
from langchain_core.messages import HumanMessage
from langchain_openai import OpenAIEmbeddings
from langgraph.store.sqlite import AsyncSqliteStore

from agent.base import (
    MODEL_VISIBLE_TZ,
    ManagerContext,
    ManagerState,
)
from agent.builder import create_agent
from agent.conversation import ConversationFrameMiddleware
from agent.effects import PendingReply, ReplyEffectMiddleware
from agent.extensions import (
    ActivateLimitMiddleware,
    MemeSendingMiddleware,
    query_expert,
    query_image,
    search_song,
    view_forward_message,
)
from agent.extensions.limiter import ActivateLimiterConfig
from agent.history import get_session_history
from agent.interaction import InteractionConfig, InteractionMiddleware
from agent.middlewares import (
    BehaviorLearnerMiddleware,
    ExpressionLearnerMiddleware,
    JargonLearnerMiddleware,
    LongMemoryMiddleware,
    SummarizationMiddleware,
)
from agent.models import get_nonthinking_model, get_thinking_model
from agent.prompts.manager import (
    BOT_NAME,
    IDENTITY,
    LANGUAGE_STYLE,
    MANAGER_PROMPT,
    MANAGER_USER_PROMPT,
    MANAGER_WITH_DECISION_PROMPT,
    MEME_PROMPT,
    SPECIAL_REMINDER,
    TOOL_PROMOT,
)
from agent.session import register_session_clearer, unregister_session_clearer
from agent.social_signals import SocialSignalAnalyzer, SocialSignalService
from agent.telemetry import close_social_telemetry
from agent.time_gate import TimeGateConfig, TimeGateMiddleware

load_dotenv()


MAX_TRUNS = 10
HISTORY_WINDOW = 20
wrap_model_call_async = cast("Any", wrap_model_call)


def manager_dynamic_prompt(
    fun: _CallableReturningSystemMessage[ManagerState, ManagerContext],
) -> AgentMiddleware[ManagerState, ManagerContext]:
    return dynamic_prompt(fun)


@manager_dynamic_prompt
def generate_prompt(request: ModelRequest[ManagerContext]) -> str:
    is_tome = request.runtime.context["is_tome"]
    if is_tome:
        return MANAGER_PROMPT.format(
            bot_name=BOT_NAME,
            identity=IDENTITY,
            language_style=LANGUAGE_STYLE,
            meme_prompt=MEME_PROMPT,
            tool_prompt=TOOL_PROMOT,
        )
    return MANAGER_WITH_DECISION_PROMPT.format(
        bot_name=BOT_NAME,
        identity=IDENTITY,
        special_reminder=SPECIAL_REMINDER,
        language_style=LANGUAGE_STYLE,
        meme_prompt=MEME_PROMPT,
        tool_prompt=TOOL_PROMOT,
    )


@wrap_model_call_async(state_schema=ManagerState)
async def add_time(
    request: ModelRequest[ManagerContext], handler: Callable
) -> ModelResponse:
    return await handler(
        request.override(
            messages=request.messages
            + [
                HumanMessage(
                    content=MANAGER_USER_PROMPT.format(
                        time=datetime.now(tz=MODEL_VISIBLE_TZ).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                    )
                )
            ]
        )
    )


# Backward-compatible public name: planner/analysis calls remain thinking by default.
get_model = get_thinking_model


async def create_agent_service():  # noqa: PLR0915
    database_conns: list[aiosqlite.Connection] = []
    jargon_middlewares: list[JargonLearnerMiddleware] = []
    memory_middlewares: list[LongMemoryMiddleware] = []
    summary_middlewares: list[SummarizationMiddleware] = []
    behavior_middlewares: list[BehaviorLearnerMiddleware] = []
    expression_middlewares: list[ExpressionLearnerMiddleware] = []
    interaction_middlewares: list[InteractionMiddleware] = []
    time_gate_middlewares: list[TimeGateMiddleware] = []
    conversation_middlewares: list[ConversationFrameMiddleware] = []
    effect_middlewares: list[ReplyEffectMiddleware] = []
    signal_services: list[SocialSignalService] = []
    memory_conn = await aiosqlite.connect(
        "./.database/long_memory.db",
        isolation_level=None,
    )
    database_conns.append(memory_conn)
    memory_store = AsyncSqliteStore(
        conn=memory_conn,
        index={
            "embed": OpenAIEmbeddings(
                model="text-embedding-3-large",
                base_url=os.getenv("OPENAI_BASE_URL"),
            ),
            "dims": 3072,
        },
    )
    jargon_conn = await aiosqlite.connect(
        "./.database/jargon_learner.db",
        isolation_level=None,
    )
    database_conns.append(jargon_conn)
    jargon_store = AsyncSqliteStore(conn=jargon_conn)
    summary_conn = await aiosqlite.connect(
        "./.database/summary_store.db",
        isolation_level=None,
    )
    database_conns.append(summary_conn)
    summary_store = AsyncSqliteStore(conn=summary_conn)

    async def get_agent(
        interaction_config: InteractionConfig,
        gate_config: TimeGateConfig,
        limiter_config: ActivateLimiterConfig,
    ):
        interaction_config = (
            InteractionConfig(
                reply_model=get_nonthinking_model(1.0),
                get_session_history=get_session_history,
            )
            | interaction_config
        )
        jargon_middleware = JargonLearnerMiddleware(
            analyze_model=get_nonthinking_model(0.1), store=jargon_store
        )
        jargon_middlewares.append(jargon_middleware)
        register_session_clearer(jargon_middleware.clear_session)
        summary_middleware = SummarizationMiddleware(
            summary_model=get_nonthinking_model(0.1), store=summary_store
        )
        summary_middlewares.append(summary_middleware)
        register_session_clearer(summary_middleware.clear_session)
        memory_middleware = LongMemoryMiddleware(
            analyze_model=get_thinking_model(reasoning_effort="high"),
            store=memory_store,
        )
        memory_middlewares.append(memory_middleware)
        register_session_clearer(memory_middleware.clear_session)
        signal_analyzer = SocialSignalAnalyzer(get_nonthinking_model(0.0))
        signal_service = SocialSignalService(signal_analyzer)
        signal_services.append(signal_service)
        conversation_middleware = ConversationFrameMiddleware(
            signal_service=signal_service
        )
        conversation_middlewares.append(conversation_middleware)
        effect_middleware = ReplyEffectMiddleware(signal_service=signal_service)
        effect_middlewares.append(effect_middleware)
        register_session_clearer(effect_middleware.clear_session)
        behavior_middleware = BehaviorLearnerMiddleware(
            analyze_model=get_thinking_model(reasoning_effort="high"),
        )
        behavior_middlewares.append(behavior_middleware)
        register_session_clearer(behavior_middleware.clear_session)
        expression_middleware = ExpressionLearnerMiddleware(
            analyze_model=get_nonthinking_model(0.2),
            selection_model=get_nonthinking_model(0.1),
            enable_precise_expression_selection=True,
        )
        expression_middlewares.append(expression_middleware)
        register_session_clearer(expression_middleware.clear_session)

        async def apply_reply_effect(
            pending: PendingReply, metrics: dict[str, Any], confidence: float
        ) -> None:
            source_ids = [
                str(item.get("message_id") or "")
                for item in metrics.get("attributed_followups", [])
                if item.get("message_id")
            ]
            async def apply_behavior_effect() -> None:
                await behavior_middleware.apply_observable_effect(
                    behavior_ids=pending.selected_behavior_ids,
                    session_id=pending.session_id,
                    metrics=metrics,
                    confidence=confidence,
                    source_ids=source_ids,
                )

            async def apply_expression_effect() -> None:
                await expression_middleware.apply_observable_effect(
                    expression_ids=pending.selected_expression_ids,
                    metrics=metrics,
                    confidence=confidence,
                )

            async with anyio.create_task_group() as task_group:
                task_group.start_soon(apply_behavior_effect)
                task_group.start_soon(apply_expression_effect)

        effect_middleware.add_effect_observer(apply_reply_effect)
        interaction_middleware = InteractionMiddleware(**interaction_config)
        interaction_middlewares.append(interaction_middleware)
        time_gate_middleware = TimeGateMiddleware(**gate_config)
        time_gate_middlewares.append(time_gate_middleware)
        return create_agent(
            model=get_thinking_model(reasoning_effort="max"),
            tools=[query_image, search_song, view_forward_message, query_expert],
            middleware=[
                interaction_middleware,
                conversation_middleware,
                effect_middleware,
                ActivateLimitMiddleware(**limiter_config),
                time_gate_middleware,
                MemeSendingMiddleware(man_send_per_turn=1),
                jargon_middleware,
                memory_middleware,
                summary_middleware,
                behavior_middleware,
                expression_middleware,
                generate_prompt,
                add_time,
            ],
            state_schema=ManagerState,
            context_schema=ManagerContext,
        )

    async def shutdown():
        for interaction_middleware in interaction_middlewares:
            unregister_session_clearer(interaction_middleware.clear_session)
        for time_gate_middleware in time_gate_middlewares:
            unregister_session_clearer(time_gate_middleware.clear_session)
        # Finalize observable reply effects while their behavior/expression
        # observers and databases are still live.
        for signal_service in signal_services:
            await signal_service.aclose()
        for effect_middleware in effect_middlewares:
            await effect_middleware.aclose()
            unregister_session_clearer(effect_middleware.clear_session)
        for jargon_middleware in jargon_middlewares:
            await jargon_middleware.aclose()
            unregister_session_clearer(jargon_middleware.clear_session)
        for memory_middleware in memory_middlewares:
            await memory_middleware.aclose()
        for summary_middleware in summary_middlewares:
            await summary_middleware.aclose()
            unregister_session_clearer(summary_middleware.clear_session)
        for behavior_middleware in behavior_middlewares:
            await behavior_middleware.aclose()
            unregister_session_clearer(behavior_middleware.clear_session)
        for expression_middleware in expression_middlewares:
            await expression_middleware.aclose()
            unregister_session_clearer(expression_middleware.clear_session)
        for memory_middleware in memory_middlewares:
            unregister_session_clearer(memory_middleware.clear_session)
        for conversation_middleware in conversation_middlewares:
            await conversation_middleware.aclose()
            unregister_session_clearer(conversation_middleware.clear_session)
        for database_conn in database_conns:
            await database_conn.close()
        await close_social_telemetry()

    return get_agent, shutdown
