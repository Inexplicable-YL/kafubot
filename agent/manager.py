import os
from collections.abc import Callable
from datetime import datetime
from functools import cache
from typing import Any, Literal, cast

import aiosqlite
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
from langchain_deepseek.chat_models import DEFAULT_API_BASE, ChatDeepSeek
from langchain_openai import OpenAIEmbeddings
from langgraph.store.sqlite import AsyncSqliteStore

from agent.base import (
    MODEL_VISIBLE_TZ,
    ManagerContext,
    ManagerState,
)
from agent.builder import create_agent
from agent.extensions import (
    ActivateLimitMiddleware,
    LongMemoryMiddleware,
    MemeSendingMiddleware,
    query_image,
    search_song,
    view_forward_message,
)
from agent.extensions.limiter import ActivateLimiterConfig
from agent.history import get_session_history
from agent.interaction import InteractionConfig, InteractionMiddleware
from agent.middlewares import (
    JargonLearnerMiddleware,
    SummarizationMiddleware,
)
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
from agent.time_gate import TimeGateConfig, TimeGateMiddleware

load_dotenv()


MAX_TRUNS = 10
MODEL_NAME = "deepseek-v4-flash"
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


@cache
def get_model(
    temperature: float = 0.8,
    reasoning_effort: Literal["high", "max"] = "high",
) -> ChatDeepSeek:
    if reasoning_effort == "max":
        return ChatDeepSeek(
            model=MODEL_NAME,
            api_base=os.getenv("DEEPSEEK_BASE_URL", DEFAULT_API_BASE),
            temperature=temperature,
            max_retries=2,
            reasoning_effort="max",
            extra_body={
                "thinking": {
                    "type": "enabled",
                }
            },
        )
    return ChatDeepSeek(
        model=MODEL_NAME,
        api_base=os.getenv("DEEPSEEK_BASE_URL", DEFAULT_API_BASE),
        temperature=temperature,
        max_retries=2,
        reasoning_effort="high",
        extra_body={
            "thinking": {
                "type": "enabled",
            }
        },
    )


async def create_agent_service():
    jargon_middlewares: list[JargonLearnerMiddleware] = []
    memory_middlewares: list[LongMemoryMiddleware] = []
    summary_middlewares: list[SummarizationMiddleware] = []
    conn = await aiosqlite.connect(
        "./.database/long_memory.db",
        isolation_level=None,
    )
    store = AsyncSqliteStore(
        conn=conn,
        index={
            "embed": OpenAIEmbeddings(
                model="text-embedding-3-large",
                base_url=os.getenv("OPENAI_BASE_URL"),
            ),
            "dims": 3072,
        },
    )

    async def get_agent(
        interaction_config: InteractionConfig,
        gate_config: TimeGateConfig,
        limiter_config: ActivateLimiterConfig,
    ):
        interaction_config = (
            InteractionConfig(
                reply_model=get_model(1.2, "high"),
                get_session_history=get_session_history,
            )
            | interaction_config
        )
        jargon_middleware = JargonLearnerMiddleware(analyze_model=get_model(0.3, "max"))
        jargon_middlewares.append(jargon_middleware)
        summary_middleware = SummarizationMiddleware(
            summary_model=get_model(0.3, "max")
        )
        summary_middlewares.append(summary_middleware)
        memory_middleware = LongMemoryMiddleware(analyze_model=get_model(0.3, "max"))
        memory_middlewares.append(memory_middleware)
        return create_agent(
            model=get_model(0.6, "max"),
            tools=[query_image, search_song, view_forward_message],
            middleware=[
                InteractionMiddleware(**interaction_config),
                ActivateLimitMiddleware(**limiter_config),
                TimeGateMiddleware(**gate_config),
                MemeSendingMiddleware(man_send_per_turn=1),
                jargon_middleware,
                # memory_middleware,
                summary_middleware,
                generate_prompt,
                add_time,
            ],
            state_schema=ManagerState,
            context_schema=ManagerContext,
            store=store,
        )

    async def shutdown():
        for jargon_middleware in jargon_middlewares:
            await jargon_middleware.aclose()
        for memory_middleware in memory_middlewares:
            await memory_middleware.aclose()
        for summary_middleware in summary_middlewares:
            await summary_middleware.aclose()
        await conn.close()

    return get_agent, shutdown
