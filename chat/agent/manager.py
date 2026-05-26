import os
from datetime import datetime
from functools import cache
from typing import Any, Literal, cast

import aiosqlite
import pandas as pd
from async_lru import alru_cache
from dotenv import load_dotenv
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    before_agent,
    dynamic_prompt,
    wrap_model_call,
)
from langchain.agents.middleware.types import _CallableReturningSystemMessage
from langchain_core.messages import AIMessage, HumanMessage
from langchain_deepseek import ChatDeepSeek
from langchain_openai import OpenAIEmbeddings
from langgraph.runtime import Runtime
from langgraph.store.sqlite import AsyncSqliteStore
from pydantic import TypeAdapter

from chat.agent.base import (
    MODEL_VISIBLE_TZ,
    ManagerContext,
    ManagerState,
    UserMessage,
)
from chat.agent.builder import create_agent
from chat.agent.get_msgs import ContextAcquisitionMiddleware
from chat.agent.history import get_session_history
from chat.agent.interaction import InteractionMiddleware
from chat.agent.logs import AgentDebugLogMiddleware
from chat.agent.memory import LongMemoryMiddleware
from chat.agent.prompt import (
    BOT_NAME,
    IDENTITY,
    LANGUAGE_STYLE,
    LESS_MEME,
    MANAGER_PROMPT,
    MANAGER_USER_PROMPT,
    MANAGER_WITH_DECISION_PROMPT,
    MEME_PROMPT,
    MORE_MEME,
    SPECIAL_REMINDER,
    TOOL_PROMOT,
)
from chat.agent.time_gate import TimeGateMiddleware
from chat.agent.tools import TOOLS
from chat.utils import content_to_text, terminal_trend

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


@before_agent
async def handle_input(
    state: ManagerState,
    runtime: Runtime[ManagerContext],
) -> dict[str, Any]:
    inputs = TypeAdapter(list[UserMessage]).validate_python(
        state["inputs"] or state["messages"]
    )
    currents = [
        HumanMessage(
            content=item.as_content(),
            additional_kwargs={"raw": item},
        )
        for item in inputs
    ]
    session_id = runtime.context.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session_id is required")
    history_messages = await get_session_history(session_id).aget_messages()
    histories = [
        HumanMessage(
            content=f"<bot-message user=可不>\n{content_to_text(message.content)}\n</bot-message>"
        )
        if isinstance(message, AIMessage)
        else message
        for message in history_messages
    ]
    messages = (histories + currents)[-HISTORY_WINDOW:]
    early_messages = (histories + currents)[:-HISTORY_WINDOW]

    user_map = {msg.user: msg.user_id for msg in inputs} | {
        msg.user: msg.user_id
        for msg in [
            TypeAdapter(UserMessage).validate_python(msg.additional_kwargs.get("raw"))
            for msg in history_messages
            if isinstance(msg, HumanMessage) and msg.additional_kwargs.get("raw")
        ]
    }
    prompt_variables = {
        key: value for key, value in state.items() if key not in {"messages"}
    }
    return {
        **prompt_variables,
        "inputs": inputs,
        "messages": messages,
        "history_messages": history_messages,
        "current_messages": currents,
        "early_messages": early_messages,
        "full_messages": history_messages + currents,
        "user_map": user_map,
    }


@before_agent(state_schema=ManagerState)
async def hardness(
    state: ManagerState,
    runtime: Runtime[ManagerContext],
) -> dict[str, Any]:
    ai_reply: list[bool] = []
    meme_reply: list[bool] = []
    reply_counts: list[int] = []
    for message in state["full_messages"]:
        if isinstance(message, AIMessage):
            ai_reply.append(True)
            if "MSG:meme" not in message.content:
                meme_reply.append(False)
                reply_counts.append(
                    len(content_to_text(message.content).strip().split("\n"))
                )
            else:
                meme_reply.append(True)
            if dt := message.additional_kwargs.get("created_at"):
                dt = cast("datetime", dt)
        else:
            ai_reply.append(False)
    real_average_count = float(
        pd.Series(reply_counts).ewm(alpha=0.1).mean().iloc[-1]
    ) * (1 + 0.5 * terminal_trend(ai_reply))
    print(real_average_count, runtime.context["average_reply_count"])
    real_meme_ratio = (
        sum(meme_reply)
        / (len(meme_reply) - sum(meme_reply))
        * (1 + 0.5 * terminal_trend(meme_reply))
    )
    print(real_meme_ratio, runtime.context["meme_reply_ratio"])
    return {
        "real_average_count": real_average_count,
        "real_meme_ratio": real_meme_ratio,
    }


@wrap_model_call_async(state_schema=ManagerState)
async def add_user_prompt(
    request: ModelRequest[ManagerContext], handler
) -> ModelResponse:
    state = cast("ManagerState", request.state)
    if state.get("real_meme_ratio", 0.6) > request.runtime.context["meme_reply_ratio"]:
        meme_style = LESS_MEME
    else:
        meme_style = MORE_MEME
    return await handler(
        request.override(
            messages=request.messages
            + [
                HumanMessage(
                    content=MANAGER_USER_PROMPT.format(
                        time=datetime.now(tz=MODEL_VISIBLE_TZ).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        meme_style=meme_style,
                    )
                )
            ]
        )
    )


@cache
def get_model(
    reasoning_effort: Literal["high", "max"] = "high",
) -> ChatDeepSeek:
    if reasoning_effort == "max":
        return ChatDeepSeek(
            model=MODEL_NAME,
            base_url=os.getenv("DEEPSEEK_BASE_URL"),
            temperature=1.2,
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
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=1.2,
        max_retries=2,
        reasoning_effort="high",
        extra_body={
            "thinking": {
                "type": "enabled",
            }
        },
    )


async def create_agent_service():
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

    @alru_cache(maxsize=2)
    async def get_agent(
        talk_value: float,
        keywords: tuple[tuple[str, float]],
        reasoning_effort: Literal["high", "max"] = "max",
    ):
        return create_agent(
            model=get_model(reasoning_effort),
            tools=TOOLS,
            middleware=[
                handle_input,
                TimeGateMiddleware(
                    config={"talk_value": talk_value, "keywords": list(keywords)}
                ),
                hardness,
                generate_prompt,
                AgentDebugLogMiddleware(
                    log_path=".logs/agent_debug.jsonl",
                    log_text_limit=1000,
                ),
                InteractionMiddleware(),
                ContextAcquisitionMiddleware(),
                LongMemoryMiddleware(
                    use_subagent=True,
                    subagent_model=get_model(reasoning_effort),
                ),
                add_user_prompt,
            ],
            state_schema=ManagerState,
            context_schema=ManagerContext,
            store=store,  # ← 闭包捕获，生命周期与 conn 绑定
        )

    async def shutdown():
        await conn.close()

    return get_agent, shutdown
