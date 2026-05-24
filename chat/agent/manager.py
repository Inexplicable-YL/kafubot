import math
import os
from datetime import UTC, datetime
from functools import cache
from itertools import groupby
from typing import Any, Literal, cast

import pandas as pd
from dotenv import load_dotenv
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    before_agent,
    dynamic_prompt,
    wrap_model_call,
    wrap_tool_call,
)
from langchain.agents.middleware.types import _CallableReturningSystemMessage
from langchain_core.messages import AIMessage, HumanMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.runtime import Runtime
from pydantic import TypeAdapter
from scipy import stats

from chat.agent.base import (
    MODEL_VISIBLE_TZ,
    ManagerContext,
    ManagerState,
    OutputMessage,
    UserMessage,
)
from chat.agent.builder import create_agent
from chat.agent.history import get_session_history
from chat.agent.interaction import InteractionMiddleware
from chat.agent.logs import log_model_response, log_tool_io
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
from chat.agent.tools import TOOLS
from chat.utils import content_to_text, terminal_trend

load_dotenv()

MAX_TRUNS = 10
MODEL_NAME = "deepseek-v4-flash"
HISTORY_WINDOW = 20
wrap_model_call_async = cast("Any", wrap_model_call)
wrap_tool_call_async = cast("Any", wrap_tool_call)


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
        "early_messages": early_messages,
        "full_messages": history_messages + currents,
        "user_map": user_map,
    }


@before_agent(state_schema=ManagerState, can_jump_to=["end"])
async def calculate_params(
    state: ManagerState,
    runtime: Runtime[ManagerContext],
) -> dict[str, Any]:
    def _ttest_signal(statistic: Any) -> float:
        value = float(statistic)
        if math.isnan(value):
            return 0.0
        return -(math.tanh(value) if value > 0 else value)

    def get_pending_count(lst: list[bool]) -> tuple[list[int], int]:
        t = [i for i, v in enumerate(lst) if v]
        if not t:
            return [], len(lst)
        a, b = t[0], t[-1]
        return [sum(1 for _ in g) for k, g in groupby(lst[a : b + 1]) if not k], len(
            lst
        ) - b - 1

    ai_reply: list[bool] = []
    meme_reply: list[bool] = []
    reply_counts: list[int] = []
    msg_times: list[float] = []
    meme_count: int = 0
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
                meme_count += 1
            if dt := message.additional_kwargs.get("created_at"):
                dt = cast("datetime", dt)
                msg_times.append(dt.astimezone(UTC).timestamp())
        else:
            ai_reply.append(False)
    if not runtime.context["is_tome"]:
        recent_intervals = [
            msg_times[i] - msg_times[i - 1] for i in range(1, len(msg_times))
        ]
        if len(recent_intervals) > 1:
            interval_statistic, _ = cast(
                "tuple[Any, Any]",
                stats.ttest_1samp(
                    recent_intervals,
                    popmean=datetime.now(UTC).timestamp() - msg_times[-1],
                ),
            )
            interval = _ttest_signal(interval_statistic)
        else:
            interval = 2.0
        pending_counts, latest_pending = get_pending_count(ai_reply)
        if len(pending_counts) > 1:
            pending_statistic, _ = cast(
                "tuple[Any, Any]",
                stats.ttest_1samp(
                    pending_counts,
                    popmean=latest_pending,
                ),
            )
            pending = _ttest_signal(pending_statistic)
        else:
            pending = 2.0
        equivalent_pending = latest_pending + (
            7
            * math.tanh((pending + interval) / 4)
            / (16 * runtime.context["talk_value"])
        )
        if equivalent_pending <= 1 / runtime.context["talk_value"]:
            return {
                "jump_to": "end",
                "outputs": [
                    OutputMessage(
                        type="finish", data={"reason": "restricted by talk_value."}
                    )
                ],
            }

    real_average_count = float(
        pd.Series(reply_counts).ewm(alpha=0.5).mean().iloc[-1]
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


@cache
def get_agent(
    reasoning_effort: Literal["high", "max"] = "max",
):
    return create_agent(
        model=get_model(reasoning_effort),
        tools=TOOLS,
        middleware=[
            handle_input,
            calculate_params,
            generate_prompt,
            log_model_response,
            log_tool_io,
            InteractionMiddleware(),
            add_user_prompt,
        ],
        context_schema=ManagerContext,
    )
