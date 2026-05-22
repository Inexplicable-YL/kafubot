from __future__ import annotations

import os
from datetime import datetime
from functools import cache
from typing import TYPE_CHECKING, Any, Literal, cast

from dotenv import load_dotenv
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    before_agent,
    before_model,
    dynamic_prompt,
    wrap_model_call,
    wrap_tool_call,
)
from langchain_core.messages import AIMessage, HumanMessage
from langchain_deepseek import ChatDeepSeek
from pydantic import TypeAdapter

from chat.agent.base import (
    MODEL_VISIBLE_TZ,
    ManagerContext,
    ManagerState,
    OutputMessage,
    UserMessage,
)
from chat.agent.builder import create_agent
from chat.agent.history import get_session_history
from chat.agent.logs import log_model_response, log_tool_io
from chat.agent.prompt import (
    IDENTITY,
    LANGUAGE_STYLE,
    MANAGER_PROMPT,
    MANAGER_WITH_DECISION_PROMPT,
    MEME_PROMPT,
    SPECIAL_REMINDER,
)
from chat.agent.tools import TOOLS
from chat.utils import content_to_text

if TYPE_CHECKING:
    from langchain.agents.middleware.types import _CallableReturningSystemMessage
    from langgraph.runtime import Runtime


load_dotenv()

MAX_TRUNS = 5
MODEL_NAME = "deepseek-v4-flash"
HISTORY_WINDOW = 20
wrap_model_call_async = cast("Any", wrap_model_call)
wrap_tool_call_async = cast("Any", wrap_tool_call)


def manager_dynamic_prompt(
    fun: _CallableReturningSystemMessage[ManagerState, ManagerContext],
) -> AgentMiddleware[ManagerState, ManagerContext]:
    return dynamic_prompt(fun)


@before_model(state_schema=ManagerState, can_jump_to=["end"])
def final(
    state: ManagerState, runtime: Runtime[ManagerContext]
) -> dict[str, Any] | None:
    _ = runtime
    if state["should_stop"]:
        return {
            "jump_to": "end",
        }
    if len([m for m in state["messages"] if isinstance(m, AIMessage)]) >= MAX_TRUNS:
        return {
            "outputs": state["outputs"]
            + [
                OutputMessage(
                    type="stop",
                    data={},
                )
            ],
            "jump_to": "end",
        }
    for msg in state["messages"]:
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            return {
                "outputs": state["outputs"]
                + [
                    OutputMessage(
                        type="stop",
                        data={},
                    )
                ],
                "jump_to": "end",
            }
    return None


@wrap_model_call_async(state_schema=ManagerState)
async def add_time(request: ModelRequest[ManagerContext], handler) -> ModelResponse:
    return await handler(
        request.override(
            messages=request.messages
            + [
                HumanMessage(
                    content=f"<time>\n{datetime.now(tz=MODEL_VISIBLE_TZ).strftime('%Y-%m-%d %H:%M:%S')}\n</time>"
                )
            ]
        )
    )


@manager_dynamic_prompt
def generate_prompt(request: ModelRequest[ManagerContext]) -> str:
    is_tome = request.runtime.context["is_tome"]
    if is_tome:
        return MANAGER_PROMPT.format(
            bot_name="可不",
            identity=IDENTITY,
            language_style=LANGUAGE_STYLE,
            meme_prompt=MEME_PROMPT,
        )
    return MANAGER_WITH_DECISION_PROMPT.format(
        bot_name="可不",
        identity=IDENTITY,
        special_reminder=SPECIAL_REMINDER,
        language_style=LANGUAGE_STYLE,
        meme_prompt=MEME_PROMPT,
    )


@before_agent
async def handle_input(
    state: ManagerState,
    runtime: Runtime[ManagerContext],
) -> dict[str, Any]:
    inputs = TypeAdapter(list[UserMessage]).validate_python(
        state.get("inputs") or state["messages"]
    )
    currents = [
        HumanMessage(
            content=item.as_content(timezone=MODEL_VISIBLE_TZ),
            additional_kwargs={"raw": item},
        )
        for item in inputs
    ]

    session_id = runtime.context.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session_id is required")
    histories = [
        HumanMessage(
            content=f"<bot-message user=可不>\n{content_to_text(message.content)}\n</bot-message>"
        )
        if isinstance(message, AIMessage)
        else message
        for message in await get_session_history(session_id).aget_messages()
    ]

    full_messages = histories + currents
    messages = full_messages[-HISTORY_WINDOW:]
    early_messages = full_messages[:-HISTORY_WINDOW]
    prompt_variables = {
        key: value
        for key, value in state.items()
        if key not in {"messages", "thinking", "reasoning_effort"}
    }
    return {
        **prompt_variables,
        "inputs": inputs,
        "reasoning_effort": effort
        if (effort := state.get("reasoning_effort", "high")) in {"high", "max"}
        else "high",
        "messages": messages,
        "early_messages": early_messages,
        "full_messages": full_messages,
    }


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


@wrap_model_call_async(state_schema=ManagerState)
async def dynamic_model_selection(
    request: ModelRequest[ManagerContext], handler
) -> ModelResponse:
    """Choose model based on conversation complexity."""
    reasoning_effort = cast("ManagerState", request.state)["reasoning_effort"]
    model = get_model(reasoning_effort=reasoning_effort)

    return await handler(request.override(model=model))


@cache
def get_agent():
    return create_agent(
        model=get_model(),
        tools=TOOLS,
        middleware=[
            handle_input,
            generate_prompt,
            dynamic_model_selection,
            log_model_response,
            log_tool_io,
            final,
            add_time,
        ],
        context_schema=ManagerContext,
    )
