from __future__ import annotations

import json
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from anyio import open_file
from dotenv import load_dotenv
from langchain.agents.middleware import (
    ModelRequest,
    ModelResponse,
    wrap_model_call,
    wrap_tool_call,
)
from langchain_core.messages import AIMessage, ToolMessage

from chat.agent.base import (
    MODEL_VISIBLE_TZ,
    ManagerContext,
    ManagerState,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware import ToolCallRequest
    from langgraph.types import Command


load_dotenv()

warnings.filterwarnings(
    "ignore",
    message=r"Pydantic serializer warnings:[\s\S]*field_name='context'.*",
    category=UserWarning,
)

AGENT_LOG_PATH = Path(os.getenv("AGENT_LOG_PATH", ".logs/agent_debug.jsonl"))
AGENT_LOG_TEXT_LIMIT = int(os.getenv("AGENT_LOG_TEXT_LIMIT", "8000"))
wrap_model_call_async = cast("Any", wrap_model_call)
wrap_tool_call_async = cast("Any", wrap_tool_call)


@wrap_model_call_async(state_schema=ManagerState)
async def log_model_response(
    request: ModelRequest[ManagerContext],
    handler: Callable[[ModelRequest[ManagerContext]], Awaitable[ModelResponse]],
) -> ModelResponse:
    response = await handler(request)
    context = cast("ManagerContext", request.runtime.context)
    AGENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with await open_file(AGENT_LOG_PATH, "a", encoding="utf-8") as file:
        for message in response.result:
            if isinstance(message, AIMessage):
                content = message.content
                reasoning_content = message.additional_kwargs.get("reasoning_content")
                await file.write(
                    json.dumps(
                        {
                            "time": datetime.now(tz=MODEL_VISIBLE_TZ).isoformat(),
                            "event": "model_response",
                            "session_id": context["session_id"],
                            "is_tome": context["is_tome"],
                            "content": content[:AGENT_LOG_TEXT_LIMIT]
                            if isinstance(content, str)
                            else content,
                            "reasoning_content": reasoning_content[
                                :AGENT_LOG_TEXT_LIMIT
                            ]
                            if isinstance(reasoning_content, str)
                            else reasoning_content,
                            "tool_calls": message.tool_calls,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    return response


@wrap_tool_call_async
async def log_tool_io(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
) -> ToolMessage | Command[Any]:
    tool_call = request.tool_call
    context = cast("ManagerContext", request.runtime.context)
    AGENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with await open_file(AGENT_LOG_PATH, "a", encoding="utf-8") as file:
        await file.write(
            json.dumps(
                {
                    "time": datetime.now(tz=MODEL_VISIBLE_TZ).isoformat(),
                    "event": "tool_call",
                    "session_id": context["session_id"],
                    "is_tome": context["is_tome"],
                    "tool": tool_call["name"],
                    "tool_call_id": tool_call["id"],
                    "args": tool_call["args"],
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    try:
        result = await handler(request)
    except Exception as exc:
        async with await open_file(AGENT_LOG_PATH, "a", encoding="utf-8") as file:
            await file.write(
                json.dumps(
                    {
                        "time": datetime.now(tz=MODEL_VISIBLE_TZ).isoformat(),
                        "event": "tool_error",
                        "session_id": context["session_id"],
                        "is_tome": context["is_tome"],
                        "tool": tool_call["name"],
                        "tool_call_id": tool_call["id"],
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:AGENT_LOG_TEXT_LIMIT],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        raise
    if isinstance(result, ToolMessage):
        tool_return = {
            "type": "tool_message",
            "content": result.content[:AGENT_LOG_TEXT_LIMIT]
            if isinstance(result.content, str)
            else result.content,
            "status": result.status,
        }
    else:
        update = cast("dict[str, Any]", result.update)
        tool_return = {
            "type": "command",
            "stop_message": update.get("stop_message"),
            "messages": [
                {
                    "content": message.content[:AGENT_LOG_TEXT_LIMIT]
                    if isinstance(message.content, str)
                    else message.content,
                    "status": message.status,
                }
                for message in update.get("messages", [])
            ],
        }
    async with await open_file(AGENT_LOG_PATH, "a", encoding="utf-8") as file:
        await file.write(
            json.dumps(
                {
                    "time": datetime.now(tz=MODEL_VISIBLE_TZ).isoformat(),
                    "event": "tool_return",
                    "session_id": context["session_id"],
                    "is_tome": context["is_tome"],
                    "tool": tool_call["name"],
                    "tool_call_id": tool_call["id"],
                    "result": tool_return,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    return result
