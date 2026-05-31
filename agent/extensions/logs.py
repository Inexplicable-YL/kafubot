import json
import os
import warnings
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from anyio import open_file
from langchain.agents.middleware import (
    AgentMiddleware,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from agent.base import (
    MODEL_VISIBLE_TZ,
    ManagerContext,
    ManagerState,
)

warnings.filterwarnings(
    "ignore",
    message=r"Pydantic serializer warnings:[\s\S]*field_name='context'.*",
    category=UserWarning,
)

AGENT_LOG_TEXT_LIMIT = 8000


class AgentDebugLogMiddleware(AgentMiddleware[ManagerState, ManagerContext]):
    """Log model responses and tool I/O for the manager agent."""

    state_schema = ManagerState

    def __init__(
        self,
        *,
        log_path: str | os.PathLike[str] | Path,
        log_text_limit: int | None = None,
    ) -> None:
        super().__init__()
        self.log_path = Path(log_path)
        self.log_text_limit = (
            int(log_text_limit) if log_text_limit is not None else AGENT_LOG_TEXT_LIMIT
        )

    def _truncate(self, value: Any) -> Any:
        if isinstance(value, str):
            return value[: self.log_text_limit]
        return value

    async def _write_log(self, payload: dict[str, Any]) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        async with await open_file(self.log_path, "a", encoding="utf-8") as file:
            await file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    async def awrap_model_call(
        self,
        request: ModelRequest[ManagerContext],
        handler: Callable[
            [ModelRequest[ManagerContext]], Awaitable[ModelResponse[Any]]
        ],
    ) -> ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]:
        response = await handler(request)
        context = cast("ManagerContext", request.runtime.context)

        for message in response.result:
            if not isinstance(message, AIMessage):
                continue

            content = message.content
            reasoning_content = message.additional_kwargs.get("reasoning_content")

            await self._write_log(
                {
                    "time": datetime.now(tz=MODEL_VISIBLE_TZ).isoformat(),
                    "event": "model_response",
                    "session_id": context["session_id"],
                    "is_tome": context["is_tome"],
                    "content": self._truncate(content),
                    "reasoning_content": self._truncate(reasoning_content),
                    "tool_calls": message.tool_calls,
                }
            )

        return response

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        tool_call = request.tool_call
        context = cast("ManagerContext", request.runtime.context)

        await self._write_log(
            {
                "time": datetime.now(tz=MODEL_VISIBLE_TZ).isoformat(),
                "event": "tool_call",
                "session_id": context["session_id"],
                "is_tome": context["is_tome"],
                "tool": tool_call["name"],
                "tool_call_id": tool_call["id"],
                "args": tool_call["args"],
            }
        )

        try:
            result = await handler(request)
        except Exception as exc:
            await self._write_log(
                {
                    "time": datetime.now(tz=MODEL_VISIBLE_TZ).isoformat(),
                    "event": "tool_error",
                    "session_id": context["session_id"],
                    "is_tome": context["is_tome"],
                    "tool": tool_call["name"],
                    "tool_call_id": tool_call["id"],
                    "error_type": type(exc).__name__,
                    "error": self._truncate(str(exc)),
                }
            )
            raise

        if isinstance(result, ToolMessage):
            tool_return = {
                "type": "tool_message",
                "content": self._truncate(result.content),
                "status": result.status,
            }
        else:
            update = cast("dict[str, Any]", result.update)
            tool_return = {
                "type": "command",
                "stop_message": update.get("stop_message"),
                "messages": [
                    {
                        "content": self._truncate(message.content),
                        "status": message.status,
                    }
                    for message in update.get("messages", [])
                ],
            }

        await self._write_log(
            {
                "time": datetime.now(tz=MODEL_VISIBLE_TZ).isoformat(),
                "event": "tool_return",
                "session_id": context["session_id"],
                "is_tome": context["is_tome"],
                "tool": tool_call["name"],
                "tool_call_id": tool_call["id"],
                "result": tool_return,
            }
        )

        return result
