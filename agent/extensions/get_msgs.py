from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
)
from langchain.agents.middleware.types import (
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AIMessage, AnyMessage
from pydantic import BaseModel, ConfigDict, Field

from agent.base import ManagerContext, ManagerState


class GetEarlyMessagesInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    limit: int = Field(default=20, description="要显示的消息数量限制。", gt=1, le=20)
    runtime: ToolRuntime = Field(exclude=True)


class ContextAcquisitionMiddleware(AgentMiddleware[ManagerState, ManagerContext]):
    def __init__(self):
        self.display_messages: dict[str, list[AnyMessage]] = defaultdict(list)
        self.tools = [
            tool(
                "get_early_messages",
                args_schema=GetEarlyMessagesInput,
                description="获取比当前可见的消息更早的消息，并自动添加到上下文中。需要使用`limit`限制显示的消息数量。",
            )(self._get_early_messages),
        ]

    def _get_early_messages(
        self,
        limit: int,
        runtime: ToolRuntime[ManagerContext, ManagerState],
    ) -> Any:
        if not runtime.state["early_messages"]:
            return "没有比当前可见的消息更早的消息。"
        session_id = runtime.context["session_id"]
        message_count = 0
        for msg in runtime.state["messages"]:
            if isinstance(msg, ToolMessage) and (
                count := int(msg.additional_kwargs.get("get_message_count", 0))
            ):
                message_count += count
        self.display_messages[session_id] = runtime.state["early_messages"][
            -(limit + message_count) :
        ]
        get_message_count = len(self.display_messages[session_id]) - message_count
        return ToolMessage(
            content=f"获取到 {get_message_count} 条比当前可见的消息更早的消息。消息已添加到上下文中。",
            tool_call_id=runtime.tool_call_id,
            additional_kwargs={"get_message_count": get_message_count},
        )

    async def awrap_model_call(
        self,
        request: ModelRequest[ManagerContext],
        handler: Callable[
            [ModelRequest[ManagerContext]], Awaitable[ModelResponse[Any]]
        ],
    ) -> ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]:
        return await handler(
            request.override(
                messages=self.display_messages[request.runtime.context["session_id"]]
                + request.messages
            )
        )
