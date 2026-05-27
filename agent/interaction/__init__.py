from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
    hook_config,
)
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

from agent.base import (
    ManagerContext,
    ManagerState,
    OutputMessage,
)

from .meme import search_meme, send_meme
from .replyer import reply


@tool(
    description="结束本轮思考，等待后续新的外部消息再继续。或者本轮不进行任何动作，等待其他用户的新消息；也用于用户可能还没说完、需要先把发言权交还给用户的场景。"
)
def finish(runtime: ToolRuntime) -> Command:
    """调用finish工具实现结束对话。"""
    return Command(
        update={
            "outputs": [OutputMessage(type="finish", data={})],
            "messages": [
                ToolMessage(
                    content="当前 Manager 已结束本轮思考，等待新的群聊消息。",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


class InteractionMiddleware(AgentMiddleware[ManagerState, ManagerContext, Any]):
    tools = [reply, finish, search_meme, send_meme]

    def __init__(
        self,
        max_turns: int = 10,
        max_tool_calls: int = 10,
        mas_retries: int = 3,
        **_: Any,
    ) -> None:
        self.max_turns = max_turns
        self.max_tool_calls = max_tool_calls
        self.mas_retries = mas_retries

    @hook_config(can_jump_to=["end"])
    async def abefore_model(
        self, state: ManagerState, runtime: Runtime[ManagerContext]
    ) -> dict[str, Any] | None:
        _ = runtime
        if any(o["type"] in {"finish", "stop", "reply"} for o in state["outputs"]):
            return {"jump_to": "end"}
        messgaes = state["messages"]
        if sum(isinstance(x, AIMessage) for x in messgaes) >= self.max_turns:
            reason = "model has reached the maximum number of rounds limit."
        elif sum(isinstance(x, ToolMessage) for x in messgaes) >= self.max_tool_calls:
            reason = "model has reached the maximum number of tool calls limit."
        elif any(isinstance(x, AIMessage) and not x.tool_calls for x in messgaes):
            reason = "model returned content but did not invoke the tool."
        else:
            return None
        return {
            "outputs": [OutputMessage(type="stop", data={"reason": reason})],
            "jump_to": "end",
        }

    async def awrap_model_call(
        self,
        request: ModelRequest[ManagerContext],
        handler: Callable[
            [ModelRequest[ManagerContext]], Awaitable[ModelResponse[Any]]
        ],
    ) -> ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]:
        resp = await handler(request)
        for _ in range(self.mas_retries - 1):
            last = resp.result[-1]
            if not (isinstance(last, AIMessage) and last.content):
                break
            resp = await handler(request)
        return resp
