from typing import Any, cast

from dotenv import load_dotenv
from langchain.agents.middleware import (
    ModelRequest,
    ModelResponse,
    wrap_model_call,
)
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command

from chat.agent.base import ManagerContext, ManagerState, OutputMessage

load_dotenv()
wrap_model_call_async = cast("Any", wrap_model_call)


@tool(description="结束本轮思考，等待后续新的外部消息再继续。")
def search_tool(runtime: ToolRuntime) -> Command:
    """调用finish工具实现结束对话。"""
    return Command(
        update={
            "should_stop": True,
            "outputs": [OutputMessage(type="finish", data={})],
            "messages": [
                ToolMessage(
                    content="当前 Planner 已结束本轮思考，等待新的群聊消息。",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@wrap_model_call_async
async def add_deferred_tools(
    request: ModelRequest[ManagerContext],
    handler,
) -> ModelResponse:
    state = cast("ManagerState", request.state)
    return await handler(
        request.override(tools=request.tools + cast("Any", state["deferred_tools"]))
    )
