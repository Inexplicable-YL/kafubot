from __future__ import annotations

from dotenv import load_dotenv
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command
from pydantic import BaseModel, Field

from chat.agent.base import ManagerContext, ManagerState, StopMessage
from chat.agent.query_image import query_image
from chat.agent.reply import reply
from chat.agent.search_song import search_song
from chat.utils import content_to_text

load_dotenv()


@tool(description="结束本轮思考，等待后续新的外部消息再继续。")
def finish(runtime: ToolRuntime[ManagerContext, ManagerState]) -> Command:
    """调用finish工具实现结束对话。"""
    return Command(
        update={
            "should_stop": True,
            "stop_message": StopMessage(type="finish", data={}),
            "messages": [
                ToolMessage(
                    content="当前 Planner 已结束本轮思考，等待新的群聊消息。",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool(
    description="本轮不进行任何动作，等待其他用户的新消息；也用于用户可能还没说完、需要先把发言权交还给用户的场景。"
)
def no_action(runtime: ToolRuntime[ManagerContext, ManagerState]) -> Command:
    """调用no_action工具实现结束对话。"""
    return Command(
        update={
            "should_stop": True,
            "stop_message": StopMessage(type="no_action", data={}),
            "messages": [
                ToolMessage(
                    content="当前 Planner 已结束本轮思考，等待新的群聊消息。",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


class GetEarlyMessagesInput(BaseModel):
    limit: int = Field(default=20, description="要显示的消息数量。", gt=1, le=40)


@tool(description="显示比当前可见的消息更早的消息。需要使用`limit`限制显示的消息数量。")
def get_early_messages(
    runtime: ToolRuntime[ManagerContext, ManagerState], limit: int = 20
) -> str:
    if not runtime.state["early_messages"]:
        return "没有比当前可见的消息更早的消息。"
    return (
        "以下是比当前可见的消息更早的消息：\n\n"
        + "\n".join(
            [
                content_to_text(m.content)
                for m in runtime.state["early_messages"][-limit:]
            ]
        )
        + "\n\n以上消息往后衔接当前可见的消息。"
    )


TOOLS = [reply, finish, no_action, get_early_messages, query_image, search_song]
