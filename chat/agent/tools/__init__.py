from typing import Any, cast

from dotenv import load_dotenv
from langchain.agents.middleware import wrap_model_call
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command

from chat.agent.base import OutputMessage
from chat.agent.tools.get_msgs import get_early_messages
from chat.agent.tools.meme import search_meme, send_meme
from chat.agent.tools.query_image import query_image
from chat.agent.tools.reply import reply
from chat.agent.tools.search_song import search_song
from chat.agent.tools.view_msg import view_forward_message

load_dotenv()
wrap_model_call_async = cast("Any", wrap_model_call)


@tool(
    description="结束本轮思考，等待后续新的外部消息再继续。或者本轮不进行任何动作，等待其他用户的新消息；也用于用户可能还没说完、需要先把发言权交还给用户的场景。"
)
def finish(runtime: ToolRuntime) -> Command:
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


TOOLS = [
    reply,
    finish,
    search_meme,
    send_meme,
    get_early_messages,
    query_image,
    search_song,
    view_forward_message,
]
