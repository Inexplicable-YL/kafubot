from typing import TYPE_CHECKING, Any, cast

from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field
from sekaibot.adapter.cqhttp.message import CQHTTPMessageSegment

from chat.agent.base import (
    ManagerContext,
    ManagerState,
    OutputMessage,
)
from chat.meme import search_memes

if TYPE_CHECKING:
    from chat.agent.base import ManagerContext, ManagerState


class SearchMemeInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    content: str = Field(description="你希望发送的表情包应该呈现出的内容。")


@tool(
    args_schema=SearchMemeInput,
    description="从表情包库搜索想要的表情包。",
)
async def search_meme(
    content: str,
    runtime: ToolRuntime,
) -> str:
    """调用reply工具实现对用户进行回复。"""
    _runtime = cast("ToolRuntime[ManagerContext, ManagerState]", runtime)

    def _gen_id():
        _runtime.state["meme_id"] += 1
        return _runtime.state["meme_id"]

    results = await search_memes(content)
    if not results:
        return "没有找到相关表情包"

    result_map = {_gen_id(): result for result in results}
    output = "搜索到以下表情包:\n\n"
    output += "\n".join(
        [f"[{i}]: {result.analysis}" for i, result in result_map.items()]
    )
    _runtime.state["search_meme_history"] |= result_map
    return output


class SendMemeInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    meme_id: int = Field(
        description="你希望发送的表情包的索引。索引与 `search_meme` 的输出对应，请参考 `search_meme` 的输出。"
    )


@tool(
    args_schema=SendMemeInput,
    description="向聊天中发送一条表情包。在发送表情包前，必须先使用 `search_meme` 搜索表情包。",
)
async def send_meme(
    meme_id: int,
    runtime: ToolRuntime,
) -> Any:
    _runtime = cast("ToolRuntime[ManagerContext, ManagerState]", runtime)
    result = _runtime.state["search_meme_history"].get(meme_id)
    if not result:
        return "没有找到相关表情包"
    await _runtime.context["node"].reply(
        CQHTTPMessageSegment.image(result.base64, sub_type=1)
    )
    return Command(
        update={
            "outputs": _runtime.state["outputs"] + [
                OutputMessage(
                    type="meme",
                    data={"content": result.analysis},
                )
            ],
            "messages": [
                ToolMessage(
                    content="已发送表情包：" + result.analysis,
                    tool_call_id=_runtime.tool_call_id,
                )
            ],
        }
    )
