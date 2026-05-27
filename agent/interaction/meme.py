import itertools
from typing import Any

from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AIMessage
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field
from sekaibot.adapter.cqhttp.message import CQHTTPMessageSegment

from agent.base import (
    ManagerContext,
    ManagerState,
    OutputMessage,
)
from agent.history import get_session_history
from agent.message import QQMessageSegment
from agent.multimodal.meme import MemeResult, search_memes


class SearchMemeInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    content: str = Field(description="你希望发送的表情包应该呈现出的内容。")


@tool(
    args_schema=SearchMemeInput,
    description="从表情包库搜索想要的表情包。",
)
async def search_meme(
    content: str,
    runtime: ToolRuntime[ManagerContext, ManagerState],
) -> Any:
    """从表情包库搜索想要的表情包。"""
    results = await search_memes(content)
    if not results:
        return "未找到匹配的表情包，请尝试更完整详细的搜索语句。"

    def _gen_id() -> str:
        runtime.state.setdefault("meme_id", itertools.count(0))
        assert "meme_id" in runtime.state
        return str(next(runtime.state["meme_id"]))

    found_meme_dict: dict[str, MemeResult] = {}
    for msg in runtime.state["messages"]:
        if (
            isinstance(msg, ToolMessage)
            and (meme_dict := msg.additional_kwargs.get("search_meme_results", {}))
            and isinstance(meme_dict, dict)
        ):
            found_meme_dict.update(meme_dict)
    found_meme_analysis = {r.analysis: k for k, r in found_meme_dict.items()}

    newly_result_map, old_result_map = {}, {}
    for result in results:
        if result.analysis in found_meme_analysis:
            old_result_map[found_meme_analysis[result.analysis]] = result
        else:
            newly_result_map[_gen_id()] = result

    result_map: dict[str, MemeResult] = {**old_result_map, **newly_result_map}

    content_lines: list[str] = [
        f"已找到 {len(results)} 个匹配的表情包，请从下列中选择你希望发送的表情包：",
        *[
            f"- [{meme_id}]{'（本次新发现）' if meme_id in newly_result_map else '（此前已发现）'} {meme_result.analysis}"
            for meme_id, meme_result in result_map.items()
        ],
    ]
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content="\n".join(content_lines),
                    tool_call_id=runtime.tool_call_id,
                    additional_kwargs={"search_meme_results": newly_result_map},
                )
            ],
        }
    )


class SendMemeInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    meme_id: str = Field(
        description="你希望发送的表情包的索引。索引与 `search_meme` 的输出对应，请参考 `search_meme` 的输出。"
    )


@tool(
    args_schema=SendMemeInput,
    description="向聊天中发送一条表情包。在发送表情包前，必须先使用 `search_meme` 搜索表情包。",
)
async def send_meme(
    meme_id: str,
    runtime: ToolRuntime[ManagerContext, ManagerState],
) -> Any:
    """向聊天中发送一条表情包。"""
    normalized_meme_id = meme_id.replace("[", "").replace("]", "").casefold().strip()
    found_meme_dict: dict[str, MemeResult] = {}
    for msg in runtime.state["messages"]:
        if (
            isinstance(msg, ToolMessage)
            and (meme_dict := msg.additional_kwargs.get("search_meme_results", {}))
            and isinstance(meme_dict, dict)
        ):
            found_meme_dict.update(meme_dict)
    result = found_meme_dict.get(normalized_meme_id)
    if not result:
        return "无法通过索引找到对应的表情包，请检查索引是否正确。"
    await runtime.context["node"].reply(
        CQHTTPMessageSegment.image(result.base64, sub_type=1)
    )
    await get_session_history(runtime.context["session_id"]).aadd_message(
        AIMessage(
            content=QQMessageSegment.meme(
                content=result.analysis,
            ).get_msgcode(),
        )
    )
    return Command(
        update={
            "outputs": [OutputMessage(type="meme", data={"content": result.analysis})],
            "messages": [
                ToolMessage(
                    content=f"已发送表情包：[{meme_id}] " + result.analysis,
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )
