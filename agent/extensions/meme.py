from collections import defaultdict, deque
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field
from sekaibot.adapter.cqhttp.message import CQHTTPMessageSegment

from agent.base import (
    ManagerContext,
    ManagerState,
    OutputMessage,
)
from agent.multimodal.meme import MemeResult, search_memes

KEEP_SEND_HISTORY = 3


class SearchMemeInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    content: str = Field(description="你希望发送的表情包应该呈现出的内容。")
    limit: int = Field(default=10, description="要显示的表情包数量。", gt=1, le=20)


class SendMemeInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    meme_id: str = Field(
        description="你希望发送的表情包的索引。索引与 `search_meme` 的输出对应，请参考 `search_meme` 的输出。"
    )


class MemeSendingMiddleware(AgentMiddleware[ManagerState, ManagerContext]):
    meme_id_map: dict[str, dict[str, MemeResult]]
    send_count: dict[str, int]
    send_history: dict[str, deque[str]]

    def __init__(
        self,
        man_send_per_turn: int | None = None,
    ) -> None:
        self.max_send_per_turn = man_send_per_turn
        self.meme_id_map: dict[str, dict[str, MemeResult]] = defaultdict(dict)
        self.send_count: dict[str, int] = defaultdict(lambda: 0)
        self.send_history: dict[str, deque[str]] = defaultdict(
            lambda: deque(maxlen=KEEP_SEND_HISTORY)
        )
        self.tools = [
            tool(
                args_schema=SearchMemeInput,
                description="从表情包库搜索想要的表情包。",
            )(self.search_meme),
            tool(
                args_schema=SendMemeInput,
                description="向聊天中发送一条表情包。在发送表情包前，必须先使用 `search_meme` 搜索表情包。",
            )(self.send_meme),
        ]

    def gen_id(self, session_id: str) -> str:
        ids = [int(k) for k in self.meme_id_map[session_id]]
        if ids:
            return str(max(ids) + 1)
        return "1"

    async def search_meme(
        self,
        content: str,
        limit: int,
        runtime: ToolRuntime[ManagerContext, ManagerState],
    ) -> Any:
        """从表情包库搜索想要的表情包。"""
        results = await search_memes(content, limit=limit + KEEP_SEND_HISTORY)
        results = [
            r
            for r in results or []
            if r.analysis not in self.send_history[runtime.context["session_id"]]
        ][:limit]
        if not results:
            return "未找到匹配的表情包，请尝试更完整详细的搜索语句。"

        found_meme_analysis = {
            r.analysis: k
            for k, r in self.meme_id_map[runtime.context["session_id"]].items()
        }

        newly_result_map, old_result_map = {}, {}
        for result in results:
            if result.analysis in found_meme_analysis:
                old_result_map[found_meme_analysis[result.analysis]] = result
            else:
                id_ = self.gen_id(runtime.context["session_id"])
                self.meme_id_map[runtime.context["session_id"]][id_] = result
                newly_result_map[id_] = result

        result_map: dict[str, MemeResult] = {**old_result_map, **newly_result_map}

        content_lines: list[str] = [
            f"已找到 {len(results)} 个匹配的表情包，请从下列中选择你希望发送的表情包：",
            *[
                f"- [{meme_id}]{'（本次新发现）' if meme_id in newly_result_map else '（此前已发现）'} {meme_result.analysis}"
                for meme_id, meme_result in result_map.items()
            ],
        ]
        return ToolMessage(
            content="\n".join(content_lines),
            tool_call_id=runtime.tool_call_id,
        )

    async def send_meme(
        self,
        meme_id: str,
        runtime: ToolRuntime[ManagerContext, ManagerState],
    ) -> Any:
        """向聊天中发送一条表情包。"""
        if (
            self.max_send_per_turn is not None
            and self.send_count[runtime.context["session_id"]] >= self.max_send_per_turn
        ):
            return "本轮系统允许的发送表情包次数已用完，请等待下一轮。"
        normalized_meme_id = (
            meme_id.replace("[", "").replace("]", "").casefold().strip()
        )
        result = self.meme_id_map[runtime.context["session_id"]].get(normalized_meme_id)
        if not result:
            return "无法通过索引找到对应的表情包，请检查索引是否正确。"
        self.send_count[runtime.context["session_id"]] += 1
        self.send_history[runtime.context["session_id"]].append(result.analysis)
        await runtime.context["node"].reply(
            CQHTTPMessageSegment.image(result.base64, sub_type=1)
        )
        return Command(
            update={
                "outputs": [
                    OutputMessage(type="meme", data={"content": result.analysis})
                ],
                "messages": [
                    ToolMessage(
                        content=f"已发送表情包：[{meme_id}] " + result.analysis,
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    async def abefore_agent(
        self, state: ManagerState, runtime: Runtime[ManagerContext]
    ) -> dict[str, Any] | None:
        _ = state
        self.send_count[runtime.context["session_id"]] = 0
        self.meme_id_map[runtime.context["session_id"]] = {}

    async def aafter_agent(
        self, state: ManagerState, runtime: Runtime[ManagerContext]
    ) -> dict[str, Any] | None:
        _ = state
        self.send_count[runtime.context["session_id"]] = 0
        self.meme_id_map[runtime.context["session_id"]] = {}
