from __future__ import annotations

from collections import defaultdict, deque
from typing import TYPE_CHECKING, cast

from langchain.tools import ToolRuntime, tool
from pydantic import BaseModel, Field

from kafubot.adapters.cqhttp.message import CQHTTPMessageSegment
from kafubot.cognition.plugins.base import PluginContext, PluginDefinition, ToolScope

from .backend import MemeResult, search_memes

if TYPE_CHECKING:
    from kafubot.cognition.plugins.lifecycle import ReplyCommitted, ReplyPreparation
    from kafubot.social import SocialAgentContext


class MemeConfig(BaseModel):
    history_size: int = Field(default=3, ge=1, le=20)


class MemePlugin:
    def __init__(self, *, history_size: int) -> None:
        self.history: dict[str, deque[str]] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        self.candidates: dict[str, list[MemeResult]] = {}

    async def search_candidates(
        self,
        session_id: str,
        intent: str,
        *,
        limit: int = 5,
    ) -> list[MemeResult]:
        results = await search_memes(intent, limit=max(1, min(limit, 8)))
        candidates = [
            result
            for result in results or []
            if result.analysis not in self.history[session_id]
        ]
        self.candidates[session_id] = candidates
        return candidates

    async def preview_candidates(
        self,
        session_id: str,
        intent: str,
        *,
        limit: int = 5,
    ) -> str:
        candidates = await self.search_candidates(session_id, intent, limit=limit)
        if not candidates:
            return "没有找到可用表情包；本轮不要设置 meme_intent。"
        return (
            "可用表情包候选；如需发送，请把选中的描述原样写入 reply 的 "
            "meme_intent：\n"
            + "\n".join(
                f"{index}. {candidate.analysis}"
                for index, candidate in enumerate(candidates, start=1)
            )
        )

    async def prepare_reply(self, preparation: ReplyPreparation) -> str | None:
        intent = preparation.contract.meme_intent.strip()
        if not intent:
            return None
        session_id = preparation.context.session_id
        selected = next(
            (
                result
                for result in self.candidates.get(session_id, [])
                if result.analysis == intent
                and result.analysis not in self.history[session_id]
            ),
            None,
        )
        if selected is None:
            results = await self.search_candidates(session_id, intent, limit=6)
            selected = next(
                (
                    result
                    for result in results
                    if result.analysis not in self.history[session_id]
                ),
                None,
            )
        if selected is None:
            return "未找到符合 meme_intent 的表情包；本次只发送文字。"
        preparation.metadata["meme"] = selected
        return f"将随回复发送表情包：{selected.analysis}"

    async def reply_committed(self, event: ReplyCommitted) -> None:
        selected = event.preparation.metadata.get("meme")
        if not isinstance(selected, MemeResult):
            return
        await event.preparation.actions.reply(
            CQHTTPMessageSegment.image(selected.base64, sub_type=1)
        )
        self.history[event.preparation.context.session_id].append(selected.analysis)

    async def clear_session(self, session_id: str) -> None:
        self.history.pop(session_id, None)
        self.candidates.pop(session_id, None)


def apply(context: PluginContext, config: MemeConfig) -> None:
    plugin_instance = MemePlugin(history_size=config.history_size)

    @tool(
        "search_meme",
        description=(
            "在当前 OPEN 会话中按情绪或动作意图搜索表情包候选。成本 1。"
            "此工具只预览候选；真正发送必须通过 reply 的 meme_intent。"
        ),
    )
    async def preview_meme(
        intent: str,
        runtime: ToolRuntime,
        limit: int = 5,
    ) -> str:
        executive_context = cast("SocialAgentContext", runtime.context)
        session_id = executive_context.open_session_id
        if session_id is None:
            return "search_meme 只能在 OPEN 会话中使用。"
        if not executive_context.spend(1):
            return "focus budget exhausted"
        result = await plugin_instance.preview_candidates(
            session_id,
            intent,
            limit=limit,
        )
        return f"{result}\n[focus budget remaining={executive_context.budget}]"

    context.provide("meme", plugin_instance)
    context.tool(preview_meme, ToolScope.OPEN)
    context.on_prepare_reply(plugin_instance.prepare_reply)
    context.on_reply_committed(plugin_instance.reply_committed)
    context.clear_session(plugin_instance.clear_session)


plugin = PluginDefinition(
    name="meme",
    apply=apply,
    config_model=MemeConfig,
)


__all__ = ["MemeConfig", "MemePlugin", "plugin"]
