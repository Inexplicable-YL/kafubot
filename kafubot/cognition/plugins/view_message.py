from __future__ import annotations

from contextlib import suppress
from datetime import UTC, datetime
from html import escape
from typing import TYPE_CHECKING, Any, cast
from zoneinfo import ZoneInfo

from langchain.tools import ToolRuntime, tool

from kafubot.cognition.plugins.base import PluginContext, PluginDefinition, ToolScope

if TYPE_CHECKING:
    from kafubot.actions import QQActions
    from kafubot.cognition.plugins.environment import SocialEnvironment
    from kafubot.social import SocialAgentContext


async def read_forward_message(
    actions: QQActions,
    *,
    id: str,
    limit: int = 10,
) -> str:
    limit = max(2, min(limit, 20))
    try:
        messages: list[dict[str, Any]] = (
            await actions.call_api("get_forward_msg", id=id)
        )["messages"]
    except Exception:
        return "无法获取转发消息，请检查消息ID是否正确。"
    msgs: list[str] = []
    for msg in messages:
        with suppress(Exception):
            text: str = ""
            for m in msg["message"]:
                if m.get("type") == "text":
                    text += m["data"]["text"]
            if text:
                msgs.append(
                    f"<forward-message time={datetime.fromtimestamp(int(msg['time']), tz=UTC).astimezone(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')}, user={escape(msg['sender']['nickname'], quote=True)}>\n{text}\n</forward-message>"
                )
    full_len = len(msgs)
    if len(msgs) > limit:
        if limit % 2 == 0:
            msgs = (
                msgs[: limit // 2]
                + [f"已省略中间{len(msgs) - limit}条消息"]
                + msgs[-limit // 2 :]
            )
        else:
            msgs = (
                msgs[: (limit - 1) // 2]
                + [f"已省略中间{len(msgs) - limit}条消息"]
                + msgs[-(limit + 1) // 2 :]
            )
    if msgs:
        shown_len = min(full_len, limit)
        return (
            f"以下是转发消息，共 {full_len} 条，已显示 {shown_len} 条:\n\n"
            + "\n".join(msgs)
        )
    return "成功获取转发消息，但无法解析消息，请放弃获取。"


def apply(context: PluginContext, _config: Any) -> None:
    environment = cast("SocialEnvironment", context.service("environment"))

    @tool(
        "view_forward_message",
        description="读取当前已打开会话中的 QQ 合并转发消息。",
    )
    async def view_forward(
        id: str,
        runtime: ToolRuntime,
        limit: int = 10,
    ) -> str:
        executive_context = cast("SocialAgentContext", runtime.context)
        session_id = executive_context.open_session_id
        if session_id is None:
            return "请先打开一个会话。"
        actions = await environment.actions_for(session_id)
        if actions is None:
            return "当前会话没有可用的 QQ action context。"
        return await read_forward_message(actions, id=id, limit=limit)

    context.tool(view_forward, ToolScope.OPEN)


plugin = PluginDefinition(
    name="view_message",
    apply=apply,
    requires=("environment",),
)


__all__ = ["plugin", "read_forward_message"]
