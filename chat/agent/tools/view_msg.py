from contextlib import suppress
from datetime import UTC, datetime
from html import escape
from typing import TYPE_CHECKING, Any, cast
from zoneinfo import ZoneInfo

from langchain.tools import ToolRuntime, tool
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from chat.agent.base import ManagerContext, ManagerState
    from nodes.group_agent import GroupAgent


class ViewMessagesInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(description="消息ID。注意不是message_id。是转发消息的id。")
    limit: int = Field(default=10, description="要显示的消息数量。", gt=1, le=40)
    runtime: ToolRuntime = Field(exclude=True)


@tool(
    args_schema=ViewMessagesInput,
    description="获取`forward`（即转发消息）消息的具体内容。",
)
async def view_forward_message(id: str, limit: int, runtime: ToolRuntime) -> str:
    _runtime = cast("ToolRuntime[ManagerContext, ManagerState]", runtime)
    node: GroupAgent = _runtime.context["node"]
    try:
        messages: list[dict[str, Any]] = (
            await node.event.adapter.call_api("get_forward_msg", id=id)
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
        return "以下是转发消息:\n\n" + "\n".join(msgs)
    return "成功获取转发消息，但无法解析消息，请放弃获取。"
