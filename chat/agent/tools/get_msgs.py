from typing import TYPE_CHECKING, cast

from langchain.tools import ToolRuntime, tool
from pydantic import BaseModel, ConfigDict, Field

from chat.utils import content_to_text

if TYPE_CHECKING:
    from chat.agent.base import ManagerContext, ManagerState


class GetEarlyMessagesInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    limit: int = Field(default=20, description="要显示的消息数量。", gt=1, le=40)
    runtime: ToolRuntime = Field(exclude=True)


@tool(
    args_schema=GetEarlyMessagesInput,
    description="显示比当前可见的消息更早的消息。需要使用`limit`限制显示的消息数量。",
)
def get_early_messages(limit: int, runtime: ToolRuntime) -> str:
    _runtime = cast("ToolRuntime[ManagerContext, ManagerState]", runtime)
    if not _runtime.state["early_messages"]:
        return "没有比当前可见的消息更早的消息。"
    return (
        "以下是比当前可见的消息更早的消息：\n\n"
        + "\n".join(
            [
                content_to_text(m.content)
                for m in _runtime.state["early_messages"][-limit:]
            ]
        )
        + "\n\n以上消息往后衔接当前可见的消息。"
    )
