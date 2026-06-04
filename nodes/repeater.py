from pathlib import Path
from typing import Any
from typing_extensions import override

from sekaibot import Node
from sekaibot.adapter.cqhttp.event import GroupMessageEvent
from sekaibot.adapter.cqhttp.message import CQHTTPMessage, CQHTTPMessageSegment
from sekaibot.rule import WordFilter

REPEAT_THRESHOLD = 3


@WordFilter(word_file=Path("./nodes/sensitive_words_lines.txt"), use_aho=True)
class Repeater(Node[GroupMessageEvent, dict, Any]):
    priority = 5

    @override
    async def handle(self) -> None:
        def del_file_id(msg: CQHTTPMessageSegment):
            msg.data.pop("file_id", None)
            msg.data.pop("url", None)
            msg.data.pop("file", None)
            return msg

        message = repr(CQHTTPMessage(list(map(del_file_id, self.event.message))))
        group_id = self.event.group_id

        if group_id not in self.node_state:
            self.node_state[group_id] = {
                "last_message": message,
                "count": 1,
                "repeated": False,
            }
            return

        state = self.node_state[group_id]

        if message == state["last_message"]:
            if state["repeated"]:
                return
            state["count"] += 1
            if state["count"] >= REPEAT_THRESHOLD:
                await self.call_api(
                    "forward_group_single_msg",
                    message_id=self.event.message_id,
                    group_id=group_id,
                )
                state["repeated"] = True
                self.stop()
            elif state["count"] > 0:
                self.stop()
        else:
            self.node_state[group_id] = {
                "last_message": message,
                "count": 1,
                "repeated": False,
            }

    @override
    async def rule(self) -> bool:
        return (not self.event.is_tome()) and self.event.user_id != 2854196310  # noqa: PLR2004
