from typing import Any
from typing_extensions import override

from sekaibot import Node
from sekaibot.adapter.cqhttp.event import FriendRequestEvent, GroupRequestEvent


class AcceptRequest(Node[FriendRequestEvent | GroupRequestEvent, Any, Any]):
    priority = 0
    block = True

    @override
    async def handle(self) -> None:
        await self.event.approve()
