"""# from typing import Any"""

import random
from typing import Any
from typing_extensions import override

from sekaibot import Node
from sekaibot.adapter.cqhttp.event import GroupMessageEvent, PrivateMessageEvent
from sekaibot.permission import User
from sekaibot.rule import Keywords


@Keywords(
    "中",
    "蒸",
    "lrc",
    "林睿晨",
    "dsz",
    "松泽",
    "四维",
    "香氤",
    "kz",
    "叩之",
)
@User("group_596488203", "group_1011357049", "group_1058218429", "group_1087911123")
class AutoReply(Node[GroupMessageEvent, dict, Any]):  # type: ignore
    """Hello, World! 示例节点。"""

    priority = 0
    zb = Keywords.Param()

    @override
    async def handle(self) -> None:
        keyw = self.zb[0] if self.zb else "蒸"

        keyw = "林睿晨" if keyw == "lrc" else keyw
        keyw = "松泽" if keyw == "dsz" else keyw
        if keyw in {"叩之", "kz"}:
            text = random.choice(  # noqa: S311
                (
                    "香草叩之",
                    "香茶叩之",
                    "叩之好可爱",
                    "香甜叩之",
                    "叩之草我",
                    "诶我草叩之怎么这么坏啊",
                    "叩之是四爱",
                    "叩之是4i",
                    "叩之就是爱慕",
                    "叩之不见了",
                    "叩之北朝的初雪",
                    "叩之北朝的初水",
                    "叩之北朝的豪爽",
                    "香甜叩之的小学",
                    "叩之很带派",
                    "想吃叩之的大汗脚",
                    "被叩之口了",
                    "叩之是冯冯",
                    "叩之是直女装姬",
                    "叩之是拉拉",
                    "叩之是lg",
                    "叩之太畜了",
                    "叩之太畜了",
                    "叩之滚",
                    "叩之卡比去",
                    "叩之扣比去",
                    "恐双了",
                    "叩之是酷儿",
                )
            )
        elif keyw == "中":
            text = random.choice(  # noqa: S311
                (
                    "中被阉割了",
                    "中蒸出栏了",
                    "中蒸二字看似普通实则不凡",
                    "中蒸二字看似普通实则不凡",
                    "中被阉割了",
                    "中蒸出栏了",
                    "中蒸糯而不膻",
                    "香草中",
                    "香茶中",
                    "中好可爱",
                    "香甜中",
                    "被中茶了",
                    "中是四爱",
                    "中就是爱慕",
                    "中不见了",
                    "香甜中的小学",
                    "中转过去一下我有急事",
                    "中很带派",
                    "中是冯冯",
                )
            )
        else:
            text = random.choice(  # noqa: S311
                (
                    "{keyw}鞭好粗",
                    "{keyw}鞭好大",
                    "香草{keyw}",
                    "香茶{keyw}",
                    "{keyw}好可爱",
                    "{keyw}立了",
                    "香甜{keyw}",
                    "{keyw}草我",
                    "诶我草{keyw}怎么这么坏啊",
                    "被{keyw}茶了",
                    "{keyw}是四爱",
                    "{keyw}是4i",
                    "{keyw}是南通",
                    "{keyw}素指南",
                    "{keyw}就是爱慕",
                    "{keyw}是正太",
                    "{keyw}不见了",
                    "{keyw}蛇了",
                    "{keyw}北朝的初雪",
                    "{keyw}北朝的初水",
                    "{keyw}北朝的豪爽",
                    "香甜{keyw}的小学",
                    "北{keyw}顶到职场了",
                    "想吃{keyw}精",
                    "想吃{keyw}的大橘瓣",
                    "想电{keyw}的前列腺",
                    "{keyw}转过去一下我有急事",
                    "想吃{keyw}的高玩",
                    "{keyw}很带派",
                    "想吃{keyw}的大汗脚",
                    "被{keyw}口了",
                    "{keyw}是蓝凉",
                    "{keyw}是冯冯",
                )
            ).format(keyw=keyw)

        await self.reply(text)

    @override
    async def rule(self) -> bool:
        return str(self.event.group_id) != "788499440"


@Keywords("芽", "老婆", "我", "妻子")
@User("413966479")
class YanCheng(Node[GroupMessageEvent, dict, Any]):  # type: ignore
    """言承"""

    priority = 0

    keyw = Keywords.Param()

    @override
    async def handle(self) -> None:
        if len(self.keyw) > 1 and "芽" in self.keyw:
            await self.reply("理芽不是言承的……！", at_sender=True)
            self.stop()


@Keywords("/回调", "/callback", "/cb")
@User("2682064633")
class CallBack(Node[PrivateMessageEvent, dict, Any]):  # type: ignore
    """回调"""

    priority = 0
    block = True

    @override
    async def handle(self) -> None:
        await self.reply("回调成功", at_sender=True)
