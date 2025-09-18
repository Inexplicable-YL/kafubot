import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from contextlib import suppress
from datetime import datetime
from itertools import count
from random import random
from typing import Any
from zoneinfo import ZoneInfo

import anyio
from anyio import Lock
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel
from sekaibot.log import logger
from sortedcontainers import SortedDict  # type: ignore
from zhconv import convert  # type: ignore

from agent import create_agent
from agent.history import AsyncPersistentLRUDict
from pipeline import PipelineProcess

from .image import image_file_to_base64_jpg
from .prompt import photo_prompt

id_gen = count(start=1)


def get_next_id() -> int:
    return next(id_gen)


class UnhandleImage(BaseModel):
    file_path: str
    file_id: str
    name: str


img_cache = AsyncPersistentLRUDict(
    file_path="../sekaibot-cache/img_cache.tmp", max_len=100
)


message_dict: dict[str, dict[int, tuple[BaseMessage | UnhandleImage, float]]] = (
    defaultdict(SortedDict)
)
message_dict_locks: dict[str, Lock] = defaultdict(Lock)

photo_model = create_agent(
    "gpt-4.1-mini",
    provider="OPENAI",
    prompt=photo_prompt,
    temperature=0.5,
    tools=[],
)


async def get_img_description(
    file_path: str, file_id: str, name: str
) -> HumanMessage | None:
    # with suppress(Exception):
    try:
        base64_image = image_file_to_base64_jpg(file_path)
        img_msg = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ]
        )
        if img_description := (await photo_model.ainvoke({"messages": [img_msg]})).get(
            "output", None
        ):
            content = [
                {
                    "type": "text",
                    "text": f"[Friend: {name}]发送了图片或表情包\n{img_description}",
                },
            ]
            await img_cache.set(file_id, img_description)
            return HumanMessage(content=content)  # type: ignore
    except Exception:
        logger.exception("get_img_description")
    return None


async def handle_img(
    session_id: str,
    name: str,
    get_img_func: Callable[[str], Awaitable[str | None]],
    file_id: str,
) -> None:
    message_id = get_next_id()
    if img_description_cache := await img_cache.get(file_id, None):
        content = [
            {
                "type": "text",
                "text": f"[Friend: {name}]发送了图片或表情包\n{img_description_cache}",
            },
        ]
        async with message_dict_locks[session_id]:
            message_dict[session_id][message_id] = (
                HumanMessage(content=content),
                time.time(),
            )  # type: ignore
        return
    if file_path := await get_img_func(file_id):
        async with message_dict_locks[session_id]:
            message_dict[session_id][message_id] = (
                UnhandleImage(file_path=file_path, file_id=file_id, name=name),
                time.time(),
            )


def get_trigger(message: str, trigger: float = 0.8) -> bool:
    value = random()  # noqa: S311
    message = convert(message, "zh-cn")
    keyws = [
        "可不",
        "星界",
        "咖喱乌冬",
        "kafu",
        "sekai",
        "花谱",
        "里命",
        "狐子",
        "羽累",
        "明透",
        "ASU",
        "异世界情绪",
        "情绪",
        "理芽",
        "幸祜",
        "春猿火",
        "神椿",
        "VIP",
        "VWP",
    ]
    if any(keyw in message for keyw in keyws):
        if "可不" in message:
            value += 0.3
        value += 0.3

    return value >= trigger


async def _get_img_description(
    msgs: dict[int, tuple[BaseMessage | UnhandleImage, float]],
    key: int,
    img_model: UnhandleImage,
) -> None:
    if _img_description := await get_img_description(
        file_path=img_model.file_path, name=img_model.name, file_id=img_model.file_id
    ):
        msgs[key][0] = _img_description
    else:
        with suppress(KeyError):
            msgs.pop(key)


async def get_answer(
    runnable: PipelineProcess,
    session_id: str,
    name: str,
    message: str,
    is_tome: bool = False,
    random_trigger: bool = False,
) -> str | None:
    message_id = get_next_id()
    content = [{"type": "text", "text": f"[Friend: {name}]: \n{message}"}]

    if not content:
        return None
    async with message_dict_locks[session_id]:
        message_dict[session_id][message_id] = (
            HumanMessage(content=content),
            time.time(),
        )  # type: ignore

    if (
        get_trigger(message, trigger=0.9 if random_trigger else 1.1)
        and len(message_dict[session_id]) > 3  # noqa: PLR2004
    ) or is_tome:
        async with message_dict_locks[session_id]:
            messages = message_dict[session_id].copy()
            message_dict.pop(session_id)
        messages = {k: messages[k] for k in list(messages.keys())[-12:]}

        with suppress(Exception):
            async with anyio.create_task_group() as tg:
                for msg_id, (msg, _) in messages.items():
                    if isinstance(msg, UnhandleImage):
                        tg.start_soon(_get_img_description, messages, msg_id, msg)

        print(list(messages.values()))
        res = await use_llm(runnable, session_id, list(messages.values()), is_tome)
        answer: str = res.get("output", [])
        if answer:
            return convert(answer, "zh-tw")

    return None


async def use_llm(
    runnable: PipelineProcess,
    session_id: str,
    messages: list[tuple[BaseMessage, float]],
    is_tome: bool,
) -> dict[str, Any]:
    try:
        await runnable.ainvoke(
            {
                "input": messages[:-1],
                "pass": False,
            },
            config={"configurable": {"session_id": session_id}},
        )
        return await runnable.ainvoke(
            {
                "input": messages[-1][0],
                "time": datetime.now(ZoneInfo("Asia/Shanghai")).strftime(
                    "%Y年%m月%d日 %H时%M分"
                ),
            }
            | ({"pass": True} if is_tome else {}),
            config={"configurable": {"session_id": session_id}},
        )
    except Exception:
        logger.exception("use llm error")
        return {"output": []}
