from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol, cast

import anyio

from kafubot.cognition.media.image import ImageReadResult, read_image
from kafubot.cognition.message import HistoryMessage, QQMessage, QQMessageSegment
from kafubot.cognition.plugins.base import PluginContext, PluginDefinition
from kafubot.cognition.types import UserMessage

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from langchain_core.runnables import RunnableConfig

    from kafubot.actions import QQActions
    from kafubot.adapters.cqhttp.event import GroupMessageEvent, PrivateMessageEvent
    from kafubot.config import AgentConfig

logger = logging.getLogger(__name__)

DEFAULT_USERNAME = "陌生用户"


class ImageAnalyzer(Protocol):
    async def ainvoke(
        self,
        input: dict[str, Any],  # noqa: A002
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> str: ...


def has_model_visible_content(message: UserMessage) -> bool:
    return any(
        bool(segment.data.get("text", "").strip())
        if segment.type == "text"
        else segment.type not in {"image", "meme"} or bool(segment.data.get("content"))
        for segment in message.message
    )


class CQHTTPMessageIngestor:
    """Translate native OneBot events into the stable cognitive message model."""

    def __init__(
        self,
        image_analyzer_factory: Callable[[], ImageAnalyzer],
        *,
        image_workers: int,
    ) -> None:
        self.image_analyzer = image_analyzer_factory()
        self.image_limiter = anyio.CapacityLimiter(image_workers)

    async def convert(
        self,
        event: GroupMessageEvent | PrivateMessageEvent,
        history: Sequence[HistoryMessage],
        actions: QQActions,
    ) -> UserMessage | None:
        message = await QQMessage.from_cqhttp_message(
            event.message,
            history,
            lambda file: self.get_image(actions, file),
        )
        if event.reply:
            message = (
                QQMessageSegment.reply(
                    message_id=str(event.reply.message_id),
                    include={"message_id"},
                )
                + message
            )
        if not message:
            return None

        images: list[tuple[ImageReadResult, bool]] = []
        for segment in message:
            if segment.type in {"image", "meme"} and not segment.data.get("content"):
                image_data = segment.data.get("image")
                if image_data is not None:
                    images.append(
                        (cast("ImageReadResult", image_data), segment.type == "meme")
                    )

        user_message = UserMessage(
            role="user",
            timestamp=datetime.fromtimestamp(event.time, tz=UTC),
            user=event.sender.card or event.sender.nickname or DEFAULT_USERNAME,
            message=message,
            user_id=str(event.user_id),
            message_id=str(event.message_id),
            is_tome=event.is_tome(),
            images=images,
            chat_type=event.message_type,
        )
        if images:
            return await self.fill_images(user_message)
        return user_message if has_model_visible_content(user_message) else None

    async def get_image(
        self,
        actions: QQActions,
        file: str,
    ) -> ImageReadResult | None:
        try:
            result: dict[str, str] = await actions.call_api("get_image", file=file)
            if path := result.get("file"):
                return await read_image(path, result.get("url"))
        except Exception:
            logger.exception("Failed to read QQ image: %s", file)
        return None

    async def fill_images(self, message: UserMessage) -> UserMessage | None:
        analyses: list[str | None] = [None] * len(message.images)

        async def analyze_one(
            result_index: int,
            image: ImageReadResult,
            as_meme: bool,
        ) -> None:
            try:
                async with self.image_limiter:
                    analyses[result_index] = await self.image_analyzer.ainvoke(
                        {
                            "image": image.base64,
                            "phash": image.phash,
                            "as_meme": as_meme,
                        }
                    )
            except Exception:
                logger.exception("Failed to analyze incoming image")

        async with anyio.create_task_group() as task_group:
            for index, (image, as_meme) in enumerate(message.images):
                task_group.start_soon(analyze_one, index, image, as_meme)

        results = iter(analyses)
        rebuilt = QQMessage()
        for segment in message.message:
            if segment.type in {"image", "meme"} and not segment.data.get("content"):
                content = next(results, None)
                if content:
                    rebuilt += getattr(QQMessageSegment, segment.type)(content=content)
                continue
            rebuilt += segment
        message.message = rebuilt
        return message if has_model_visible_content(message) else None


def apply(context: PluginContext, _config: Any) -> None:
    config = cast("AgentConfig", context.service("config"))
    factory = cast(
        "Callable[[], ImageAnalyzer]", context.service("image_analyzer_factory")
    )
    context.provide(
        "ingress",
        CQHTTPMessageIngestor(factory, image_workers=config.image_analyzer_workers),
    )


plugin = PluginDefinition(name="ingress", apply=apply)


__all__ = [
    "CQHTTPMessageIngestor",
    "ImageAnalyzer",
    "has_model_visible_content",
    "plugin",
]
