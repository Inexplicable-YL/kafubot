from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from kafubot.cognition.plugins.base import PluginContext, PluginDefinition

from .ingress import CQHTTPMessageIngestor, ImageAnalyzer, has_model_visible_content
from .store import (
    AgentHistoryRepository,
    ConversationState,
    HistoryRepository,
    InMemoryHistoryRepository,
    SocialEnvironment,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from kafubot.config import AgentConfig


def apply(context: PluginContext, _config: Any) -> None:
    config = cast("AgentConfig", context.service("config"))
    environment = cast(
        "SocialEnvironment | None",
        context.optional_service("environment"),
    )
    if environment is None:
        context.provide(
            "environment",
            SocialEnvironment(
                max_entries=config.environment_message_limit,
                context_window_size=config.executive.context_message_limit,
            ),
        )
    else:
        environment.context_window_size = config.executive.context_message_limit
    factory = cast(
        "Callable[[], ImageAnalyzer]", context.service("image_analyzer_factory")
    )
    context.provide(
        "ingress",
        CQHTTPMessageIngestor(factory, image_workers=config.image_analyzer_workers),
    )


plugin = PluginDefinition(name="environment", apply=apply)


__all__ = [
    "AgentHistoryRepository",
    "ConversationState",
    "CQHTTPMessageIngestor",
    "has_model_visible_content",
    "HistoryRepository",
    "ImageAnalyzer",
    "InMemoryHistoryRepository",
    "plugin",
    "SocialEnvironment",
]
