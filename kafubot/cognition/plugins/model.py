from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from kafubot.cognition.models import get_thinking_model
from kafubot.cognition.plugins.base import PluginContext, PluginDefinition

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.language_models.chat_models import BaseChatModel


def apply(context: PluginContext, _config: Any) -> None:
    factory = cast(
        "Callable[[], BaseChatModel] | None",
        context.optional_service("model_factory_override"),
    )
    context.provide(
        "model_factory",
        factory or (lambda: get_thinking_model(reasoning_effort="max")),
    )


plugin = PluginDefinition(name="model", apply=apply)
