from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kafubot.cognition.models import get_thinking_model
from kafubot.cognition.plugins.base import PluginContext, PluginDefinition

from .learner import (
    BehaviorLearner,
)

if TYPE_CHECKING:
    from kafubot.cognition.plugins.lifecycle import (
        ContextWindowEvicted,
        ReplyCommitted,
        ReplyPreparation,
    )


def apply(context: PluginContext, config: dict[str, Any]) -> None:
    learner = BehaviorLearner(
        analyze_model=get_thinking_model(reasoning_effort="high"),
        **config,
    )

    async def evicted(event: ContextWindowEvicted) -> None:
        await learner.consume_evicted(event.session_id, event.model_messages)

    async def prepare(preparation: ReplyPreparation) -> str | None:
        content, selected_ids = learner.prepare_reply_context(
            preparation.context.session_id
        )
        preparation.metadata["selected_behavior_ids"] = selected_ids
        return content or None

    async def committed(event: ReplyCommitted) -> None:
        await learner.record_reply(event.preparation.context.session_id)

    context.resource("behavior", learner)
    context.on_context_evicted(evicted)
    context.on_prepare_reply(prepare)
    context.on_reply_committed(committed)
    context.clear_session(learner.clear_session)


plugin = PluginDefinition(name="behavior", apply=apply)

__all__ = [
    "BehaviorLearner",
    "plugin",
]
