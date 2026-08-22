from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from kafubot.cognition.models import get_nonthinking_model
from kafubot.cognition.plugins.base import PluginContext, PluginDefinition

from .learner import ExpressionLearner

if TYPE_CHECKING:
    from kafubot.cognition.plugins.lifecycle import (
        ContextWindowEvicted,
        ReplyCommitted,
        ReplyPreparation,
    )


def apply(context: PluginContext, config: dict[str, Any]) -> None:
    values: dict[str, Any] = {
        "enable_precise_expression_selection": True,
        **config,
    }
    learner = cast("Any", ExpressionLearner)(
        analyze_model=get_nonthinking_model(
            float(values.pop("analyze_temperature", 0.2))
        ),
        selection_model=get_nonthinking_model(
            float(values.pop("selection_temperature", 0.1))
        ),
        **values,
    )

    async def evicted(event: ContextWindowEvicted) -> None:
        await learner.consume_evicted(event.session_id, event.model_messages)

    async def prepare(preparation: ReplyPreparation) -> str | None:
        content, selected_ids = learner.prepare_reply_context(
            preparation.context.session_id
        )
        preparation.metadata["selected_expression_ids"] = selected_ids
        return content or None

    async def committed(event: ReplyCommitted) -> None:
        await learner.record_reply(
            list(event.preparation.metadata.get("selected_expression_ids") or [])
        )

    context.resource("expression", learner)
    context.on_context_evicted(evicted)
    context.on_prepare_reply(prepare)
    context.on_reply_committed(committed)
    context.clear_session(learner.clear_session)


plugin = PluginDefinition(name="expression", apply=apply)


__all__ = ["ExpressionLearner", "plugin"]
