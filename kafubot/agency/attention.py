from __future__ import annotations

import math
import random
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from .models import (
    AttentionFeatures,
    ConversationCandidate,
    HomeItem,
    SelfState,
    SocialHome,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from kafubot.config import AttentionConfig


class AttentionScheduler:
    """Ranks conversation contexts without creating an agent per context."""

    def __init__(
        self,
        config: AttentionConfig,
        *,
        clock: Callable[[], datetime] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.config = config
        self.clock = clock or (lambda: datetime.now(UTC))
        self.rng = rng or random.Random()  # noqa: S311 - attention jitter, not security

    def rank(
        self,
        candidates: Sequence[ConversationCandidate],
        state: SelfState,
        *,
        limit: int,
    ) -> SocialHome:
        now = self.clock()
        items: list[HomeItem] = []
        for candidate in candidates:
            features = self._features(candidate, state, now)
            score = (
                self.config.directedness * features.directedness
                + self.config.social_obligation * features.social_obligation
                + self.config.relationship * features.relationship
                + self.config.urgency * features.urgency
                + self.config.continuity * features.continuity
                + self.config.novelty * features.novelty
                - self.config.fatigue * features.fatigue
                - self.config.interruption * features.interruption
                + self.rng.uniform(
                    -self.config.random_jitter,
                    self.config.random_jitter,
                )
            )
            items.append(
                HomeItem(
                    **candidate.model_dump(),
                    score=score,
                    features=features,
                )
            )
        items.sort(key=lambda item: (item.score, item.latest_at), reverse=True)
        return SocialHome(generated_at=now, items=items[:limit])

    def _features(
        self,
        candidate: ConversationCandidate,
        state: SelfState,
        now: datetime,
    ) -> AttentionFeatures:
        age = max(0.0, (now - candidate.latest_at).total_seconds())
        thread = state.active_threads.get(candidate.session_id)
        continuity = 0.0
        if thread is not None:
            residue_age = max(
                0.0,
                (now - thread.residue_updated_at).total_seconds(),
            )
            continuity = thread.attention_residue * math.pow(
                0.5,
                residue_age / self.config.residue_half_life_seconds,
            )
        current_focus = state.current_focus[0] if state.current_focus else None
        return AttentionFeatures(
            directedness=min(1.0, float(candidate.directed_count)),
            social_obligation=min(1.0, candidate.question_count / 2),
            relationship=1.0 if thread is not None else 0.0,
            urgency=math.exp(-age / 180),
            continuity=continuity,
            novelty=min(1.0, math.log1p(candidate.new_count) / math.log(6)),
            fatigue=min(1.0, state.fatigue_by_session.get(candidate.session_id, 0)),
            interruption=(
                1.0
                if current_focus is not None
                and current_focus != candidate.session_id
                and candidate.directed_count == 0
                else 0.0
            ),
        )
