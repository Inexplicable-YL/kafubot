from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import anyio

if TYPE_CHECKING:
    from agent.base import UserMessage
    from kafubot.config import GateConfig

    from .models import ConversationCandidate, SelfState


@dataclass(slots=True)
class GateDecision:
    wake: bool
    score: float
    reasons: tuple[str, ...]


class GlobalGate:
    """A cheap wake-up gate. It never decides whether or how to reply."""

    def __init__(
        self,
        config: GateConfig,
        *,
        proactive_sessions: set[str],
        talk_value: float,
        keywords: set[tuple[str, float]],
    ) -> None:
        self.config = config
        self.proactive_sessions = proactive_sessions
        self.talk_value = talk_value
        self.keywords = keywords
        self._event = anyio.Event()
        self._lock = anyio.Lock()

    def evaluate(
        self,
        session_id: str,
        message: UserMessage,
        candidate: ConversationCandidate,
        state: SelfState,
    ) -> GateDecision:
        reasons: list[str] = []
        score = 0.0
        if message.is_tome:
            score += 1.0
            reasons.append("directed")
        if session_id in self.proactive_sessions:
            score += self.talk_value
            reasons.append("participation")
        keyword_score = max(
            (
                weight
                for keyword, weight in self.keywords
                if keyword and keyword in message.message.get_plain_text()
            ),
            default=0.0,
        )
        if keyword_score:
            score += keyword_score
            reasons.append("keyword")
        novelty = min(
            self.config.max_novelty,
            candidate.new_count * self.config.novelty_per_message,
        )
        score += novelty
        if novelty:
            reasons.append("novelty")
        thread = state.active_threads.get(session_id)
        if thread is not None:
            residue_age = max(
                0.0,
                (datetime.now(UTC) - thread.residue_updated_at).total_seconds(),
            )
            residue = thread.attention_residue * math.pow(
                0.5,
                residue_age / self.config.residue_half_life_seconds,
            )
            if residue:
                score += residue
                reasons.append("attention_residue")
        allowed = (
            message.is_tome
            or session_id in self.proactive_sessions
            or thread is not None
        )
        return GateDecision(
            wake=allowed and score >= self.config.threshold,
            score=score,
            reasons=tuple(reasons),
        )

    async def consider(
        self,
        session_id: str,
        message: UserMessage,
        candidate: ConversationCandidate,
        state: SelfState,
    ) -> GateDecision:
        decision = self.evaluate(session_id, message, candidate, state)
        if decision.wake:
            await self.force_wake()
        return decision

    async def force_wake(self) -> None:
        async with self._lock:
            self._event.set()

    async def wait(self, *, timeout: float | None = None) -> bool:
        async with self._lock:
            event = self._event
        if timeout is None:
            await event.wait()
        else:
            with anyio.move_on_after(timeout) as cancel_scope:
                await event.wait()
            if cancel_scope.cancel_called:
                return False
        if self.config.debounce_seconds:
            await anyio.sleep(self.config.debounce_seconds)
        async with self._lock:
            if self._event is event:
                self._event = anyio.Event()
        return True
