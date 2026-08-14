from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from typing_extensions import override

import aiosqlite
import anyio
from langchain.agents.middleware import AgentMiddleware

from agent.base import ManagerContext, ManagerState, UserMessage
from agent.session import register_session_clearer
from agent.social_signals import (
    SocialSignalAnalysis,
    SocialSignalAnalyzer,
    SocialSignalService,
    is_reliable_signal,
)
from agent.telemetry import log_social_event

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from anyio.abc import TaskGroup
    from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PendingReply:
    effect_id: str
    session_id: str
    sent_at: datetime
    text: str
    action: dict[str, Any]
    selected_behavior_ids: list[int] = field(default_factory=list)
    selected_expression_ids: list[int] = field(default_factory=list)
    observed_messages: list[UserMessage] = field(default_factory=list)
    competing_reply_count: int = 1
    signal_analyses: list[SocialSignalAnalysis] = field(default_factory=list)
    session_generation: int = 0


class ReplyEffectMiddleware(AgentMiddleware[ManagerState, ManagerContext]):
    """Track only effects observable through the installed QQ adapter."""

    state_schema = ManagerState

    def __init__(
        self,
        db_path: str = ".database/reply_effects.db",
        *,
        observation_message_limit: int = 8,
        observation_seconds: int = 180,
        signal_analyzer: SocialSignalAnalyzer | None = None,
        signal_service: SocialSignalService | None = None,
        effect_observers: list[
            Callable[[PendingReply, dict[str, Any], float], Awaitable[None]]
        ]
        | None = None,
    ) -> None:
        self.db_path = db_path
        self.observation_message_limit = observation_message_limit
        self.observation_window = timedelta(seconds=max(30, observation_seconds))
        self._pending: dict[str, list[PendingReply]] = {}
        self._session_generation: dict[str, int] = {}
        self._db: aiosqlite.Connection | None = None
        self._effect_observers = effect_observers or []
        self._owns_signal_service = signal_service is None
        self._signal_service = signal_service or SocialSignalService(
            signal_analyzer or SocialSignalAnalyzer(None)
        )
        self._task_group: TaskGroup | None = None
        self._start_lock = anyio.Lock()
        self._db_lock = anyio.Lock()
        register_session_clearer(self.clear_session)

    async def _ensure_task_group(self) -> TaskGroup:
        if self._task_group is not None:
            return self._task_group
        async with self._start_lock:
            if self._task_group is None:
                task_group = anyio.create_task_group()
                await task_group.__aenter__()
                self._task_group = task_group
        return self._task_group

    async def _schedule_finalize(self, pending: PendingReply) -> None:
        task_group = await self._ensure_task_group()
        task_group.start_soon(self._finalize, pending)

    def add_effect_observer(
        self,
        observer: Callable[[PendingReply, dict[str, Any], float], Awaitable[None]],
    ) -> None:
        self._effect_observers.append(observer)

    async def _ensure_db(self) -> aiosqlite.Connection:
        if self._db is None:
            self._db = await aiosqlite.connect(self.db_path)
            await self._db.execute("PRAGMA journal_mode=WAL")
            await self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS reply_effects (
                    effect_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    sent_at TEXT NOT NULL,
                    reply_text TEXT NOT NULL,
                    action_json TEXT NOT NULL,
                    followup_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    attribution_confidence REAL NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            await self._db.execute(
                "CREATE INDEX IF NOT EXISTS ix_reply_effects_session ON reply_effects(session_id, sent_at)"
            )
            await self._db.commit()
        return self._db

    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    def _score(self, pending: PendingReply) -> tuple[dict[str, Any], float]:
        targets = set(pending.action.get("target_user_ids") or [])
        followups = pending.observed_messages
        target_followups = [msg for msg in followups if msg.user_id in targets]
        directed_followups = [msg for msg in followups if msg.is_tome]
        attributed_followups = list(
            {
                msg.message_id: msg for msg in [*target_followups, *directed_followups]
            }.values()
        )
        analysis_by_id = {
            analysis.message_id: analysis for analysis in pending.signal_analyses
        }
        attributed_analyses = [
            analysis_by_id[msg.message_id]
            for msg in attributed_followups
            if msg.message_id in analysis_by_id
            and analysis_by_id[msg.message_id].target == "bot"
        ]
        correction_count = sum(
            analysis.correction and is_reliable_signal(analysis)
            for analysis in attributed_analyses
        )
        positive_count = sum(
            analysis.valence == "positive" and is_reliable_signal(analysis)
            for analysis in attributed_analyses
        )
        negative_count = sum(
            analysis.valence == "negative" and is_reliable_signal(analysis)
            for analysis in attributed_analyses
        )
        unique_speakers = len({msg.user_id for msg in followups})
        target_continued = (
            bool(target_followups) if targets else bool(directed_followups)
        )
        ignored = bool(followups) and not target_continued and not directed_followups
        metrics = {
            "followup_count": len(followups),
            "unique_followup_speakers": unique_speakers,
            "target_continued": target_continued,
            "directed_followup_count": len(directed_followups),
            "correction_count": correction_count,
            "positive_signal_count": positive_count,
            "negative_signal_count": negative_count,
            "ignored_in_window": ignored,
            "competing_reply_count": pending.competing_reply_count,
            "signal_analysis_methods": [
                analysis.analysis_method for analysis in attributed_analyses
            ],
            "observable_reward": (
                1.2 * int(target_continued)
                + 0.25 * positive_count
                - 1.5 * correction_count
                - 0.8 * negative_count
            ),
            "caveat": "窗口相关性，不代表因果效果；QQ发送接口未返回机器人消息ID。",
            "followups": [
                {
                    "message_id": msg.message_id,
                    "user_id": msg.user_id,
                    "is_tome": msg.is_tome,
                }
                for msg in followups
            ],
            "attributed_followups": [
                {
                    "message_id": msg.message_id,
                    "user_id": msg.user_id,
                    "is_tome": msg.is_tome,
                }
                for msg in attributed_followups
            ],
        }
        confidence = 0.25
        if targets and target_followups:
            confidence += 0.25
        if directed_followups:
            confidence += 0.2
        if correction_count or negative_count:
            confidence += 0.1
        confidence /= max(1, pending.competing_reply_count)
        return metrics, min(confidence, 0.75)

    async def _finalize(self, pending: PendingReply) -> None:
        await self._signal_service.wait_ready(
            pending.session_id,
            [message.message_id for message in pending.observed_messages],
            bot_message=pending.text,
        )
        cached_analyses = self._signal_service.get(
            pending.session_id,
            [message.message_id for message in pending.observed_messages],
            bot_message=pending.text,
        )
        if self._session_generation.get(pending.session_id, 0) != (
            pending.session_generation
        ):
            return
        analysis_by_id = {
            analysis.message_id: analysis
            for analysis in [*pending.signal_analyses, *cached_analyses]
        }
        pending.signal_analyses = list(analysis_by_id.values())
        metrics, confidence = self._score(pending)
        followup_payload = [
            {
                "message_id": msg.message_id,
                "user_id": msg.user_id,
                "timestamp": msg.timestamp.isoformat(),
                "is_tome": msg.is_tome,
                "text": msg.message.get_plain_text(),
            }
            for msg in pending.observed_messages
        ]
        async with self._db_lock:
            db = await self._ensure_db()
            await db.execute(
                """
                INSERT OR REPLACE INTO reply_effects(
                    effect_id, session_id, sent_at, reply_text, action_json,
                    followup_json, metrics_json, attribution_confidence, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pending.effect_id,
                    pending.session_id,
                    pending.sent_at.isoformat(),
                    pending.text,
                    json.dumps(pending.action, ensure_ascii=False),
                    json.dumps(followup_payload, ensure_ascii=False),
                    json.dumps(metrics, ensure_ascii=False),
                    confidence,
                    datetime.now(UTC).isoformat(),
                ),
            )
            await db.commit()
        for observer in self._effect_observers:
            try:
                await observer(pending, metrics, confidence)
            except Exception:
                # Learning must not block the next QQ event.
                logger.exception("Reply effect observer failed")
        await log_social_event(
            "reply_effect_observed",
            effect_id=pending.effect_id,
            session_id=pending.session_id,
            metrics=metrics,
            attribution_confidence=confidence,
            selected_behavior_ids=pending.selected_behavior_ids,
            selected_expression_ids=pending.selected_expression_ids,
        )

    @override
    async def abefore_agent(
        self, state: ManagerState, runtime: Runtime[ManagerContext]
    ) -> dict[str, Any] | None:
        session_id = runtime.context["session_id"]
        pending_items = self._pending.get(session_id, [])
        if not pending_items:
            return None
        incoming = list(state.get("inputs", []))
        newest_incoming_at = max(
            (self._as_utc(item.timestamp) for item in incoming),
            default=None,
        )
        remaining: list[PendingReply] = []
        competition = max(1, len(pending_items))
        for pending in pending_items:
            pending.competing_reply_count = max(
                pending.competing_reply_count, competition
            )
            sent_at = self._as_utc(pending.sent_at)
            new_observations = [
                msg
                for msg in incoming
                if self._as_utc(msg.timestamp) > sent_at
                and msg.message_id
                not in {item.message_id for item in pending.observed_messages}
            ]
            pending.observed_messages.extend(new_observations)
            if new_observations:
                new_analyses = await self._signal_service.schedule(
                    session_id=session_id,
                    target_messages=new_observations,
                    context_messages=pending.observed_messages,
                    bot_message=pending.text,
                    force=True,
                )
                analysis_by_id = {
                    analysis.message_id: analysis
                    for analysis in [*pending.signal_analyses, *new_analyses]
                }
                pending.signal_analyses = list(analysis_by_id.values())
            window_elapsed = bool(
                newest_incoming_at
                and newest_incoming_at - sent_at >= self.observation_window
            )
            if (
                len(pending.observed_messages) >= self.observation_message_limit
                or window_elapsed
            ):
                await self._schedule_finalize(pending)
            else:
                remaining.append(pending)
        if remaining:
            self._pending[session_id] = remaining[-3:]
        else:
            self._pending.pop(session_id, None)
        return None

    @override
    async def aafter_agent(
        self, state: ManagerState, runtime: Runtime[ManagerContext]
    ) -> dict[str, Any] | None:
        session_id = runtime.context["session_id"]
        for output in state["outputs"]:
            if output["type"] != "reply":
                continue
            data = output["data"]
            sent_at_raw = data.get("sent_at")
            sent_at = (
                datetime.fromisoformat(str(sent_at_raw))
                if sent_at_raw
                else datetime.now(UTC)
            )
            self._pending.setdefault(session_id, []).append(
                PendingReply(
                    effect_id=str(uuid.uuid4()),
                    session_id=session_id,
                    sent_at=self._as_utc(sent_at),
                    text=str(data.get("full_text") or ""),
                    action=dict(data.get("social_action") or {}),
                    selected_behavior_ids=list(data.get("selected_behavior_ids") or []),
                    selected_expression_ids=list(
                        data.get("selected_expression_ids") or []
                    ),
                    session_generation=self._session_generation.get(session_id, 0),
                )
            )
        return None

    async def clear_session(self, session_id: str) -> None:
        self._session_generation[session_id] = (
            self._session_generation.get(session_id, 0) + 1
        )
        self._pending.pop(session_id, None)
        await self._signal_service.clear_session(session_id)
        async with self._db_lock:
            db = await self._ensure_db()
            await db.execute(
                "DELETE FROM reply_effects WHERE session_id = ?", (session_id,)
            )
            await db.commit()

    async def aclose(self) -> None:
        if self._owns_signal_service:
            await self._signal_service.aclose()
        for pending_items in tuple(self._pending.values()):
            for pending in pending_items:
                await self._finalize(pending)
        self._pending.clear()
        task_group, self._task_group = self._task_group, None
        if task_group is not None:
            await task_group.__aexit__(None, None, None)
        if self._db is not None:
            await self._db.close()
            self._db = None


__all__ = ["ReplyEffectMiddleware"]
