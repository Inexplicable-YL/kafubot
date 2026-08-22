from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict, deque
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, cast

import aiosqlite
import anyio
import jieba
from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel, Field

from kafubot.cognition.message import QQMessage
from kafubot.cognition.plugins.base import PluginContext, PluginDefinition
from kafubot.cognition.plugins.social_signals import (
    SocialSignalAnalysis,
    SocialSignalAnalyzer,
    SocialSignalService,
    is_reliable_signal,
)
from kafubot.cognition.telemetry import log_social_event
from kafubot.cognition.types import UserMessage

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anyio.abc import TaskGroup

    from kafubot.cognition.plugins.lifecycle import (
        ObservationEvent,
        ReplyCommitted,
        ReplyPreparation,
    )

_QUESTION_MARKS = ("?", "？")


class EvidenceLink(BaseModel):
    relation: Literal["reply_to", "mentions", "same_topic", "follows_speaker"]
    source_message_id: str
    target_id: str
    confidence: float = Field(ge=0.0, le=1.0)
    inferred: bool = False


class MessageEvent(BaseModel):
    """Lossless-enough projection of fields actually exposed by QQ/OneBot."""

    message_id: str
    session_id: str
    chat_type: Literal["group", "private"]
    sender_id: str
    sender_name: str
    timestamp: datetime
    text: str
    msgcode: str
    reply_to_id: str | None = None
    mention_user_ids: list[str] = Field(default_factory=list)
    directed_to_bot: bool = False
    has_image: bool = False

    @classmethod
    def from_user_message(cls, session_id: str, message: UserMessage) -> MessageEvent:
        return cls(
            message_id=message.message_id,
            session_id=session_id,
            chat_type=message.chat_type,
            sender_id=message.user_id,
            sender_name=message.user,
            timestamp=message.timestamp,
            text=message.message.get_plain_text().strip(),
            msgcode=message.message.get_msgcode(),
            reply_to_id=message.reply_to_id,
            mention_user_ids=message.mention_user_ids,
            directed_to_bot=message.is_tome,
            has_image=bool(message.images),
        )

    def to_user_message(self) -> UserMessage:
        """Rebuild the text/transport projection needed by semantic analysis."""

        return UserMessage(
            timestamp=self.timestamp,
            user=self.sender_name,
            message=QQMessage(self.text),
            user_id=self.sender_id,
            message_id=self.message_id,
            is_tome=self.directed_to_bot,
            chat_type=self.chat_type,
            reply_to_id=self.reply_to_id,
            mention_user_ids=self.mention_user_ids,
        )


class TopicFrame(BaseModel):
    thread_id: str
    latest_message_ids: list[str] = Field(default_factory=list)
    participant_ids: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    question_message_ids: list[str] = Field(default_factory=list)
    heat: float = 0.0


class ParticipantFrame(BaseModel):
    user_id: str
    names: list[str] = Field(default_factory=list)
    recent_message_ids: list[str] = Field(default_factory=list)
    recent_threads: list[str] = Field(default_factory=list)
    directed_to_bot_count: int = 0
    observable_affect: Literal["positive", "negative", "mixed", "neutral"] = "neutral"
    affect_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    affect_evidence_message_ids: list[str] = Field(default_factory=list)


class AddresseeHypothesis(BaseModel):
    source_message_id: str
    target_kind: Literal["bot", "user", "unknown"]
    target_id: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: Literal["directed_to_bot", "reply_to", "mention", "turn_adjacency"]
    inferred: bool = False


class GroupState(BaseModel):
    active_speaker_count: int = 0
    messages_per_minute: float = 0.0
    observable_tone: Literal["positive", "negative", "mixed", "neutral"] = "neutral"
    tone_evidence_message_ids: list[str] = Field(default_factory=list)
    recent_question_ids: list[str] = Field(default_factory=list)


class ConversationFrame(BaseModel):
    session_id: str
    updated_at: datetime
    active_threads: list[TopicFrame] = Field(default_factory=list)
    participants: list[ParticipantFrame] = Field(default_factory=list)
    evidence_links: list[EvidenceLink] = Field(default_factory=list)
    addressee_hypotheses: list[AddresseeHypothesis] = Field(default_factory=list)
    group_state: GroupState = Field(default_factory=GroupState)
    bot_recently_spoke: bool = False
    observed_message_ids: list[str] = Field(default_factory=list)

    def to_prompt_text(self) -> str:
        # The graph retains a larger observation window for attribution and
        # telemetry.  The planner only needs the hottest, most recent slice.
        active_threads = sorted(
            self.active_threads, key=lambda thread: thread.heat, reverse=True
        )[:4]
        participants = sorted(
            self.participants,
            key=lambda participant: (
                participant.directed_to_bot_count,
                len(participant.recent_message_ids),
            ),
            reverse=True,
        )[:12]
        payload = {
            "session_id": self.session_id,
            "active_threads": [
                thread.model_copy(
                    update={
                        "latest_message_ids": thread.latest_message_ids[-8:],
                        "participant_ids": thread.participant_ids[-8:],
                        "keywords": thread.keywords[:5],
                        "question_message_ids": thread.question_message_ids[-4:],
                    }
                ).model_dump(mode="json")
                for thread in active_threads
            ],
            "participants": [
                participant.model_copy(
                    update={
                        "recent_message_ids": participant.recent_message_ids[-8:],
                        "recent_threads": participant.recent_threads[-4:],
                        "affect_evidence_message_ids": (
                            participant.affect_evidence_message_ids[-4:]
                        ),
                    }
                ).model_dump(mode="json")
                for participant in participants
            ],
            "evidence_links": [
                link.model_dump(mode="json") for link in self.evidence_links[-30:]
            ],
            "addressee_hypotheses": [
                item.model_dump(mode="json") for item in self.addressee_hypotheses[-24:]
            ],
            "group_state": self.group_state.model_dump(mode="json"),
            "bot_recently_spoke": self.bot_recently_spoke,
            "observed_message_ids": self.observed_message_ids[-24:],
        }
        return (
            "<conversation-structure>\n"
            "以下是系统基于QQ可观测字段和局部语义形成的结构化视图。"
            "reply_to/mentions 是平台事实；inferred=true 的边只是概率线索，不得当成事实。\n"
            f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n"
            "</conversation-structure>"
        )


class _SessionGraph:
    def __init__(self, session_id: str, *, max_events: int = 160) -> None:
        self.session_id = session_id
        self.events: dict[str, MessageEvent] = {}
        self.event_order: deque[str] = deque(maxlen=max_events)
        self.message_threads: dict[str, str] = {}
        self.thread_events: dict[str, deque[str]] = defaultdict(
            lambda: deque(maxlen=30)
        )
        self.thread_tokens: dict[str, Counter[str]] = defaultdict(Counter)
        self.links: deque[EvidenceLink] = deque(maxlen=240)
        self.last_by_sender: dict[str, str] = {}
        self.bot_recently_spoke = False
        self.social_signals: dict[str, SocialSignalAnalysis] = {}

    @classmethod
    def _tokens(cls, event: MessageEvent) -> set[str]:
        return {
            normalized
            for token in jieba.lcut(event.text)
            if len(normalized := token.casefold().strip()) >= 2
            and any(character.isalnum() for character in normalized)
        }

    def set_social_signals(self, analyses: list[SocialSignalAnalysis]) -> None:
        self.social_signals.update(
            (analysis.message_id, analysis) for analysis in analyses
        )

    def _new_thread_id(self, message_id: str) -> str:
        digest = hashlib.blake2s(
            f"{self.session_id}:{message_id}".encode(), digest_size=5
        ).hexdigest()
        return f"t_{digest}"

    def _select_thread(
        self, event: MessageEvent, tokens: set[str]
    ) -> tuple[str, float]:
        if event.reply_to_id and event.reply_to_id in self.message_threads:
            return self.message_threads[event.reply_to_id], 1.0
        best_thread = ""
        best_score = 0.0
        for thread_id, aggregate in self.thread_tokens.items():
            existing = set(aggregate)
            union = tokens | existing
            score = len(tokens & existing) / len(union) if union else 0.0
            latest = self.thread_events[thread_id]
            if latest and self.events[latest[-1]].sender_id == event.sender_id:
                score += 0.12
            if score > best_score:
                best_thread, best_score = thread_id, score
        if best_thread and best_score >= 0.18:
            return best_thread, min(best_score, 0.95)
        return self._new_thread_id(event.message_id), 0.0

    def observe_user(self, event: MessageEvent) -> None:
        if event.message_id in self.events:
            return
        if len(self.event_order) == self.event_order.maxlen:
            expired = self.event_order[0]
            expired_event = self.events.pop(expired, None)
            expired_thread = self.message_threads.pop(expired, None)
            if expired_event is not None and expired_thread is not None:
                self.thread_tokens[expired_thread].subtract(self._tokens(expired_event))
                self.thread_tokens[expired_thread] += Counter()
                with_context = self.thread_events.get(expired_thread)
                if with_context and expired in with_context:
                    with_context.remove(expired)
                if self.last_by_sender.get(expired_event.sender_id) == expired:
                    self.last_by_sender.pop(expired_event.sender_id, None)
        tokens = self._tokens(event)
        thread_id, confidence = self._select_thread(event, tokens)
        self.events[event.message_id] = event
        self.event_order.append(event.message_id)
        self.message_threads[event.message_id] = thread_id
        self.thread_events[thread_id].append(event.message_id)
        self.thread_tokens[thread_id].update(tokens)
        if event.reply_to_id:
            self.links.append(
                EvidenceLink(
                    relation="reply_to",
                    source_message_id=event.message_id,
                    target_id=event.reply_to_id,
                    confidence=1.0,
                )
            )
        for target_id in event.mention_user_ids:
            self.links.append(
                EvidenceLink(
                    relation="mentions",
                    source_message_id=event.message_id,
                    target_id=target_id,
                    confidence=1.0,
                )
            )
        if confidence:
            self.links.append(
                EvidenceLink(
                    relation="same_topic",
                    source_message_id=event.message_id,
                    target_id=thread_id,
                    confidence=confidence,
                    inferred=True,
                )
            )
        previous = self.last_by_sender.get(event.sender_id)
        if previous and previous != event.reply_to_id:
            self.links.append(
                EvidenceLink(
                    relation="follows_speaker",
                    source_message_id=event.message_id,
                    target_id=previous,
                    confidence=0.45,
                    inferred=True,
                )
            )
        self.last_by_sender[event.sender_id] = event.message_id

    def observe_bot(self) -> None:
        self.bot_recently_spoke = True

    def frame(self) -> ConversationFrame:
        recent_ids = list(self.event_order)[-60:]
        recent_set = set(recent_ids)
        threads: list[TopicFrame] = []
        for thread_id, ids in self.thread_events.items():
            visible = [item for item in ids if item in recent_set]
            if not visible:
                continue
            events = [self.events[item] for item in visible]
            participants = list(dict.fromkeys(e.sender_id for e in events))
            questions = [
                e.message_id for e in events if e.text.endswith(_QUESTION_MARKS)
            ]
            threads.append(
                TopicFrame(
                    thread_id=thread_id,
                    latest_message_ids=visible[-8:],
                    participant_ids=participants,
                    keywords=[
                        item for item, _ in self.thread_tokens[thread_id].most_common(8)
                    ],
                    question_message_ids=questions[-4:],
                    heat=min(1.0, len(visible) / 8 + len(participants) / 10),
                )
            )
        threads.sort(key=lambda item: item.heat, reverse=True)
        participant_map: dict[str, ParticipantFrame] = {}
        for event_id in recent_ids:
            event = self.events[event_id]
            participant = participant_map.setdefault(
                event.sender_id, ParticipantFrame(user_id=event.sender_id)
            )
            if event.sender_name and event.sender_name not in participant.names:
                participant.names.append(event.sender_name)
            participant.recent_message_ids = (
                participant.recent_message_ids + [event.message_id]
            )[-8:]
            thread_id = self.message_threads[event.message_id]
            if thread_id not in participant.recent_threads:
                participant.recent_threads.append(thread_id)
            if event.directed_to_bot:
                participant.directed_to_bot_count += 1
        for participant in participant_map.values():
            participant_events = [
                self.events[item]
                for item in participant.recent_message_ids
                if item in self.events
            ]
            positive = [
                event.message_id
                for event in participant_events
                if (signal := self.social_signals.get(event.message_id))
                and signal.valence == "positive"
                and is_reliable_signal(signal)
            ]
            negative = [
                event.message_id
                for event in participant_events
                if (signal := self.social_signals.get(event.message_id))
                and signal.valence == "negative"
                and is_reliable_signal(signal)
            ]
            if positive and negative:
                participant.observable_affect = "mixed"
            elif positive:
                participant.observable_affect = "positive"
            elif negative:
                participant.observable_affect = "negative"
            participant.affect_evidence_message_ids = (positive + negative)[-5:]
            participant.affect_confidence = min(
                0.8, len(participant.affect_evidence_message_ids) * 0.2
            )
        return ConversationFrame(
            session_id=self.session_id,
            updated_at=datetime.now(UTC),
            active_threads=threads[:6],
            participants=list(participant_map.values()),
            evidence_links=[
                link for link in self.links if link.source_message_id in recent_set
            ][-80:],
            addressee_hypotheses=self._addressee_hypotheses(recent_ids),
            group_state=self._group_state(recent_ids, threads),
            bot_recently_spoke=self.bot_recently_spoke,
            observed_message_ids=recent_ids,
        )

    def _addressee_hypotheses(self, recent_ids: list[str]) -> list[AddresseeHypothesis]:
        hypotheses: list[AddresseeHypothesis] = []
        for index, event_id in enumerate(recent_ids[-24:]):
            event = self.events[event_id]
            if event.directed_to_bot:
                hypotheses.append(
                    AddresseeHypothesis(
                        source_message_id=event_id,
                        target_kind="bot",
                        confidence=1.0,
                        evidence="directed_to_bot",
                    )
                )
            if event.reply_to_id:
                replied = self.events.get(event.reply_to_id)
                hypotheses.append(
                    AddresseeHypothesis(
                        source_message_id=event_id,
                        target_kind="user" if replied else "unknown",
                        target_id=replied.sender_id if replied else None,
                        confidence=1.0 if replied else 0.7,
                        evidence="reply_to",
                    )
                )
            hypotheses.extend(
                AddresseeHypothesis(
                    source_message_id=event_id,
                    target_kind="user",
                    target_id=target_id,
                    confidence=1.0,
                    evidence="mention",
                )
                for target_id in event.mention_user_ids
            )
            absolute_index = max(0, len(recent_ids) - 24) + index
            if (
                not event.directed_to_bot
                and not event.reply_to_id
                and not event.mention_user_ids
                and absolute_index > 0
            ):
                previous = self.events[recent_ids[absolute_index - 1]]
                if previous.sender_id != event.sender_id:
                    hypotheses.append(
                        AddresseeHypothesis(
                            source_message_id=event_id,
                            target_kind="user",
                            target_id=previous.sender_id,
                            confidence=0.25,
                            evidence="turn_adjacency",
                            inferred=True,
                        )
                    )
        return hypotheses[-60:]

    def _group_state(
        self, recent_ids: list[str], threads: list[TopicFrame]
    ) -> GroupState:
        events = [self.events[item] for item in recent_ids[-20:]]
        positive = [
            event.message_id
            for event in events
            if (signal := self.social_signals.get(event.message_id))
            and signal.valence == "positive"
            and is_reliable_signal(signal)
        ]
        negative = [
            event.message_id
            for event in events
            if (signal := self.social_signals.get(event.message_id))
            and signal.valence == "negative"
            and is_reliable_signal(signal)
        ]
        if positive and negative:
            tone = "mixed"
        elif positive:
            tone = "positive"
        elif negative:
            tone = "negative"
        else:
            tone = "neutral"
        message_rate = 0.0
        if len(events) > 1:
            duration = max(
                1.0,
                (events[-1].timestamp - events[0].timestamp).total_seconds(),
            )
            message_rate = min(60.0, (len(events) - 1) * 60 / duration)
        questions = [
            question_id
            for thread in threads
            for question_id in thread.question_message_ids
        ]
        return GroupState(
            active_speaker_count=len({event.sender_id for event in events}),
            messages_per_minute=round(message_rate, 2),
            observable_tone=tone,
            tone_evidence_message_ids=(positive + negative)[-8:],
            recent_question_ids=questions[-8:],
        )


class ConversationTracker:
    def __init__(
        self,
        db_path: str = ".database/qq_event_ledger.db",
        *,
        signal_analyzer: SocialSignalAnalyzer | None = None,
        signal_service: SocialSignalService | None = None,
    ) -> None:
        self.db_path = db_path
        self._graphs: dict[str, _SessionGraph] = {}
        self._loaded_sessions: set[str] = set()
        self._db: aiosqlite.Connection | None = None
        self._db_lock = anyio.Lock()
        self._task_group: TaskGroup | None = None
        self._task_group_lock = anyio.Lock()
        self._session_generation: dict[str, int] = {}
        self._owns_signal_service = signal_service is None
        self._signal_service = signal_service or SocialSignalService(
            signal_analyzer or SocialSignalAnalyzer(None)
        )
        self._signal_service.add_observer(self._accept_semantic_signals)

    async def _ensure_task_group(self) -> TaskGroup:
        if self._task_group is not None:
            return self._task_group
        async with self._task_group_lock:
            if self._task_group is None:
                task_group = anyio.create_task_group()
                await task_group.__aenter__()
                self._task_group = task_group
        return self._task_group

    async def _schedule_event_persistence(
        self, session_id: str, events: list[MessageEvent]
    ) -> None:
        if not events:
            return
        task_group = await self._ensure_task_group()
        task_group.start_soon(
            self._persist_events,
            events,
            self._session_generation.get(session_id, 0),
        )

    async def _accept_semantic_signals(
        self, session_id: str, analyses: list[SocialSignalAnalysis]
    ) -> None:
        graph = self._graphs.get(session_id)
        if graph is not None:
            graph.set_social_signals(analyses)
        await self._persist_social_signals(session_id, analyses)

    async def _ensure_db(self) -> aiosqlite.Connection:
        if self._db is None:
            self._db = await aiosqlite.connect(self.db_path)
            await self._db.execute("PRAGMA journal_mode=WAL")
            await self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS qq_message_events (
                    session_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    chat_type TEXT NOT NULL,
                    sender_id TEXT NOT NULL,
                    sender_name TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    text TEXT NOT NULL,
                    msgcode TEXT NOT NULL,
                    reply_to_id TEXT,
                    mention_user_ids_json TEXT NOT NULL,
                    directed_to_bot INTEGER NOT NULL,
                    has_image INTEGER NOT NULL,
                    recorded_at TEXT NOT NULL,
                    PRIMARY KEY (session_id, message_id)
                )
                """
            )
            await self._db.execute(
                "CREATE INDEX IF NOT EXISTS ix_qq_events_session_time "
                "ON qq_message_events(session_id, timestamp)"
            )
            await self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS qq_social_signals (
                    session_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    analysis_json TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    PRIMARY KEY (session_id, message_id)
                )
                """
            )
            await self._db.commit()
        return self._db

    async def _load_session(self, session_id: str, graph: _SessionGraph) -> None:
        if session_id in self._loaded_sessions:
            return
        async with self._db_lock:
            if session_id in self._loaded_sessions:
                return
            db = await self._ensure_db()
            cursor = await db.execute(
                """
                SELECT message_id, chat_type, sender_id, sender_name, timestamp,
                       text, msgcode, reply_to_id, mention_user_ids_json,
                       directed_to_bot, has_image
                FROM qq_message_events
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT 160
                """,
                (session_id,),
            )
            fetched_rows: list[tuple[Any, ...]] = [
                tuple(row) for row in await cursor.fetchall()
            ]
            rows = list(reversed(fetched_rows))
            await cursor.close()
            for row in rows:
                graph.observe_user(
                    MessageEvent(
                        session_id=session_id,
                        message_id=str(row[0]),
                        chat_type=row[1],
                        sender_id=str(row[2]),
                        sender_name=str(row[3]),
                        timestamp=datetime.fromisoformat(row[4]),
                        text=str(row[5]),
                        msgcode=str(row[6]),
                        reply_to_id=str(row[7]) if row[7] else None,
                        mention_user_ids=json.loads(row[8] or "[]"),
                        directed_to_bot=bool(row[9]),
                        has_image=bool(row[10]),
                    )
                )
            signal_cursor = await db.execute(
                """
                SELECT analysis_json
                FROM qq_social_signals
                WHERE session_id = ?
                ORDER BY recorded_at DESC
                LIMIT 160
                """,
                (session_id,),
            )
            signal_rows = await signal_cursor.fetchall()
            await signal_cursor.close()
            graph.set_social_signals(
                [
                    SocialSignalAnalysis.model_validate_json(str(row[0]))
                    for row in signal_rows
                ]
            )
            self._loaded_sessions.add(session_id)

    async def _persist_events(
        self, events: list[MessageEvent], generation: int
    ) -> None:
        if not events:
            return
        async with self._db_lock:
            if any(
                self._session_generation.get(event.session_id, 0) != generation
                for event in events
            ):
                return
            db = await self._ensure_db()
            await db.executemany(
                """
                INSERT OR IGNORE INTO qq_message_events(
                    session_id, message_id, chat_type, sender_id, sender_name,
                    timestamp, text, msgcode, reply_to_id,
                    mention_user_ids_json, directed_to_bot, has_image, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        event.session_id,
                        event.message_id,
                        event.chat_type,
                        event.sender_id,
                        event.sender_name,
                        event.timestamp.isoformat(),
                        event.text,
                        event.msgcode,
                        event.reply_to_id,
                        json.dumps(event.mention_user_ids, ensure_ascii=False),
                        int(event.directed_to_bot),
                        int(event.has_image),
                        datetime.now(UTC).isoformat(),
                    )
                    for event in events
                ],
            )
            await db.commit()

    async def _persist_social_signals(
        self, session_id: str, analyses: list[SocialSignalAnalysis]
    ) -> None:
        if not analyses:
            return
        async with self._db_lock:
            db = await self._ensure_db()
            await db.executemany(
                """
                INSERT OR REPLACE INTO qq_social_signals(
                    session_id, message_id, analysis_json, recorded_at
                ) VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        session_id,
                        analysis.message_id,
                        analysis.model_dump_json(),
                        datetime.now(UTC).isoformat(),
                    )
                    for analysis in analyses
                ],
            )
            await db.commit()

    async def clear_session(self, session_id: str) -> None:
        self._session_generation[session_id] = (
            self._session_generation.get(session_id, 0) + 1
        )
        self._graphs.pop(session_id, None)
        self._loaded_sessions.discard(session_id)
        if self._owns_signal_service:
            await self._signal_service.clear_session(session_id)
        async with self._db_lock:
            db = await self._ensure_db()
            await db.execute(
                "DELETE FROM qq_message_events WHERE session_id = ?", (session_id,)
            )
            await db.execute(
                "DELETE FROM qq_social_signals WHERE session_id = ?", (session_id,)
            )
            await db.commit()

    async def observe_messages(
        self,
        session_id: str,
        inputs: Sequence[UserMessage],
        history: Sequence[BaseMessage],
    ) -> ConversationFrame:
        """Update the conversation graph from the unified observation event."""
        graph = self._graphs.setdefault(session_id, _SessionGraph(session_id))
        await self._load_session(session_id, graph)
        events = [
            MessageEvent.from_user_message(session_id, message) for message in inputs
        ]
        for event in events:
            graph.observe_user(event)
        await self._schedule_event_persistence(session_id, events)
        current_inputs = {message.message_id: message for message in inputs}
        context_messages = [
            current_inputs.get(message_id, graph.events[message_id].to_user_message())
            for message_id in list(graph.event_order)[-24:]
        ]
        latest_bot_message = next(
            (
                message.text
                for message in reversed(history)
                if isinstance(message, AIMessage)
            ),
            None,
        )
        analyses = await self._signal_service.schedule(
            session_id=session_id,
            target_messages=list(inputs),
            context_messages=context_messages,
            bot_message=latest_bot_message,
            allow_semantic=False,
        )
        graph.set_social_signals(analyses)
        graph.bot_recently_spoke = any(
            isinstance(message, AIMessage) for message in list(history)[-10:]
        )
        frame = graph.frame()
        await log_social_event(
            "conversation_frame",
            session_id=session_id,
            frame=frame.model_dump(mode="json"),
            signal_analyses=[analysis.model_dump(mode="json") for analysis in analyses],
        )
        return frame

    def record_reply(self, session_id: str) -> None:
        graph = self._graphs.get(session_id)
        if graph is not None:
            graph.observe_bot()

    async def aclose(self) -> None:
        if self._owns_signal_service:
            await self._signal_service.aclose()
        task_group, self._task_group = self._task_group, None
        if task_group is not None:
            await task_group.__aexit__(None, None, None)
        if self._db is not None:
            await self._db.close()
            self._db = None


def apply(context: PluginContext, config: dict[str, Any]) -> None:
    analyzer = ConversationTracker(
        signal_service=cast("SocialSignalService", context.service("social_signals")),
        **config,
    )
    frames: dict[str, ConversationFrame] = {}

    async def observe(event: ObservationEvent) -> None:
        frames[event.session_id] = await analyzer.observe_messages(
            event.session_id,
            (event.message,),
            event.history_messages,
        )

    async def prepare(preparation: ReplyPreparation) -> str | None:
        frame = frames.get(preparation.context.session_id)
        return frame.to_prompt_text() if frame is not None else None

    async def committed(event: ReplyCommitted) -> None:
        analyzer.record_reply(event.preparation.context.session_id)

    async def clear(session_id: str) -> None:
        frames.pop(session_id, None)
        await analyzer.clear_session(session_id)

    context.resource("conversation", analyzer)
    context.on_observe(observe)
    context.on_prepare_reply(prepare)
    context.on_reply_committed(committed)
    context.clear_session(clear)


plugin = PluginDefinition(
    name="conversation",
    apply=apply,
    requires=("social_signals",),
)


__all__ = [
    "ConversationFrame",
    "ConversationTracker",
    "AddresseeHypothesis",
    "EvidenceLink",
    "GroupState",
    "MessageEvent",
    "ParticipantFrame",
    "TopicFrame",
    "plugin",
]
