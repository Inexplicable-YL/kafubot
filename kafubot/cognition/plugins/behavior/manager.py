# ruff: noqa: TC002, TC003, DTZ005, TRY400, TRY401, SIM103, PERF401
"""Behavior learning middleware.

This middleware ports MaiBot's behavior system into KafuBot's middleware
architecture:

1. Learn reusable scene-action-outcome paths from pruned chat history.
2. Maintain the learned paths with decay/disable rules.
3. Retrieve behavior references for the current planner context.
4. Evaluate later feedback for selected behaviors from subsequent pruned chat.

Unlike the upstream project, KafuBot does not have Maisaka's context message
types. The middleware therefore reuses the same message source as
`jargon_learner.py`: `summary_pruned_messages`.
"""

import hashlib
import json
import logging
import time
import uuid
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from math import exp, log
from typing import Any, Literal, cast

import anyio
import jieba
from json_repair import repair_json
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    select,
)
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from kafubot.cognition.prompts.manager import BOT_NAME

from .constants import (
    ACTOR_GROUP_COLLECTIVE,
    ACTOR_MAIBOT_SELF,
    ACTOR_OTHER_USER,
    COLD_SINGLETON_BEHAVIOR_FACTOR,
    DECAY_COOLDOWN_DAYS,
    DEFAULT_DB_URL,
    DIRECT_DOMAIN_OVERLAP_THRESHOLD,
    DIRECT_DOMAIN_OVERLAP_TOPK,
    DIRECT_LOCK_THRESHOLD,
    FEEDBACK_HISTORY_LIMIT,
    LEARNING_OBSERVED,
    LEARNING_SELF_REFLECTION,
    LOCKED_DIRECT_SPREAD_FACTOR,
    MAINTENANCE_COOLDOWN_SECONDS,
    MAINTENANCE_SOURCE,
    MAX_SCENE_CLUSTER_BEHAVIOR_IDS,
    MAX_SELECTOR_CANDIDATES,
    MAX_TAG_CLUSTER_MEMBERS,
    MIN_BEHAVIOR_SCORE,
    MIN_TAG_CLUSTER_MERGE_OVERLAP,
    POSITIVE_STALE_DECAY_AFTER_DAYS,
    PROFILE_TAG_MATCH_BONUS_CAP,
    PROFILE_TAG_MATCH_BONUS_FACTOR,
    PROFILE_TAG_MATCH_KINDS,
    SCENE_CLUSTER_REUSE_THRESHOLD,
    TAG_CLUSTER_SPREAD_DECAY,
    TAG_CLUSTER_SPREAD_TOPK,
    TAG_KIND_WEIGHTS,
    UNRESPONDED_DECAY_AFTER_DAYS,
    UNUSED_DECAY_AFTER_DAYS,
    UNUSED_DISABLE_AFTER_DAYS,
)
from .prompt import BEHAVIOR_SCENE_ANALYZE_PROMPT
from .utils import (
    clamp_score,
    clean_text,
    dump_json_list,
    load_json_list,
    strip_json_code_fence,
)

_TAG_KIND_ALIASES = {
    "attitude": "attitude",
    "domain": "domain",
    "need": "need",
    "other_attitude": "attitude",
    "other_traits": "attitude",
}
_ALLOWED_TAG_KINDS = {"attitude", "domain", "need"}
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BehaviorReferenceCandidate:
    behavior_id: int
    action: str
    outcome: str
    actor_type: str
    learning_type: str
    session_id: str = ""


@dataclass
class BehaviorPatternRetrievalResult:
    reference_text: str = ""
    behaviors: list[dict[str, Any]] = field(default_factory=list)
    scenario_profile: "BehaviorScenarioProfile" = field(
        default_factory=lambda: BehaviorScenarioProfile()
    )
    references: list[BehaviorReferenceCandidate] = field(default_factory=list)


class _BehaviorBase(DeclarativeBase):
    pass


class BehaviorExperiencePath(_BehaviorBase):
    __tablename__ = "behavior_experience_paths"
    __table_args__ = (
        UniqueConstraint(
            "session_id",
            "scene_cluster_id",
            "action_id",
            "outcome_id",
            "actor_type",
            "learning_type",
            name="uq_behavior_experience_path_scope_cluster_action_outcome_actor",
        ),
        Index("ix_behavior_experience_paths_session_enabled", "session_id", "enabled"),
        Index("ix_behavior_experience_paths_cluster", "scene_cluster_id"),
        Index("ix_behavior_experience_paths_learning_type", "learning_type"),
        Index("ix_behavior_experience_paths_actor_type", "actor_type"),
        Index("ix_behavior_experience_paths_action", "action_id"),
        Index("ix_behavior_experience_paths_outcome", "outcome_id"),
    )

    id: Mapped[int | None] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        nullable=False,
    )
    session_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
        default=None,
    )
    scene_cluster_id: Mapped[int] = mapped_column(Integer, index=True)
    action_id: Mapped[int] = mapped_column(Integer, index=True)
    outcome_id: Mapped[int] = mapped_column(Integer, index=True)
    actor_type: Mapped[str] = mapped_column(
        String(40),
        nullable=False,
        default=ACTOR_OTHER_USER,
    )
    learning_type: Mapped[str] = mapped_column(
        String(40),
        nullable=False,
        default=LEARNING_OBSERVED,
    )
    evidence_list: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    feedback_list: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    activation_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    success_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failure_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        server_default="0",
    )
    enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default="1",
    )
    last_active_time: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.now,
        index=True,
    )
    last_feedback_time: Mapped[datetime | None] = mapped_column(
        DateTime,
        default=None,
        nullable=True,
    )
    create_time: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.now,
    )
    update_time: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.now,
        index=True,
    )


class BehaviorSceneCluster(_BehaviorBase):
    __tablename__ = "behavior_scene_clusters"
    __table_args__ = (Index("ix_behavior_scene_clusters_session_id", "session_id"),)

    id: Mapped[int | None] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        nullable=False,
    )
    session_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        default=None,
    )
    tag_distribution: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    source_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    update_time: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.now,
        index=True,
    )


class BehaviorSceneTagCluster(_BehaviorBase):
    __tablename__ = "behavior_scene_tag_clusters"
    __table_args__ = (
        UniqueConstraint(
            "tag_kind",
            "tag",
            name="uq_behavior_scene_tag_cluster_kind_tag",
        ),
        Index("ix_behavior_scene_tag_clusters_kind_cluster", "tag_kind", "cluster_key"),
    )

    id: Mapped[int | None] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        nullable=False,
    )
    tag_kind: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    tag: Mapped[str] = mapped_column(Text, nullable=False)
    cluster_key: Mapped[str] = mapped_column(Text, nullable=False)
    source_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    update_time: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.now,
        index=True,
    )


class BehaviorAction(_BehaviorBase):
    __tablename__ = "behavior_actions"
    __table_args__ = (
        UniqueConstraint(
            "session_id", "action_hash", name="uq_behavior_action_scope_hash"
        ),
    )

    id: Mapped[int | None] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        nullable=False,
    )
    session_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
        default=None,
    )
    action: Mapped[str] = mapped_column(Text, nullable=False)
    action_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    create_time: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.now,
    )
    update_time: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.now,
        index=True,
    )


class BehaviorOutcome(_BehaviorBase):
    __tablename__ = "behavior_outcomes"
    __table_args__ = (
        UniqueConstraint(
            "session_id",
            "outcome_hash",
            name="uq_behavior_outcome_scope_hash",
        ),
    )

    id: Mapped[int | None] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        nullable=False,
    )
    session_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
        default=None,
    )
    outcome: Mapped[str] = mapped_column(Text, nullable=False)
    outcome_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    create_time: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.now,
    )
    update_time: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.now,
        index=True,
    )


class BehaviorDatabase:
    def __init__(
        self, *, engine: AsyncEngine | None = None, db_url: str = DEFAULT_DB_URL
    ) -> None:
        self.engine = engine or create_async_engine(
            self._normalize_async_db_url(db_url)
        )
        self._sessionmaker = async_sessionmaker(self.engine, expire_on_commit=False)
        self._schema_ready = False
        self._schema_lock = anyio.Lock()

    @staticmethod
    def _normalize_async_db_url(db_url: str) -> str:
        if db_url.startswith("sqlite+"):
            return db_url
        if db_url.startswith("sqlite:///"):
            return db_url.replace("sqlite:///", "sqlite+aiosqlite:///", 1)
        if db_url.startswith("sqlite://"):
            return db_url.replace("sqlite://", "sqlite+aiosqlite://", 1)
        return db_url

    async def ensure_schema(self) -> None:
        if self._schema_ready:
            return
        async with self._schema_lock:
            if self._schema_ready:
                return
            async with self.engine.begin() as conn:
                await conn.run_sync(_BehaviorBase.metadata.create_all)
            self._schema_ready = True

    @asynccontextmanager
    async def session(self, *, auto_commit: bool = True):
        await self.ensure_schema()
        async with self._sessionmaker() as session:
            yield session
            if auto_commit:
                await session.commit()

    async def dispose(self) -> None:
        await self.engine.dispose()


def _normalize_display_text(value: str, *, max_length: int = 180) -> str:
    normalized = clean_text(value)
    if len(normalized) <= max_length:
        return normalized
    return normalized[:max_length].rstrip()


def _normalize_name(value: str, *, max_length: int = 160) -> str:
    normalized = clean_text(value).lower()
    if len(normalized) <= max_length:
        return normalized
    return normalized[:max_length].rstrip()


def _normalize_tag_kind(raw_value: Any) -> str:
    normalized_kind = _normalize_name(str(raw_value or ""), max_length=40)
    return _TAG_KIND_ALIASES.get(normalized_kind, normalized_kind)


def _normalize_tag_value(value: str) -> str:
    display_value = _normalize_display_text(value, max_length=80)
    return _normalize_name(display_value, max_length=80)


def _text_hash(value: str) -> str:
    normalized = clean_text(value)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BehaviorScenarioTagCluster:
    kind: str
    tags: list[str] = field(default_factory=list)

    def to_prompt_payload(self) -> dict[str, Any]:
        values = self.all_values()
        return {
            "tag_name": values[0] if values else "",
            "tag_aliases": values[1:],
        }

    def all_values(self) -> list[str]:
        values: list[str] = []
        for value in self.tags:
            normalized_value = clean_text(value)
            if normalized_value and normalized_value not in values:
                values.append(normalized_value)
        return values


@dataclass(frozen=True)
class BehaviorScenarioProfile:
    summary: str = ""
    tag_clusters: list[BehaviorScenarioTagCluster] = field(default_factory=list)
    confidence: float = 0.0

    @property
    def has_signal(self) -> bool:
        return bool(self.tag_clusters)

    def tag_cluster_text(self) -> str:
        cluster_texts: list[str] = []
        for cluster in self.tag_clusters:
            if _normalize_tag_kind(cluster.kind) not in _ALLOWED_TAG_KINDS:
                continue
            values = cluster.all_values()
            if not values:
                continue
            cluster_texts.append(f"{cluster.kind}:{'/'.join(values)}")
        return " ".join(cluster_texts)

    def domain_prompt_payloads(self) -> list[dict[str, Any]]:
        return [
            cluster.to_prompt_payload()
            for cluster in self.tag_clusters
            if _normalize_tag_kind(cluster.kind) == "domain" and cluster.all_values()
        ]

    def need_prompt_payload(self) -> dict[str, Any]:
        for cluster in self.tag_clusters:
            if _normalize_tag_kind(cluster.kind) == "need" and cluster.all_values():
                return cluster.to_prompt_payload()
        return {"tag_name": "", "tag_aliases": []}

    def other_traits_prompt_payloads(self) -> list[dict[str, Any]]:
        return [
            cluster.to_prompt_payload()
            for cluster in self.tag_clusters
            if _normalize_tag_kind(cluster.kind) == "attitude" and cluster.all_values()
        ]


@dataclass(frozen=True)
class BehaviorScenarioSegment:
    segment_id: str
    title: str
    source_ids: list[str] = field(default_factory=list)
    profile: BehaviorScenarioProfile = field(default_factory=BehaviorScenarioProfile)

    def to_prompt_payload(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "title": self.title,
            "source_ids": self.source_ids,
            "profile": {
                "summary": self.profile.summary,
                "tag_clusters": self.profile.domain_prompt_payloads(),
                "need": self.profile.need_prompt_payload(),
                "other_traits": self.profile.other_traits_prompt_payloads(),
                "confidence": self.profile.confidence,
            },
        }


def _coerce_string_list(raw_value: Any, *, max_items: int = 8) -> list[str]:
    if isinstance(raw_value, list):
        raw_items = raw_value
    elif raw_value is None:
        raw_items = []
    else:
        raw_items = [raw_value]
    values: list[str] = []
    for raw_item in raw_items:
        value = clean_text(raw_item)
        if not value or value in values:
            continue
        values.append(value)
        if len(values) >= max_items:
            break
    return values


def _coerce_float(raw_value: Any) -> float:
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, value))


def _coerce_segment_id(raw_value: Any, *, fallback_index: int) -> str:
    segment_id = clean_text(raw_value)
    return segment_id[:40] if segment_id else f"s{fallback_index}"


def _coerce_segment_source_ids(raw_value: Any, *, max_items: int = 24) -> list[str]:
    raw_items = raw_value if isinstance(raw_value, list) else [raw_value]
    source_ids: list[str] = []
    for raw_item in raw_items:
        split_items = (
            raw_item.split(",")
            if isinstance(raw_item, str) and "," in raw_item
            else [raw_item]
        )
        for split_item in split_items:
            source_id = str(split_item or "").strip()
            if source_id and source_id not in source_ids:
                source_ids.append(source_id)
                if len(source_ids) >= max_items:
                    return source_ids
    return source_ids


def _coerce_tag_cluster_items(
    raw_value: Any,
    *,
    kind: str,
    max_items: int = 16,
) -> list[BehaviorScenarioTagCluster]:
    if not isinstance(raw_value, list):
        return []
    clusters: list[BehaviorScenarioTagCluster] = []
    for raw_item in raw_value:
        if not isinstance(raw_item, dict) or "kind" in raw_item:
            continue
        tag_name = clean_text(raw_item.get("tag_name"))
        raw_aliases = raw_item.get("tag_aliases")
        tags = _coerce_string_list(
            [tag_name, *_coerce_string_list(raw_aliases, max_items=8)],
            max_items=8,
        )
        if not tags:
            continue
        clusters.append(BehaviorScenarioTagCluster(kind=kind, tags=tags))
        if len(clusters) >= max_items:
            break
    return clusters


def _coerce_need_tag_cluster(raw_value: Any) -> BehaviorScenarioTagCluster | None:
    if isinstance(raw_value, dict):
        if "kind" in raw_value:
            return None
        tag_name = clean_text(raw_value.get("tag_name"))
        raw_aliases = raw_value.get("tag_aliases")
        tags = _coerce_string_list(
            [tag_name, *_coerce_string_list(raw_aliases, max_items=8)],
            max_items=8,
        )
    else:
        tags = _coerce_string_list(raw_value, max_items=1)
    if not tags:
        return None
    return BehaviorScenarioTagCluster(kind="need", tags=tags)


def _profile_from_mapping(parsed_response: dict[str, Any]) -> BehaviorScenarioProfile:
    tag_clusters = _coerce_tag_cluster_items(
        parsed_response.get("tag_clusters"), kind="domain"
    )
    tag_clusters.extend(
        _coerce_tag_cluster_items(
            parsed_response.get("other_traits"),
            kind="attitude",
            max_items=8,
        )
    )
    need_cluster = _coerce_need_tag_cluster(parsed_response.get("need"))
    if need_cluster is not None:
        tag_clusters.append(need_cluster)
    return BehaviorScenarioProfile(
        summary=clean_text(parsed_response.get("summary")),
        tag_clusters=tag_clusters,
        confidence=_coerce_float(parsed_response.get("confidence")),
    )


def parse_behavior_scenario_response(response: str) -> BehaviorScenarioProfile:
    normalized_response = strip_json_code_fence(response or "")
    if not normalized_response:
        return BehaviorScenarioProfile()
    try:
        parsed_response = json.loads(repair_json(normalized_response))
    except Exception:  # noqa: BLE001
        logger.warning("行为表现情景画像解析失败: %r", normalized_response)
        return BehaviorScenarioProfile()
    if not isinstance(parsed_response, dict):
        return BehaviorScenarioProfile()
    if isinstance(parsed_response.get("segments"), list):
        segments = parse_behavior_scenario_segments_response(response)
        return segments[0].profile if segments else BehaviorScenarioProfile()
    return _profile_from_mapping(parsed_response)


def parse_behavior_scenario_segments_response(
    response: str,
) -> list[BehaviorScenarioSegment]:
    normalized_response = strip_json_code_fence(response or "")
    if not normalized_response:
        return []
    try:
        parsed_response = json.loads(repair_json(normalized_response))
    except Exception:  # noqa: BLE001
        logger.warning("行为表现多场景片段解析失败: %r", normalized_response)
        return []
    if isinstance(parsed_response, dict) and isinstance(
        parsed_response.get("segments"), list
    ):
        raw_segments = parsed_response.get("segments") or []
    elif isinstance(parsed_response, list):
        raw_segments = parsed_response
    elif isinstance(parsed_response, dict):
        raw_segments = [
            {
                "segment_id": "s1",
                "title": parsed_response.get("summary") or "主场景",
                "source_ids": parsed_response.get("source_ids") or [],
                "profile": parsed_response,
            }
        ]
    else:
        return []

    segments: list[BehaviorScenarioSegment] = []
    seen_ids: set[str] = set()
    for index, raw_segment in enumerate(raw_segments[:3], start=1):
        if not isinstance(raw_segment, dict):
            continue
        raw_profile = raw_segment.get("profile")
        if not isinstance(raw_profile, dict):
            raw_profile = raw_segment
        profile = _profile_from_mapping(raw_profile)
        if not profile.has_signal:
            continue
        segment_id = _coerce_segment_id(
            raw_segment.get("segment_id") or raw_segment.get("id"),
            fallback_index=index,
        )
        if segment_id in seen_ids:
            segment_id = f"{segment_id}_{index}"
        seen_ids.add(segment_id)
        title = clean_text(raw_segment.get("title") or profile.summary or segment_id)
        segments.append(
            BehaviorScenarioSegment(
                segment_id=segment_id,
                title=title[:120],
                source_ids=_coerce_segment_source_ids(raw_segment.get("source_ids")),
                profile=profile,
            )
        )
    return segments


class BehaviorScenarioAnalyzer:
    async def analyze(
        self,
        *,
        context_text: str,
        sub_agent_runner: Callable[[str], Awaitable[str]] | None,
        include_context_in_prompt: bool = True,
    ) -> BehaviorScenarioProfile:
        del include_context_in_prompt
        if sub_agent_runner is None or not str(context_text or "").strip():
            return BehaviorScenarioProfile()
        try:
            raw_response = await sub_agent_runner(
                BEHAVIOR_SCENE_ANALYZE_PROMPT.format(bot_name=BOT_NAME)
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("行为表现情景画像子代理失败，已退回空画像: %s", exc)
            return BehaviorScenarioProfile()
        return parse_behavior_scenario_response(raw_response)

    async def analyze_segments(
        self,
        *,
        context_text: str,
        sub_agent_runner: Callable[[str], Awaitable[str]] | None,
    ) -> list[BehaviorScenarioSegment]:
        if sub_agent_runner is None or not str(context_text or "").strip():
            return []
        try:
            raw_response = await sub_agent_runner(
                BEHAVIOR_SCENE_ANALYZE_PROMPT.format(bot_name=BOT_NAME)
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("行为表现多场景片段分析失败，跳过本轮场景切分: %s", exc)
            return []
        return parse_behavior_scenario_segments_response(raw_response)


behavior_scenario_analyzer = BehaviorScenarioAnalyzer()


async def load_tag_cluster_lookup(
    session: AsyncSession,
) -> dict[tuple[str, str], str]:
    rows = (await session.scalars(select(BehaviorSceneTagCluster))).all()
    return {
        (row.tag_kind, row.tag): row.cluster_key
        for row in rows
        if row.tag_kind and row.tag and row.cluster_key
    }


def _tag_cluster_values(cluster: BehaviorScenarioTagCluster) -> list[str]:
    values: list[str] = []
    for value in cluster.tags:
        display_value = _normalize_display_text(value, max_length=80)
        if display_value and display_value not in values:
            values.append(display_value)
    return values


async def _select_tag_cluster_rows(
    session: AsyncSession,
    *,
    tag_kind: str,
    normalized_tags: set[str],
) -> list[BehaviorSceneTagCluster]:
    if not tag_kind or not normalized_tags:
        return []
    return list(
        (
            await session.scalars(
                select(BehaviorSceneTagCluster)
                .where(BehaviorSceneTagCluster.tag_kind == tag_kind)
                .where(BehaviorSceneTagCluster.tag.in_(normalized_tags))  # type: ignore[attr-defined]
            )
        ).all()
    )


async def _select_tag_cluster_rows_by_keys(
    session: AsyncSession,
    *,
    tag_kind: str,
    cluster_keys: set[str],
) -> list[BehaviorSceneTagCluster]:
    if not tag_kind or not cluster_keys:
        return []
    return list(
        (
            await session.scalars(
                select(BehaviorSceneTagCluster)
                .where(BehaviorSceneTagCluster.tag_kind == tag_kind)
                .where(BehaviorSceneTagCluster.cluster_key.in_(cluster_keys))  # type: ignore[attr-defined]
            )
        ).all()
    )


def _new_tag_cluster_key() -> str:
    return f"tc_{uuid.uuid4().hex}"


def _choose_merge_tag_cluster_key(
    *,
    values: Sequence[str],
    existing_rows: Sequence[BehaviorSceneTagCluster],
) -> str:
    incoming_tags = {
        _normalize_tag_value(value) for value in values if _normalize_tag_value(value)
    }
    rows_by_cluster: dict[str, list[BehaviorSceneTagCluster]] = {}
    for row in existing_rows:
        if row.cluster_key:
            rows_by_cluster.setdefault(row.cluster_key, []).append(row)

    best_key = ""
    best_score = -1
    for cluster_key, rows in rows_by_cluster.items():
        row_tags = {row.tag for row in rows if row.tag}
        overlap_count = len(incoming_tags & row_tags)
        if overlap_count < MIN_TAG_CLUSTER_MERGE_OVERLAP:
            continue
        if overlap_count > best_score:
            best_key = cluster_key
            best_score = overlap_count
    return best_key


async def _upsert_profile_tag_clusters(
    session: AsyncSession, profile: BehaviorScenarioProfile
) -> None:
    if not profile.tag_clusters:
        return
    now = datetime.now()
    for cluster in profile.tag_clusters:
        tag_kind = _normalize_tag_kind(cluster.kind)
        if tag_kind not in TAG_KIND_WEIGHTS:
            continue
        values = _tag_cluster_values(cluster)
        if not values:
            continue
        normalized_tags = {
            _normalize_tag_value(value)
            for value in values
            if _normalize_tag_value(value)
        }
        existing_rows = await _select_tag_cluster_rows(
            session,
            tag_kind=tag_kind,
            normalized_tags=normalized_tags,
        )
        existing_keys = {row.cluster_key for row in existing_rows if row.cluster_key}
        related_rows = await _select_tag_cluster_rows_by_keys(
            session,
            tag_kind=tag_kind,
            cluster_keys=existing_keys,
        )
        related_rows_by_id = {id(row): row for row in [*existing_rows, *related_rows]}
        candidate_rows = list(related_rows_by_id.values())
        merge_cluster_key = _choose_merge_tag_cluster_key(
            values=values, existing_rows=candidate_rows
        )
        selected_rows = (
            [row for row in candidate_rows if row.cluster_key == merge_cluster_key]
            if merge_cluster_key
            else []
        )
        if selected_rows:
            chosen_row = max(selected_rows, key=lambda row: int(row.source_count or 0))
            cluster_key = chosen_row.cluster_key
        else:
            cluster_key = _new_tag_cluster_key()
        if not cluster_key:
            continue
        members: list[str] = []
        for value in values:
            normalized_value = _normalize_tag_value(value)
            if normalized_value and normalized_value not in members:
                members.append(normalized_value)
        for row in selected_rows:
            if row.tag and row.tag not in members:
                members.append(row.tag)
            if len(members) >= MAX_TAG_CLUSTER_MEMBERS:
                break
        members = members[:MAX_TAG_CLUSTER_MEMBERS]
        row_by_key = {(row.tag_kind, row.tag): row for row in selected_rows}
        blocked_row_keys = {
            (row.tag_kind, row.tag)
            for row in existing_rows
            if row.cluster_key != cluster_key and row.tag_kind and row.tag
        }
        for member in members:
            normalized_member = _normalize_tag_value(member)
            if not normalized_member:
                continue
            row_key = (tag_kind, normalized_member)
            if row_key in blocked_row_keys:
                continue
            row = row_by_key.get(row_key)
            if row is None:
                row = BehaviorSceneTagCluster(
                    tag_kind=tag_kind,
                    tag=normalized_member,
                    cluster_key=cluster_key,
                    source_count=1,
                    update_time=now,
                )
            else:
                row.tag = normalized_member
                row.cluster_key = cluster_key
                row.source_count = int(row.source_count or 0) + 1
                row.update_time = now
            session.add(row)
    await session.flush()


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, value))


def _normalize_tag_name(
    tag_kind: str,
    value: str,
    *,
    tag_lookup: dict[tuple[str, str], str] | None = None,
) -> str:
    normalized_kind = _normalize_tag_kind(tag_kind)
    normalized_key = _normalize_tag_value(value)
    if normalized_kind not in TAG_KIND_WEIGHTS or not normalized_key:
        return ""
    cluster_key = (tag_lookup or {}).get(
        (normalized_kind, normalized_key), normalized_key
    )
    return f"{normalized_kind}:{cluster_key}"


def _normalize_stored_tag_name(
    tag_name: str,
    *,
    tag_lookup: dict[tuple[str, str], str] | None = None,
) -> str:
    normalized_tag = str(tag_name or "").strip()
    if ":" not in normalized_tag:
        return ""
    tag_kind, tag_value = normalized_tag.split(":", 1)
    normalized_kind = _normalize_tag_kind(tag_kind)
    if normalized_kind not in TAG_KIND_WEIGHTS:
        return ""
    normalized_value = _normalize_tag_value(tag_value)
    if normalized_kind and normalized_value.startswith("tc_"):
        return f"{normalized_kind}:{normalized_value}"
    return _normalize_tag_name(tag_kind, tag_value, tag_lookup=tag_lookup)


def _build_cluster_tag_weights(
    profile: BehaviorScenarioProfile,
    *,
    allowed_kinds: set[str] | None = None,
    tag_lookup: dict[tuple[str, str], str] | None = None,
) -> dict[str, float]:
    tag_weights: dict[str, float] = {}
    for cluster in profile.tag_clusters:
        values = _tag_cluster_values(cluster)
        if not values:
            continue
        normalized_kind = _normalize_tag_kind(cluster.kind)
        if normalized_kind not in TAG_KIND_WEIGHTS:
            continue
        if allowed_kinds is not None and normalized_kind not in allowed_kinds:
            continue
        normalized_values: list[str] = []
        for value in values:
            normalized_value = _normalize_tag_value(value)
            if normalized_value and normalized_value not in normalized_values:
                normalized_values.append(normalized_value)
        if not normalized_values:
            continue
        mapped_cluster_key = ""
        for normalized_value in normalized_values:
            mapped_cluster_key = (tag_lookup or {}).get(
                (normalized_kind, normalized_value), ""
            )
            if mapped_cluster_key:
                break
        cluster_key = mapped_cluster_key or normalized_values[0]
        tag_name = f"{normalized_kind}:{cluster_key}"
        tag_weights[tag_name] = max(
            tag_weights.get(tag_name, 0.0), TAG_KIND_WEIGHTS[normalized_kind]
        )
    return tag_weights


def _tag_weights_to_distribution(
    tag_weights: dict[str, float],
) -> list[dict[str, float | str]]:
    total_weight = sum(tag_weights.values())
    if total_weight <= 0:
        return []
    return [
        {"tag": tag, "probability": round(weight / total_weight, 6)}
        for tag, weight in sorted(tag_weights.items())
    ]


def build_scene_cluster_distribution(
    profile: BehaviorScenarioProfile,
    *,
    tag_lookup: dict[tuple[str, str], str] | None = None,
) -> list[dict[str, float | str]]:
    return _tag_weights_to_distribution(
        _build_cluster_tag_weights(
            profile, allowed_kinds={"domain"}, tag_lookup=tag_lookup
        )
    )


def build_profile_tag_distribution(
    profile: BehaviorScenarioProfile,
    *,
    tag_lookup: dict[tuple[str, str], str] | None = None,
) -> list[dict[str, float | str]]:
    return _tag_weights_to_distribution(
        _build_cluster_tag_weights(profile, tag_lookup=tag_lookup)
    )


def _distribution_to_mapping(
    distribution: Sequence[dict[str, Any]],
    *,
    tag_lookup: dict[tuple[str, str], str] | None = None,
) -> dict[str, float]:
    tag_probs: dict[str, float] = {}
    for item in distribution:
        if not isinstance(item, dict):
            continue
        tag = str(item.get("tag") or "").strip()
        if not tag:
            continue
        tag = _normalize_stored_tag_name(tag, tag_lookup=tag_lookup)
        if not tag:
            continue
        tag_kind, _ = tag.split(":", 1)
        if tag_kind not in TAG_KIND_WEIGHTS:
            continue
        try:
            probability = float(item.get("probability") or 0.0)
        except (TypeError, ValueError):
            continue
        if probability <= 0:
            continue
        tag_probs[tag] = tag_probs.get(tag, 0.0) + probability
    total_probability = sum(tag_probs.values())
    if total_probability <= 0:
        return {}
    return {
        tag: probability / total_probability for tag, probability in tag_probs.items()
    }


def _mapping_to_distribution(
    tag_probs: dict[str, float],
) -> list[dict[str, float | str]]:
    total_probability = sum(max(probability, 0.0) for probability in tag_probs.values())
    if total_probability <= 0:
        return []
    return [
        {"tag": tag, "probability": round(max(probability, 0.0) / total_probability, 6)}
        for tag, probability in sorted(tag_probs.items())
        if probability > 0
    ]


async def build_profile_tag_mapping(
    db: BehaviorDatabase,
    profile: BehaviorScenarioProfile,
) -> dict[str, float]:
    if not profile.has_signal:
        return {}
    try:
        async with db.session(auto_commit=False) as session:
            tag_lookup = await load_tag_cluster_lookup(session)
            return _distribution_to_mapping(
                build_profile_tag_distribution(profile, tag_lookup=tag_lookup),
                tag_lookup=tag_lookup,
            )
    except Exception as exc:  # noqa: BLE001
        logger.debug("构建行为画像 tag 映射失败: error=%s", exc)
        return {}


def _dump_cluster_distribution(distribution: Sequence[dict[str, Any]]) -> str:
    return json.dumps(list(distribution), ensure_ascii=False, sort_keys=True)


def _load_cluster_distribution(raw_value: Any) -> list[dict[str, Any]]:
    if isinstance(raw_value, list):
        return [item for item in raw_value if isinstance(item, dict)]
    if not isinstance(raw_value, str) or not raw_value.strip():
        return []
    try:
        parsed_value = json.loads(raw_value)
    except (TypeError, ValueError):
        return []
    return (
        [item for item in parsed_value if isinstance(item, dict)]
        if isinstance(parsed_value, list)
        else []
    )


def format_scene_cluster_distribution(
    distribution: Sequence[dict[str, Any]],
    *,
    tag_lookup: dict[tuple[str, str], str] | None = None,
) -> str:
    tag_probs = _distribution_to_mapping(distribution, tag_lookup=tag_lookup)
    if not tag_probs:
        return ""
    parts = [
        f"{tag}={probability:.3f}"
        for tag, probability in sorted(
            tag_probs.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:8]
    ]
    return "；".join(parts)


def _cluster_distribution_overlap(
    left_distribution: Sequence[dict[str, Any]],
    right_distribution: Sequence[dict[str, Any]],
    *,
    tag_lookup: dict[tuple[str, str], str] | None = None,
) -> float:
    left_probs = _distribution_to_mapping(left_distribution, tag_lookup=tag_lookup)
    right_probs = _distribution_to_mapping(right_distribution, tag_lookup=tag_lookup)
    if not left_probs or not right_probs:
        return 0.0
    shared_tags = set(left_probs) & set(right_probs)
    return round(sum(min(left_probs[tag], right_probs[tag]) for tag in shared_tags), 4)


def _build_frequency_weight_by_tag(
    clusters: Sequence[BehaviorSceneCluster],
    *,
    tag_lookup: dict[tuple[str, str], str] | None = None,
) -> dict[str, float]:
    df_by_tag: Counter[str] = Counter()
    for cluster in clusters:
        cluster_tags = _distribution_to_mapping(
            _load_cluster_distribution(cluster.tag_distribution),
            tag_lookup=tag_lookup,
        )
        df_by_tag.update(cluster_tags.keys())
    if not df_by_tag:
        return {}

    cluster_count = max(len(clusters), 1)
    raw_weights: dict[str, float] = {}
    for tag, count in df_by_tag.items():
        df_ratio = float(count) / float(cluster_count)
        idf = 1.0 + log((float(cluster_count) + 1.0) / (float(count) + 1.0))
        idf_soft = 1.0 + log(idf)
        rare_reliability = 1.0 - exp(-float(count) / 2.0)
        common_gate = 1.0 / (1.0 + (df_ratio / 0.08) ** 1.8)
        raw_weights[tag] = max(0.05, idf_soft * rare_reliability * common_gate)

    average_weight = sum(raw_weights.values()) / float(len(raw_weights))
    if average_weight <= 0:
        return dict.fromkeys(raw_weights, 1.0)
    return {tag: weight / average_weight for tag, weight in raw_weights.items()}


def _weighted_distribution_overlap(
    query_tags: dict[str, float],
    cluster_tags: dict[str, float],
    *,
    frequency_weight_by_tag: dict[str, float],
) -> float:
    shared_tags = set(query_tags) & set(cluster_tags)
    if not shared_tags:
        return 0.0
    query_weight = sum(
        probability * frequency_weight_by_tag.get(tag, 1.0)
        for tag, probability in query_tags.items()
    )
    if query_weight <= 0:
        return 0.0
    hit_weight = sum(
        min(query_tags[tag], cluster_tags[tag]) * frequency_weight_by_tag.get(tag, 1.0)
        for tag in shared_tags
    )
    return _clamp(hit_weight / query_weight, 0.0, 1.0)


def _scene_cluster_session_ids(raw_session_id: str | None) -> set[str]:
    normalized_session_id = str(raw_session_id or "").strip()
    if not normalized_session_id:
        return set()
    if normalized_session_id.startswith("["):
        try:
            parsed_value = json.loads(normalized_session_id)
        except (TypeError, ValueError):
            return {normalized_session_id}
        if isinstance(parsed_value, list):
            return {
                str(item or "").strip()
                for item in parsed_value
                if str(item or "").strip()
            }
    return {normalized_session_id}


def _scene_cluster_matches_sessions(
    raw_session_id: str | None, session_ids: set[str]
) -> bool:
    if not session_ids:
        return raw_session_id is None
    cluster_session_ids = _scene_cluster_session_ids(raw_session_id)
    return not cluster_session_ids or bool(cluster_session_ids & session_ids)


def _session_scope_condition(model: Any, session_ids: set[str]):
    if session_ids:
        return (model.session_id.in_(session_ids)) | (model.session_id.is_(None))  # type: ignore[attr-defined]
    return model.session_id.is_(None)  # type: ignore[attr-defined]


@dataclass(frozen=True)
class BehaviorGraphRefs:
    scene_cluster: BehaviorSceneCluster
    scene_cluster_id: int
    action_id: int
    outcome_id: int


def _merge_cluster_distributions(
    existing_distribution: Sequence[dict[str, Any]],
    new_distribution: Sequence[dict[str, Any]],
    *,
    existing_weight: int,
    tag_lookup: dict[tuple[str, str], str] | None = None,
) -> str:
    existing_probs = _distribution_to_mapping(
        existing_distribution, tag_lookup=tag_lookup
    )
    new_probs = _distribution_to_mapping(new_distribution, tag_lookup=tag_lookup)
    if not existing_probs:
        return _dump_cluster_distribution(new_distribution)
    merged_probs: dict[str, float] = {}
    all_tags = set(existing_probs) | set(new_probs)
    for tag in all_tags:
        merged_probs[tag] = (
            existing_probs.get(tag, 0.0) * float(existing_weight)
            + new_probs.get(tag, 0.0)
        ) / (float(existing_weight) + 1.0)
    return _dump_cluster_distribution(_mapping_to_distribution(merged_probs))


async def _upsert_scene_cluster(
    session: AsyncSession,
    *,
    session_id: str,
    profile: BehaviorScenarioProfile,
) -> BehaviorSceneCluster | None:
    await _upsert_profile_tag_clusters(session, profile)
    tag_lookup = await load_tag_cluster_lookup(session)
    distribution = build_scene_cluster_distribution(profile, tag_lookup=tag_lookup)
    if not distribution:
        return None

    cluster_candidates = [
        cluster
        for cluster in (await session.scalars(select(BehaviorSceneCluster))).all()
        if _scene_cluster_matches_sessions(cluster.session_id, {session_id})
    ]
    best_cluster: BehaviorSceneCluster | None = None
    best_overlap = 0.0
    for candidate in cluster_candidates:
        overlap = _cluster_distribution_overlap(
            _load_cluster_distribution(candidate.tag_distribution),
            distribution,
            tag_lookup=tag_lookup,
        )
        if overlap > best_overlap:
            best_cluster = candidate
            best_overlap = overlap
    cluster = (
        best_cluster
        if best_cluster is not None and best_overlap >= SCENE_CLUSTER_REUSE_THRESHOLD
        else None
    )

    now = datetime.now()
    if cluster is None:
        cluster = BehaviorSceneCluster(
            session_id=session_id,
            tag_distribution=_dump_cluster_distribution(distribution),
            source_count=1,
            update_time=now,
        )
    else:
        cluster.tag_distribution = _merge_cluster_distributions(
            _load_cluster_distribution(cluster.tag_distribution),
            distribution,
            existing_weight=max(int(cluster.source_count or 0), 1),
            tag_lookup=tag_lookup,
        )
        cluster.source_count += 1
        cluster.update_time = now
    session.add(cluster)
    await session.flush()
    return cluster


async def _upsert_action(
    session: AsyncSession, *, session_id: str, action: str
) -> BehaviorAction:
    normalized_action = _normalize_display_text(action, max_length=240)
    action_hash = _text_hash(normalized_action)
    statement = (
        select(BehaviorAction)
        .where(BehaviorAction.session_id == session_id)
        .where(BehaviorAction.action_hash == action_hash)
    )
    node = (await session.scalars(statement)).first()
    now = datetime.now()
    if node is None:
        node = BehaviorAction(
            session_id=session_id,
            action=normalized_action,
            action_hash=action_hash,
            source_count=1,
            create_time=now,
            update_time=now,
        )
    else:
        node.source_count += 1
        node.action = normalized_action
        node.update_time = now
    session.add(node)
    await session.flush()
    return node


async def _upsert_outcome(
    session: AsyncSession, *, session_id: str, outcome: str
) -> BehaviorOutcome:
    normalized_outcome = _normalize_display_text(outcome, max_length=220)
    outcome_hash = _text_hash(normalized_outcome)
    statement = (
        select(BehaviorOutcome)
        .where(BehaviorOutcome.session_id == session_id)
        .where(BehaviorOutcome.outcome_hash == outcome_hash)
    )
    node = (await session.scalars(statement)).first()
    now = datetime.now()
    if node is None:
        node = BehaviorOutcome(
            session_id=session_id,
            outcome=normalized_outcome,
            outcome_hash=outcome_hash,
            source_count=1,
            create_time=now,
            update_time=now,
        )
    else:
        node.source_count += 1
        node.outcome = normalized_outcome
        node.update_time = now
    session.add(node)
    await session.flush()
    return node


async def upsert_behavior_graph_refs(
    *,
    session: AsyncSession,
    session_id: str,
    profile: BehaviorScenarioProfile,
    scene_start: str,
    action: str,
    outcome: str,
) -> BehaviorGraphRefs | None:
    normalized_action = _normalize_display_text(action, max_length=240)
    normalized_outcome = _normalize_display_text(outcome, max_length=220)
    scene_cluster = await _upsert_scene_cluster(
        session, session_id=session_id, profile=profile
    )
    del scene_start
    if scene_cluster is None or scene_cluster.id is None:
        return None
    if not normalized_action or not normalized_outcome:
        return None
    action_node = await _upsert_action(
        session, session_id=session_id, action=normalized_action
    )
    outcome_node = await _upsert_outcome(
        session, session_id=session_id, outcome=normalized_outcome
    )
    if action_node.id is None or outcome_node.id is None:
        return None
    return BehaviorGraphRefs(
        scene_cluster=scene_cluster,
        scene_cluster_id=int(scene_cluster.id),
        action_id=int(action_node.id),
        outcome_id=int(outcome_node.id),
    )


async def _path_texts_from_session(
    session: AsyncSession,
    path: BehaviorExperiencePath,
) -> tuple[str, str]:
    action_node = await session.get(BehaviorAction, path.action_id)
    outcome_node = await session.get(BehaviorOutcome, path.outcome_id)
    action_text = action_node.action if action_node is not None else ""
    outcome_text = outcome_node.outcome if outcome_node is not None else ""
    return action_text, outcome_text


async def _path_to_dict_from_session(
    session: AsyncSession,
    path: BehaviorExperiencePath,
) -> dict[str, Any]:
    action, outcome = await _path_texts_from_session(session, path)
    return {
        "id": path.id,
        "action": action,
        "outcome": outcome,
        "scene_cluster_id": path.scene_cluster_id,
        "action_id": path.action_id,
        "outcome_id": path.outcome_id,
        "actor_type": path.actor_type,
        "learning_type": path.learning_type,
        "count": path.count,
        "activation_count": path.activation_count,
        "success_count": path.success_count,
        "failure_count": path.failure_count,
        "score": path.score,
        "enabled": path.enabled,
        "session_id": path.session_id,
        "profile_tag_distribution": _merge_profile_tag_distribution_from_evidence(
            path.evidence_list
        ),
        "last_active_time": path.last_active_time.isoformat()
        if path.last_active_time
        else "",
        "last_feedback_time": path.last_feedback_time.isoformat()
        if path.last_feedback_time
        else "",
    }


def _merge_profile_tag_distribution_from_evidence(
    evidence_list: Any,
) -> list[dict[str, float | str]]:
    tag_totals: dict[str, float] = {}
    distribution_count = 0
    for evidence_item in load_json_list(evidence_list):
        if not isinstance(evidence_item, dict):
            continue
        raw_distribution = evidence_item.get("profile_tag_distribution")
        if not isinstance(raw_distribution, list):
            continue
        local_tags: dict[str, float] = {}
        for item in raw_distribution:
            if not isinstance(item, dict):
                continue
            tag = str(item.get("tag") or "").strip()
            if not tag:
                continue
            try:
                probability = float(item.get("probability") or 0.0)
            except (TypeError, ValueError):
                continue
            if probability > 0:
                local_tags[tag] = local_tags.get(tag, 0.0) + probability
        if not local_tags:
            continue
        distribution_count += 1
        for tag, probability in local_tags.items():
            tag_totals[tag] = tag_totals.get(tag, 0.0) + probability
    if distribution_count <= 0:
        return []
    averaged_tags = {
        tag: probability / float(distribution_count)
        for tag, probability in tag_totals.items()
    }
    total_probability = sum(averaged_tags.values())
    if total_probability <= 0:
        return []
    return [
        {"tag": tag, "probability": round(probability / total_probability, 6)}
        for tag, probability in sorted(averaged_tags.items())
    ]


async def behavior_pattern_to_dict(
    db: BehaviorDatabase,
    path: BehaviorExperiencePath,
) -> dict[str, Any]:
    if path.id is None:
        return {
            "id": None,
            "action": "",
            "outcome": "",
            "actor_type": path.actor_type,
            "learning_type": path.learning_type,
            "count": path.count,
            "activation_count": path.activation_count,
            "success_count": path.success_count,
            "failure_count": path.failure_count,
            "score": path.score,
            "enabled": path.enabled,
            "session_id": path.session_id,
            "profile_tag_distribution": [],
            "last_active_time": path.last_active_time.isoformat()
            if path.last_active_time
            else "",
            "last_feedback_time": path.last_feedback_time.isoformat()
            if path.last_feedback_time
            else "",
        }
    try:
        async with db.session(auto_commit=False) as session:
            attached_path = await session.get(BehaviorExperiencePath, path.id)
            if attached_path is None:
                return {}
            return await _path_to_dict_from_session(session, attached_path)
    except Exception as exc:  # noqa: BLE001
        logger.error("读取行为经验路径文本失败: id=%s error=%s", path.id, exc)
        return {}


async def list_behavior_patterns_for_sessions(
    db: BehaviorDatabase,
    *,
    session_ids: set[str],
    include_global: bool = False,
    min_score: float = -4.0,
) -> list[BehaviorExperiencePath]:
    try:
        async with db.session(auto_commit=False) as session:
            statement = select(BehaviorExperiencePath).where(
                cast("Any", BehaviorExperiencePath.enabled).is_(True)
            )  # type: ignore[attr-defined]
            statement = statement.where(BehaviorExperiencePath.score >= min_score)
            if include_global:
                pass
            elif session_ids:
                statement = statement.where(
                    (BehaviorExperiencePath.session_id.in_(session_ids))  # type: ignore[attr-defined]
                    | (BehaviorExperiencePath.session_id.is_(None))  # type: ignore[attr-defined]
                )
            else:
                statement = statement.where(BehaviorExperiencePath.session_id.is_(None))  # type: ignore[attr-defined]
            paths = (await session.scalars(statement)).all()
            for path in paths:
                session.expunge(path)
            return list(paths)
    except Exception as exc:  # noqa: BLE001
        logger.error("读取行为经验路径候选失败: %s", exc)
        return []


async def get_behavior_pattern(
    db: BehaviorDatabase,
    path_id: int,
) -> BehaviorExperiencePath | None:
    if path_id <= 0:
        return None
    try:
        async with db.session(auto_commit=False) as session:
            path = await session.get(BehaviorExperiencePath, path_id)
            if path is not None:
                session.expunge(path)
            return path
    except Exception as exc:  # noqa: BLE001
        logger.error("读取行为经验路径失败: id=%s error=%s", path_id, exc)
        return None


async def mark_behavior_scene_links_selected(
    db: BehaviorDatabase, experience_path_id: int
) -> None:
    if experience_path_id <= 0:
        return
    now = datetime.now()
    try:
        async with db.session() as session:
            path = await session.get(BehaviorExperiencePath, experience_path_id)
            if path is None:
                return
            cluster = await session.get(BehaviorSceneCluster, path.scene_cluster_id)
            if cluster is None:
                return
            cluster.update_time = now
            session.add(cluster)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "更新行为场景簇选中状态失败: experience_id=%s error=%s",
            experience_path_id,
            exc,
        )


async def mark_behavior_pattern_selected(
    db: BehaviorDatabase,
    path_id: int,
) -> BehaviorExperiencePath | None:
    if path_id <= 0:
        return None
    now = datetime.now()
    try:
        async with db.session() as session:
            path = await session.get(BehaviorExperiencePath, path_id)
            if path is None:
                return None
            path.activation_count += 1
            path.last_active_time = now
            path.update_time = now
            session.add(path)
            await session.flush()
            await session.refresh(path)
            session.expunge(path)
            selected_path = path
    except Exception as exc:  # noqa: BLE001
        logger.error("更新行为经验路径激活状态失败: id=%s error=%s", path_id, exc)
        return None
    await mark_behavior_scene_links_selected(db, path_id)
    return selected_path


async def _score_scene_clusters_by_direct_domain_overlap(
    session: AsyncSession,
    *,
    profile: BehaviorScenarioProfile,
    session_ids: set[str],
    include_global: bool,
) -> tuple[dict[int, float], dict[str, Any]]:
    tag_lookup = await load_tag_cluster_lookup(session)
    direct_tags = _distribution_to_mapping(
        build_scene_cluster_distribution(profile, tag_lookup=tag_lookup),
        tag_lookup=tag_lookup,
    )
    if not direct_tags:
        return {}, {"direct_tag_count": 0, "cluster_count": 0}
    clusters = [
        cluster
        for cluster in (await session.scalars(select(BehaviorSceneCluster))).all()
        if include_global
        or _scene_cluster_matches_sessions(cluster.session_id, session_ids)
    ]
    frequency_weight_by_tag = _build_frequency_weight_by_tag(
        clusters, tag_lookup=tag_lookup
    )
    cluster_scores: dict[int, float] = {}
    for cluster in clusters:
        if cluster.id is None:
            continue
        cluster_tags = _distribution_to_mapping(
            _load_cluster_distribution(cluster.tag_distribution),
            tag_lookup=tag_lookup,
        )
        if not cluster_tags:
            continue
        score = _weighted_distribution_overlap(
            direct_tags,
            cluster_tags,
            frequency_weight_by_tag=frequency_weight_by_tag,
        )
        if score < DIRECT_DOMAIN_OVERLAP_THRESHOLD:
            continue
        cluster_scores[int(cluster.id)] = round(score * 2.0, 4)
    sorted_cluster_scores = dict(
        sorted(cluster_scores.items(), key=lambda item: item[1], reverse=True)[
            :DIRECT_DOMAIN_OVERLAP_TOPK
        ]
    )
    return sorted_cluster_scores, {
        "direct_tag_count": len(direct_tags),
        "cluster_count": len(sorted_cluster_scores),
        "frequency_weight_enabled": True,
    }


def _build_tag_cluster_adjacency(
    clusters: Sequence[BehaviorSceneCluster],
    *,
    tag_lookup: dict[tuple[str, str], str] | None = None,
) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = {}
    for cluster in clusters:
        cluster_tags = _distribution_to_mapping(
            _load_cluster_distribution(cluster.tag_distribution),
            tag_lookup=tag_lookup,
        )
        tag_names = sorted(cluster_tags)
        if len(tag_names) < 2:
            for tag_name in tag_names:
                adjacency.setdefault(tag_name, set())
            continue
        for left_index, left_tag in enumerate(tag_names):
            adjacency.setdefault(left_tag, set())
            for right_tag in tag_names[left_index + 1 :]:
                adjacency.setdefault(right_tag, set())
                adjacency[left_tag].add(right_tag)
                adjacency[right_tag].add(left_tag)
    return adjacency


def _expand_tag_cluster_weights(
    direct_tags: dict[str, float],
    adjacency: dict[str, set[str]],
    *,
    max_depth: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    direct_tag_names = set(direct_tags)
    tag_weights = dict.fromkeys(direct_tag_names, 1.0)
    visited_tags = set(direct_tag_names)
    frontier = set(direct_tag_names)
    hop_counts: dict[int, int] = {0: len(direct_tag_names)}
    for depth in range(1, max_depth + 1):
        next_frontier: set[str] = set()
        for tag in frontier:
            next_frontier.update(adjacency.get(tag, set()))
        next_frontier -= visited_tags
        if not next_frontier:
            hop_counts[depth] = 0
            frontier = set()
            continue
        weight = TAG_CLUSTER_SPREAD_DECAY**depth
        for tag in next_frontier:
            tag_weights[tag] = weight
        visited_tags.update(next_frontier)
        frontier = next_frontier
        hop_counts[depth] = len(next_frontier)
    return tag_weights, {
        "direct_tag_count": len(direct_tag_names),
        "expanded_tag_count": max(0, len(tag_weights) - len(direct_tag_names)),
        "hop_counts": hop_counts,
        "total_query_tag_count": len(tag_weights),
    }


async def _score_scene_clusters_by_tag_cluster_spread(
    session: AsyncSession,
    *,
    profile: BehaviorScenarioProfile,
    session_ids: set[str],
    include_global: bool,
    max_depth: int,
) -> tuple[dict[int, float], dict[str, Any]]:
    tag_lookup = await load_tag_cluster_lookup(session)
    direct_tags = _distribution_to_mapping(
        build_scene_cluster_distribution(profile, tag_lookup=tag_lookup),
        tag_lookup=tag_lookup,
    )
    if not direct_tags:
        return {}, {
            "direct_tag_count": 0,
            "expanded_tag_count": 0,
            "hop_counts": {0: 0},
            "total_query_tag_count": 0,
            "cluster_count": 0,
        }
    clusters = [
        cluster
        for cluster in (await session.scalars(select(BehaviorSceneCluster))).all()
        if include_global
        or _scene_cluster_matches_sessions(cluster.session_id, session_ids)
    ]
    adjacency = _build_tag_cluster_adjacency(clusters, tag_lookup=tag_lookup)
    frequency_weight_by_tag = _build_frequency_weight_by_tag(
        clusters, tag_lookup=tag_lookup
    )
    query_tag_weights, debug_payload = _expand_tag_cluster_weights(
        direct_tags,
        adjacency,
        max_depth=max_depth,
    )
    total_query_weight = sum(
        query_weight * frequency_weight_by_tag.get(tag, 1.0)
        for tag, query_weight in query_tag_weights.items()
    )
    if total_query_weight <= 0:
        debug_payload["cluster_count"] = 0
        debug_payload["frequency_weight_enabled"] = True
        return {}, debug_payload

    cluster_scores: dict[int, float] = {}
    for cluster in clusters:
        if cluster.id is None:
            continue
        cluster_tags = _distribution_to_mapping(
            _load_cluster_distribution(cluster.tag_distribution),
            tag_lookup=tag_lookup,
        )
        if not cluster_tags:
            continue
        shared_tags = set(query_tag_weights) & set(cluster_tags)
        if not shared_tags:
            continue
        hit_weight = sum(
            query_tag_weights[tag] * frequency_weight_by_tag.get(tag, 1.0)
            for tag in shared_tags
        )
        hit_ratio = hit_weight / total_query_weight
        cluster_scores[int(cluster.id)] = round(hit_ratio * 2.0, 4)

    sorted_cluster_scores = dict(
        sorted(cluster_scores.items(), key=lambda item: item[1], reverse=True)[
            :TAG_CLUSTER_SPREAD_TOPK
        ]
    )
    debug_payload["cluster_count"] = len(sorted_cluster_scores)
    debug_payload["frequency_weight_enabled"] = True
    return sorted_cluster_scores, debug_payload


async def _score_behavior_clusters(
    session: AsyncSession,
    *,
    cluster_scores: dict[int, float],
    session_ids: set[str],
    include_global: bool,
) -> dict[int, float]:
    if not cluster_scores:
        return {}
    statement = select(BehaviorExperiencePath).where(
        BehaviorExperiencePath.scene_cluster_id.in_(set(cluster_scores))  # type: ignore[attr-defined]
    )
    if not include_global:
        statement = statement.where(
            _session_scope_condition(BehaviorExperiencePath, session_ids)
        )
    behavior_scores: dict[int, float] = {}
    for path in (await session.scalars(statement)).all():
        if path.id is None or not path.enabled:
            continue
        cluster_score = cluster_scores.get(path.scene_cluster_id, 0.0)
        if cluster_score <= 0:
            continue
        history_bonus = 1.0 + min(float(path.count or 0), 20.0) * 0.02
        score = cluster_score * history_bonus
        behavior_scores[path.id] = behavior_scores.get(path.id, 0.0) + score
    return behavior_scores


async def retrieve_behavior_scores_from_scene_clusters(
    db: BehaviorDatabase,
    *,
    session_ids: set[str],
    include_global: bool,
    profile: BehaviorScenarioProfile,
    max_count: int = MAX_SCENE_CLUSTER_BEHAVIOR_IDS,
    retrieval_mode: Literal[
        "direct_domain_overlap", "tag_cluster_spread_1", "tag_cluster_spread_2"
    ] = "tag_cluster_spread_1",
) -> dict[int, float]:
    if not profile.tag_clusters:
        return {}
    active_retrieval_mode = (
        retrieval_mode
        if retrieval_mode
        in {"direct_domain_overlap", "tag_cluster_spread_1", "tag_cluster_spread_2"}
        else "tag_cluster_spread_1"
    )
    try:
        async with db.session(auto_commit=False) as session:
            behavior_scores: dict[int, float] = {}
            if active_retrieval_mode == "direct_domain_overlap":
                (
                    cluster_scores,
                    _,
                ) = await _score_scene_clusters_by_direct_domain_overlap(
                    session,
                    profile=profile,
                    session_ids=session_ids,
                    include_global=include_global,
                )
                behavior_cluster_scores = await _score_behavior_clusters(
                    session,
                    cluster_scores=cluster_scores,
                    session_ids=session_ids,
                    include_global=include_global,
                )
                for experience_path_id, score in behavior_cluster_scores.items():
                    behavior_scores[experience_path_id] = (
                        behavior_scores.get(experience_path_id, 0.0) + score
                    )
            else:
                spread_depth = (
                    1 if active_retrieval_mode == "tag_cluster_spread_1" else 2
                )
                (
                    direct_cluster_scores,
                    _,
                ) = await _score_scene_clusters_by_direct_domain_overlap(
                    session,
                    profile=profile,
                    session_ids=session_ids,
                    include_global=include_global,
                )
                (
                    spread_cluster_scores,
                    _,
                ) = await _score_scene_clusters_by_tag_cluster_spread(
                    session,
                    profile=profile,
                    session_ids=session_ids,
                    include_global=include_global,
                    max_depth=spread_depth,
                )
                direct_behavior_scores = await _score_behavior_clusters(
                    session,
                    cluster_scores=direct_cluster_scores,
                    session_ids=session_ids,
                    include_global=include_global,
                )
                spread_behavior_scores = await _score_behavior_clusters(
                    session,
                    cluster_scores=spread_cluster_scores,
                    session_ids=session_ids,
                    include_global=include_global,
                )
                direct_top_score = max(direct_behavior_scores.values(), default=0.0)
                if direct_top_score >= DIRECT_LOCK_THRESHOLD:
                    behavior_scores.update(direct_behavior_scores)
                    for experience_path_id, score in spread_behavior_scores.items():
                        protected_score = (
                            float(score or 0.0) * LOCKED_DIRECT_SPREAD_FACTOR
                        )
                        behavior_scores[experience_path_id] = max(
                            behavior_scores.get(experience_path_id, 0.0),
                            protected_score,
                        )
                else:
                    behavior_scores.update(spread_behavior_scores)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "行为场景簇检索失败: session_ids=%s mode=%s error=%s",
            session_ids,
            active_retrieval_mode,
            exc,
        )
        return {}

    return dict(
        sorted(behavior_scores.items(), key=lambda item: item[1], reverse=True)[
            :max_count
        ]
    )


@dataclass(frozen=True)
class BehaviorPatternMaintenanceResult:
    session_id: str
    scanned_count: int = 0
    decayed_count: int = 0
    disabled_count: int = 0
    merged_count: int = 0
    skipped_reason: str = ""
    touched_pattern_ids: list[int] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return (
            self.decayed_count > 0 or self.disabled_count > 0 or self.merged_count > 0
        )


class BehaviorPatternMaintenanceService:
    def __init__(self, db: BehaviorDatabase) -> None:
        self._db = db
        self._last_run_at_by_session_id: dict[str, float] = {}

    async def maybe_maintain_session(
        self,
        *,
        session_id: str,
        related_session_ids: set[str] | None = None,
        force: bool = False,
    ) -> BehaviorPatternMaintenanceResult:
        normalized_session_id = str(session_id or "").strip()
        if not normalized_session_id:
            return BehaviorPatternMaintenanceResult(
                session_id="", skipped_reason="empty_session_id"
            )
        current_time = time.time()
        if not force:
            last_run_at = self._last_run_at_by_session_id.get(
                normalized_session_id, 0.0
            )
            if current_time - last_run_at < MAINTENANCE_COOLDOWN_SECONDS:
                return BehaviorPatternMaintenanceResult(
                    session_id=normalized_session_id,
                    skipped_reason="cooldown",
                )
        result = await self.maintain_session(
            session_id=normalized_session_id,
            related_session_ids=related_session_ids,
        )
        self._last_run_at_by_session_id[normalized_session_id] = current_time
        return result

    async def maintain_session(
        self,
        *,
        session_id: str,
        related_session_ids: set[str] | None = None,
        now: datetime | None = None,
    ) -> BehaviorPatternMaintenanceResult:
        normalized_session_id = str(session_id or "").strip()
        target_session_ids = self._normalize_session_ids(related_session_ids)
        target_session_ids.add(normalized_session_id)
        if not normalized_session_id or not target_session_ids:
            return BehaviorPatternMaintenanceResult(
                session_id=normalized_session_id, skipped_reason="empty_session_id"
            )
        maintenance_time = now or datetime.now()
        result = BehaviorPatternMaintenanceResult(session_id=normalized_session_id)
        try:
            async with self._db.session() as session:
                statement = select(BehaviorExperiencePath).where(
                    BehaviorExperiencePath.session_id.in_(target_session_ids)  # type: ignore[attr-defined]
                )
                patterns = list((await session.scalars(statement)).all())
                result = BehaviorPatternMaintenanceResult(
                    session_id=normalized_session_id,
                    scanned_count=len(patterns),
                )
                if not patterns:
                    return result
                touched_pattern_ids: list[int] = []
                decayed_count = 0
                disabled_count = 0
                for pattern in patterns:
                    if not pattern.enabled:
                        continue
                    decay_result = self._apply_decay(pattern, now=maintenance_time)
                    if decay_result.decayed:
                        decayed_count += 1
                        if pattern.id is not None:
                            touched_pattern_ids.append(pattern.id)
                    if decay_result.disabled:
                        disabled_count += 1
                        if pattern.id is not None:
                            touched_pattern_ids.append(pattern.id)
                    if decay_result.decayed or decay_result.disabled:
                        session.add(pattern)
                result = BehaviorPatternMaintenanceResult(
                    session_id=normalized_session_id,
                    scanned_count=len(patterns),
                    decayed_count=decayed_count,
                    disabled_count=disabled_count,
                    merged_count=0,
                    touched_pattern_ids=self._dedupe_ids(touched_pattern_ids),
                )
                if result.changed:
                    logger.info(
                        "行为表现维护完成: session_id=%s 扫描=%s 衰减=%s 禁用=%s",
                        normalized_session_id,
                        result.scanned_count,
                        result.decayed_count,
                        result.disabled_count,
                    )
                return result
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "行为表现维护失败: session_id=%s error=%s", normalized_session_id, exc
            )
            return BehaviorPatternMaintenanceResult(
                session_id=normalized_session_id,
                skipped_reason="error",
            )

    @staticmethod
    def _normalize_session_ids(session_ids: set[str] | None) -> set[str]:
        if not session_ids:
            return set()
        return {
            str(session_id or "").strip()
            for session_id in session_ids
            if str(session_id or "").strip()
        }

    @staticmethod
    def _dedupe_ids(pattern_ids: Sequence[int]) -> list[int]:
        deduped_ids: list[int] = []
        for pattern_id in pattern_ids:
            if pattern_id not in deduped_ids:
                deduped_ids.append(pattern_id)
        return deduped_ids

    @staticmethod
    def _days_since(now: datetime, timestamp: datetime | None) -> int:
        if timestamp is None:
            return 0
        return max(0, (now - timestamp).days)

    @staticmethod
    def _latest_activity_time(pattern: BehaviorExperiencePath) -> datetime:
        timestamps = [
            timestamp
            for timestamp in [
                pattern.last_active_time,
                pattern.last_feedback_time,
                pattern.create_time,
            ]
            if timestamp is not None
        ]
        return max(timestamps) if timestamps else datetime.now()

    def _last_maintenance_time(
        self, pattern: BehaviorExperiencePath
    ) -> datetime | None:
        feedback_items = load_json_list(pattern.feedback_list)
        maintenance_times: list[datetime] = []
        for feedback_item in feedback_items:
            if not isinstance(feedback_item, dict):
                continue
            if feedback_item.get("source") != MAINTENANCE_SOURCE:
                continue
            raw_created_at = str(feedback_item.get("created_at") or "").strip()
            if not raw_created_at:
                continue
            try:
                maintenance_times.append(datetime.fromisoformat(raw_created_at))
            except ValueError:
                continue
        return max(maintenance_times) if maintenance_times else None

    def _append_maintenance_event(
        self,
        pattern: BehaviorExperiencePath,
        *,
        now: datetime,
        score_delta: float,
        status: str,
        reason: str,
        outcome: str = "",
        extra: dict[str, Any] | None = None,
    ) -> None:
        feedback_items = load_json_list(pattern.feedback_list)
        event = {
            "score_delta": score_delta,
            "status": status,
            "reason": reason,
            "outcome": outcome,
            "session_id": pattern.session_id,
            "created_at": now.isoformat(timespec="seconds"),
            "source": MAINTENANCE_SOURCE,
        }
        if extra:
            event.update(extra)
        feedback_items.append(event)
        pattern.feedback_list = dump_json_list(feedback_items[-FEEDBACK_HISTORY_LIMIT:])

    @dataclass(frozen=True)
    class _DecayResult:
        decayed: bool = False
        disabled: bool = False

    def _apply_decay(
        self,
        pattern: BehaviorExperiencePath,
        *,
        now: datetime,
    ) -> _DecayResult:
        last_maintenance_time = self._last_maintenance_time(pattern)
        if (
            last_maintenance_time is not None
            and self._days_since(now, last_maintenance_time) < DECAY_COOLDOWN_DAYS
        ):
            return self._DecayResult()
        latest_activity_time = self._latest_activity_time(pattern)
        inactive_days = self._days_since(now, latest_activity_time)
        score_delta, reason = self._calculate_decay(
            pattern, inactive_days=inactive_days
        )
        decayed = score_delta < 0
        disabled = False
        if decayed:
            pattern.score = clamp_score(float(pattern.score or 0.0) + score_delta)
            self._append_maintenance_event(
                pattern,
                now=now,
                score_delta=score_delta,
                status="maintenance_decay",
                reason=reason,
            )
        if self._should_disable(pattern, inactive_days=inactive_days):
            pattern.enabled = False
            disabled = True
            self._append_maintenance_event(
                pattern,
                now=now,
                score_delta=0.0,
                status="maintenance_disable",
                reason="长期缺少有效强化或负反馈过多，暂时停止作为行为表现候选。",
            )
        if decayed or disabled:
            pattern.update_time = now
        return self._DecayResult(decayed=decayed, disabled=disabled)

    @staticmethod
    def _calculate_decay(
        pattern: BehaviorExperiencePath,
        *,
        inactive_days: int,
    ) -> tuple[float, str]:
        count = int(pattern.count or 0)
        activation_count = int(pattern.activation_count or 0)
        success_count = int(pattern.success_count or 0)
        failure_count = int(pattern.failure_count or 0)

        if (
            count <= 1
            and activation_count <= 0
            and inactive_days >= UNUSED_DECAY_AFTER_DAYS
        ):
            periods = min(4, max(1, inactive_days // UNUSED_DECAY_AFTER_DAYS))
            return (
                -0.35 * periods,
                "一次性观察长期未被再次强化，按用进废退规则轻度衰减。",
            )
        if (
            activation_count > 0
            and success_count <= 0
            and inactive_days >= UNRESPONDED_DECAY_AFTER_DAYS
        ):
            periods = min(3, max(1, inactive_days // UNRESPONDED_DECAY_AFTER_DAYS))
            return -0.25 * periods, "被选择后长期没有成功反馈，降低后续抽样权重。"
        if (
            success_count > 0
            and failure_count <= success_count
            and inactive_days >= POSITIVE_STALE_DECAY_AFTER_DAYS
        ):
            return -0.15, "曾经有效但长期未再出现，轻微衰减以给新行为让路。"
        return 0.0, ""

    @staticmethod
    def _should_disable(
        pattern: BehaviorExperiencePath,
        *,
        inactive_days: int,
    ) -> bool:
        score = float(pattern.score or 0.0)
        count = int(pattern.count or 0)
        activation_count = int(pattern.activation_count or 0)
        success_count = int(pattern.success_count or 0)
        failure_count = int(pattern.failure_count or 0)
        if score <= MIN_BEHAVIOR_SCORE and (
            failure_count >= 2 or activation_count >= 3
        ):
            return True
        if (
            count <= 1
            and activation_count <= 0
            and inactive_days >= UNUSED_DISABLE_AFTER_DAYS
            and score <= -3.0
        ):
            return True
        if failure_count >= 3 and success_count <= 0 and score <= -4.0:
            return True
        return False


class BehaviorPatternSelector:
    def __init__(
        self,
        db: BehaviorDatabase,
        maintenance: BehaviorPatternMaintenanceService,
        group_resolver: Callable[[str], set[str] | tuple[set[str], bool]] | None = None,
    ) -> None:
        self._db = db
        self._maintenance = maintenance
        self._group_resolver = group_resolver

    @staticmethod
    def _build_compact_scenario_text(scenario_profile: BehaviorScenarioProfile) -> str:
        if not scenario_profile.has_signal:
            return "无可用场景画像。"
        lines = []
        if scenario_profile.summary:
            lines.append(f"场景摘要：{scenario_profile.summary}")
        if scenario_profile.tag_clusters:
            lines.append(f"场景标签：{scenario_profile.tag_cluster_text()}")
        return "\n".join(lines) if lines else "无可用场景画像。"

    @staticmethod
    def _format_priority_label(index: int, total_count: int) -> str:
        if index <= 1:
            return "高"
        if total_count <= 2 or index <= 2:
            return "中"
        return "低"

    def _resolve_behavior_group_scope(self, session_id: str) -> tuple[set[str], bool]:
        related_session_ids = {session_id} if session_id else set()
        has_global_share = False
        if self._group_resolver is None or not session_id:
            return related_session_ids, has_global_share
        try:
            resolved = self._group_resolver(session_id)
        except Exception:  # noqa: BLE001
            logger.exception(
                "Failed to resolve behavior group scope for %s", session_id
            )
            return related_session_ids, has_global_share
        if isinstance(resolved, tuple):
            raw_ids, has_global_share = resolved
        else:
            raw_ids = resolved
        related_session_ids.update(str(item) for item in raw_ids if str(item).strip())
        return related_session_ids, bool(has_global_share)

    async def has_candidates(self, session_id: str) -> bool:
        """Cheap demand gate used before spending a scene-analysis model call."""

        related_session_ids, has_global_share = self._resolve_behavior_group_scope(
            session_id
        )
        patterns = await list_behavior_patterns_for_sessions(
            self._db,
            session_ids=related_session_ids,
            include_global=has_global_share,
        )
        return bool(patterns)

    @staticmethod
    def _candidate_weight(candidate: dict[str, Any]) -> float:
        count = max(float(candidate.get("count") or 0.0), 0.0)
        score = float(candidate.get("score") or 0.0)
        success_count = max(float(candidate.get("success_count") or 0.0), 0.0)
        failure_count = max(float(candidate.get("failure_count") or 0.0), 0.0)
        activation_count = max(float(candidate.get("activation_count") or 0.0), 0.0)
        learning_type = str(candidate.get("learning_type") or "").strip()
        self_feedback_bonus = 0.15 if learning_type == LEARNING_SELF_REFLECTION else 0.0
        weight = max(
            0.2,
            1.0
            + count * 0.15
            + score * 0.7
            + success_count * 0.4
            - failure_count * 0.6
            - activation_count * 0.03
            + self_feedback_bonus,
        )
        is_cold_singleton = (
            count <= 1
            and activation_count <= 0
            and success_count <= 0
            and failure_count <= 0
            and abs(score) <= 0.0001
        )
        if is_cold_singleton:
            weight *= COLD_SINGLETON_BEHAVIOR_FACTOR
        return max(0.2, weight)

    @staticmethod
    def _profile_tag_mapping_from_distribution(distribution: Any) -> dict[str, float]:
        if not isinstance(distribution, list):
            return {}
        tag_probs: dict[str, float] = {}
        for item in distribution:
            if not isinstance(item, dict):
                continue
            tag = str(item.get("tag") or "").strip()
            if ":" not in tag:
                continue
            tag_kind = tag.split(":", 1)[0]
            if tag_kind not in PROFILE_TAG_MATCH_KINDS:
                continue
            try:
                probability = float(item.get("probability") or 0.0)
            except (TypeError, ValueError):
                continue
            if probability > 0:
                tag_probs[tag] = tag_probs.get(tag, 0.0) + probability
        total_probability = sum(tag_probs.values())
        if total_probability <= 0:
            return {}
        return {
            tag: probability / total_probability
            for tag, probability in tag_probs.items()
        }

    @classmethod
    def _profile_tag_match_score(
        cls,
        candidate: dict[str, Any],
        *,
        query_profile_tags: dict[str, float],
    ) -> float:
        if not query_profile_tags:
            return 0.0
        candidate_profile_tags = cls._profile_tag_mapping_from_distribution(
            candidate.get("profile_tag_distribution")
        )
        if not candidate_profile_tags:
            return 0.0
        shared_tags = set(query_profile_tags) & set(candidate_profile_tags)
        if not shared_tags:
            return 0.0
        return sum(
            min(query_profile_tags[tag], candidate_profile_tags[tag])
            for tag in shared_tags
        )

    async def _rank_candidates_by_scene_cluster(
        self,
        candidates: list[dict[str, Any]],
        *,
        scene_cluster_scores: dict[int, float],
        scenario_profile: BehaviorScenarioProfile | None = None,
        max_count: int,
    ) -> list[dict[str, Any]]:
        if not scene_cluster_scores:
            return []
        matched_candidates: list[dict[str, Any]] = []
        query_profile_tags = (
            self._profile_tag_mapping_from_distribution(
                [
                    {"tag": tag, "probability": probability}
                    for tag, probability in (
                        await build_profile_tag_mapping(self._db, scenario_profile)
                    ).items()
                ]
            )
            if scenario_profile is not None and scenario_profile.has_signal
            else {}
        )
        for candidate in candidates:
            candidate_id = candidate.get("id")
            if not isinstance(candidate_id, int):
                continue
            cluster_score = scene_cluster_scores.get(candidate_id)
            if cluster_score is None:
                continue
            candidate = dict(candidate)
            candidate["scene_cluster_score"] = round(cluster_score, 4)
            candidate_weight = self._candidate_weight(candidate)
            profile_tag_match_score = self._profile_tag_match_score(
                candidate,
                query_profile_tags=query_profile_tags,
            )
            profile_tag_bonus = 1.0 + min(
                PROFILE_TAG_MATCH_BONUS_CAP,
                profile_tag_match_score * PROFILE_TAG_MATCH_BONUS_FACTOR,
            )
            candidate["profile_tag_match_score"] = round(profile_tag_match_score, 4)
            candidate["behavior_retrieval_score"] = round(
                float(cluster_score) * candidate_weight * profile_tag_bonus,
                4,
            )
            matched_candidates.append(candidate)
        matched_candidates.sort(
            key=lambda candidate: (
                float(candidate.get("behavior_retrieval_score") or 0.0),
                float(candidate.get("scene_cluster_score") or 0.0),
                float(candidate.get("profile_tag_match_score") or 0.0),
                int(candidate.get("success_count") or 0),
                int(candidate.get("id") or 0),
            ),
            reverse=True,
        )
        return matched_candidates[:max_count]

    async def _load_behavior_candidates(
        self,
        session_id: str,
        *,
        scenario_profile: BehaviorScenarioProfile | None = None,
        context_text: str = "",
        run_maintenance: bool = True,
        max_count: int = MAX_SELECTOR_CANDIDATES,
    ) -> list[dict[str, Any]]:
        related_session_ids, has_global_share = self._resolve_behavior_group_scope(
            session_id
        )
        if run_maintenance:
            await self._maintenance.maybe_maintain_session(
                session_id=session_id,
                related_session_ids=related_session_ids,
            )
        patterns = await list_behavior_patterns_for_sessions(
            self._db,
            session_ids=related_session_ids,
            include_global=has_global_share,
        )
        candidates: list[dict[str, Any]] = []
        for pattern in patterns:
            if pattern.id is None:
                continue
            candidate = await behavior_pattern_to_dict(self._db, pattern)
            if not candidate:
                continue
            if not candidate.get("action") or not candidate.get("outcome"):
                continue
            candidates.append(candidate)
        if scenario_profile is not None and scenario_profile.has_signal:
            scene_cluster_scores = await retrieve_behavior_scores_from_scene_clusters(
                self._db,
                session_ids=related_session_ids,
                include_global=has_global_share,
                profile=scenario_profile,
            )
            scene_cluster_ranked_candidates = (
                await self._rank_candidates_by_scene_cluster(
                    candidates,
                    scene_cluster_scores=scene_cluster_scores,
                    scenario_profile=scenario_profile,
                    max_count=max_count,
                )
            )
            if scene_cluster_ranked_candidates:
                return scene_cluster_ranked_candidates
        if scenario_profile is None and context_text.strip():
            context_tokens = {
                token.casefold().strip()
                for token in jieba.lcut(context_text)
                if len(token.strip()) >= 2
            }
            matched_candidates: list[dict[str, Any]] = []
            for candidate in candidates:
                candidate_tokens = {
                    token.casefold().strip()
                    for token in jieba.lcut(
                        " ".join(
                            str(candidate.get(field) or "")
                            for field in ("action", "outcome", "scenario_summary")
                        )
                    )
                    if len(token.strip()) >= 2
                }
                shared_count = len(context_tokens & candidate_tokens)
                if shared_count <= 0:
                    continue
                ranked = dict(candidate)
                ranked["behavior_retrieval_score"] = round(
                    shared_count
                    / max(1, len(context_tokens))
                    * self._candidate_weight(candidate),
                    4,
                )
                matched_candidates.append(ranked)
            matched_candidates.sort(
                key=lambda candidate: (
                    float(candidate.get("behavior_retrieval_score") or 0.0),
                    int(candidate.get("success_count") or 0),
                    int(candidate.get("id") or 0),
                ),
                reverse=True,
            )
            return matched_candidates[:max_count]
        return []

    @staticmethod
    def _build_group_reference_text(
        *,
        behaviors: list[dict[str, Any]],
        scenario_profile: BehaviorScenarioProfile,
    ) -> str:
        reference_items: list[str] = []
        total_count = len(behaviors)
        for index, behavior in enumerate(behaviors, start=1):
            behavior_id = behavior.get("id")
            action = str(behavior.get("action") or "").strip()
            outcome = str(behavior.get("outcome") or "").strip()
            priority_label = BehaviorPatternSelector._format_priority_label(
                index, total_count
            )
            reference_items.append(
                f"{index}.\n"
                f"behavior_id：{behavior_id}\n"
                f"优先级：{priority_label}\n"
                f"行为：{action}\n"
                f"预期结果：{outcome}"
            )
        scenario_text = BehaviorPatternSelector._build_compact_scenario_text(
            scenario_profile
        )
        return (
            "以下是基于本轮 planner 已裁切上下文召回的行为表现参考，不是强制任务；"
            "只有在当前情境自然匹配时才采纳。\n"
            f"当前场景画像：\n{scenario_text}\n\n"
            "候选行为表现：\n"
            f"{chr(10).join(reference_items)}"
        )

    async def retrieve_for_planner(
        self,
        *,
        session_id: str,
        scenario_agent_runner: Callable[[str], Awaitable[str]] | None = None,
        context_text: str = "",
        include_context_in_prompt: bool = True,
        max_count: int = 3,
    ) -> BehaviorPatternRetrievalResult:
        if not session_id:
            return BehaviorPatternRetrievalResult()
        if not await self.has_candidates(session_id):
            return BehaviorPatternRetrievalResult()
        scenario_profile = await behavior_scenario_analyzer.analyze(
            context_text=context_text,
            sub_agent_runner=scenario_agent_runner,
            include_context_in_prompt=include_context_in_prompt,
        )
        candidates = await self._load_behavior_candidates(
            session_id,
            scenario_profile=scenario_profile,
            max_count=max(1, min(3, int(max_count))),
        )
        if not candidates:
            logger.debug("行为表现召回未命中候选: session_id=%s", session_id)
            return BehaviorPatternRetrievalResult(scenario_profile=scenario_profile)

        selected_behaviors: list[dict[str, Any]] = []
        references: list[BehaviorReferenceCandidate] = []
        for candidate in candidates[: max(1, min(3, int(max_count)))]:
            selected_behavior = candidate
            if selected_behavior:
                for score_key in (
                    "scene_cluster_score",
                    "profile_tag_match_score",
                    "behavior_retrieval_score",
                ):
                    if score_key in candidate:
                        selected_behavior[score_key] = candidate[score_key]
                selected_behaviors.append(selected_behavior)
                if behavior_id := int(selected_behavior.get("id") or 0):
                    references.append(
                        BehaviorReferenceCandidate(
                            behavior_id=behavior_id,
                            action=str(selected_behavior.get("action") or "").strip(),
                            outcome=str(selected_behavior.get("outcome") or "").strip(),
                            actor_type=str(
                                selected_behavior.get("actor_type") or ""
                            ).strip(),
                            learning_type=str(
                                selected_behavior.get("learning_type") or ""
                            ).strip(),
                            session_id=str(
                                selected_behavior.get("session_id") or ""
                            ).strip(),
                        )
                    )
        if not selected_behaviors:
            return BehaviorPatternRetrievalResult(scenario_profile=scenario_profile)

        reference_text = self._build_group_reference_text(
            behaviors=selected_behaviors,
            scenario_profile=scenario_profile,
        )
        logger.debug(
            "行为表现参考已召回: session_id=%s ids=%s",
            session_id,
            [behavior.get("id") for behavior in selected_behaviors],
        )
        return BehaviorPatternRetrievalResult(
            reference_text=reference_text,
            behaviors=selected_behaviors,
            scenario_profile=scenario_profile,
            references=references,
        )

    async def retrieve_fast_for_planner(
        self,
        *,
        session_id: str,
        context_text: str,
        max_count: int = 3,
    ) -> BehaviorPatternRetrievalResult:
        """Recall high-overlap habits without another model call on the hot path."""

        if not session_id or not context_text.strip():
            return BehaviorPatternRetrievalResult()
        scenario_profile = BehaviorScenarioProfile()
        candidates = await self._load_behavior_candidates(
            session_id,
            context_text=context_text,
            run_maintenance=False,
            max_count=max(1, min(3, int(max_count))),
        )
        if not candidates:
            return BehaviorPatternRetrievalResult(scenario_profile=scenario_profile)
        references = [
            BehaviorReferenceCandidate(
                behavior_id=int(candidate["id"]),
                action=str(candidate.get("action") or "").strip(),
                outcome=str(candidate.get("outcome") or "").strip(),
                actor_type=str(candidate.get("actor_type") or "").strip(),
                learning_type=str(candidate.get("learning_type") or "").strip(),
                session_id=str(candidate.get("session_id") or "").strip(),
            )
            for candidate in candidates
            if isinstance(candidate.get("id"), int)
        ]
        return BehaviorPatternRetrievalResult(
            reference_text=self._build_group_reference_text(
                behaviors=candidates, scenario_profile=scenario_profile
            ),
            behaviors=candidates,
            scenario_profile=scenario_profile,
            references=references,
        )


__all__ = [
    "ACTOR_GROUP_COLLECTIVE",
    "ACTOR_MAIBOT_SELF",
    "ACTOR_OTHER_USER",
    "BehaviorDatabase",
    "BehaviorExperiencePath",
    "BehaviorPatternSelector",
    "BehaviorReferenceCandidate",
    "BehaviorPatternRetrievalResult",
    "BehaviorScenarioProfile",
    "BehaviorScenarioSegment",
    "BehaviorScenarioTagCluster",
    "LEARNING_OBSERVED",
    "LEARNING_SELF_REFLECTION",
    "build_profile_tag_mapping",
]
