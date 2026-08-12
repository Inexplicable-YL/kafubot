# ruff: noqa: TC002, TC003, DTZ005, TRY400, TRY401, SIM103, PERF401
"""Behavior learner middleware entrypoint."""

import json
import re
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, cast
from typing_extensions import override

import anyio
from json_repair import repair_json
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain.agents.middleware.types import ExtendedModelResponse
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from agent.base import ManagerContext, ManagerState, UserMessage
from agent.middlewares.base import BaseDaemonMiddleware, SessionProcessOutput
from agent.prompts.manager import BOT_NAME
from agent.utils import content_to_text

from .constants import (
    ACTOR_GROUP_COLLECTIVE,
    ACTOR_MAIBOT_SELF,
    ACTOR_OTHER_USER,
    ACTOR_UNKNOWN,
    ALLOWED_FEEDBACK_STATUSES,
    ALLOWED_LEARNING_TYPES,
    BEHAVIOR_SELECTOR_CONTEXT_MESSAGE_LIMIT,
    BEHAVIOR_SELECTOR_CONTEXT_TEXT_LIMIT,
    DEFAULT_DB_URL,
    DEFAULT_MAX_CONCURRENT_LEARNERS,
    DEFAULT_MAX_FEEDBACK_ATTEMPTS,
    EVIDENCE_HISTORY_LIMIT,
    FEEDBACK_HISTORY_LIMIT,
    FEEDBACK_STATUS_FAILED,
    FEEDBACK_STATUS_NEUTRAL,
    FEEDBACK_STATUS_PARTIAL_SUCCESS,
    FEEDBACK_STATUS_SUCCESS,
    LEARNING_OBSERVED,
    LEARNING_SELF_REFLECTION,
    MIN_BEHAVIOR_SCORE,
    NEGATIVE_FEEDBACK_STATUSES,
    PARTIAL_POSITIVE_FEEDBACK_STATUSES,
    POSITIVE_FEEDBACK_STATUSES,
)
from .manager import (
    BehaviorDatabase,
    BehaviorExperiencePath,
    BehaviorPatternMaintenanceService,
    BehaviorPatternRetrievalResult,
    BehaviorPatternSelector,
    BehaviorReferenceCandidate,
    BehaviorScenarioProfile,
    BehaviorScenarioSegment,
    BehaviorSceneCluster,
    behavior_scenario_analyzer,
    build_profile_tag_distribution,
    load_tag_cluster_lookup,
    logger,
    upsert_behavior_graph_refs,
)
from .prompt import BEHAVIOR_FEEDBACK_PROMPT, BEHAVIOR_LEARN_PROMPT
from .utils import (
    clamp_score,
    clean_text,
    dump_json_list,
    load_json_list,
    normalize_source_ids,
    strip_json_code_fence,
)

_USER_TAG_PATTERN = re.compile(r"(?is)^<user-message\b[^>]*>(.*?)</user-message>$")
_BOT_TAG_PATTERN = re.compile(r"(?is)^<bot-message\b[^>]*>(.*?)</bot-message>$")
BEHAVIOR_SCENE_TEMPERATURE = 0.2
BEHAVIOR_LEARN_TEMPERATURE = 0.25
BEHAVIOR_FEEDBACK_TEMPERATURE = 0.15


@dataclass(frozen=True, slots=True)
class BehaviorMessageRecord:
    speaker: Literal["SELF", "USER"]
    content: str
    name: str = ""
    timestamp: str = ""
    occurred_at: datetime | None = None


@dataclass(frozen=True)
class BehaviorCandidate:
    action: str
    outcome: str
    source_ids: list[str]
    segment_id: str = ""
    actor_type: str = ACTOR_OTHER_USER
    learning_type: str = LEARNING_OBSERVED


@dataclass(frozen=True)
class BehaviorParseDiagnostics:
    normalized_response: str
    parsed_item_count: int = 0
    accepted_item_count: int = 0
    invalid_item_count: int = 0
    empty_output: bool = False
    missing_scene_start: bool = False
    parse_error: str = ""
    non_list_output: bool = False


@dataclass(frozen=True)
class BehaviorParseResult:
    candidates: list[BehaviorCandidate]
    diagnostics: BehaviorParseDiagnostics


@dataclass(frozen=True)
class BehaviorFilterResult:
    candidates: list[BehaviorCandidate]
    skipped_reasons: dict[str, int]


@dataclass(frozen=True)
class BehaviorFeedbackCandidate:
    behavior_id: int
    adopted: bool
    status: str
    score_delta: float
    reason: str
    outcome: str
    source_ids: list[str]


@dataclass(frozen=True)
class BehaviorLearningAcquireResult:
    acquired: bool
    reason: str = ""
    active_count: int = 0
    max_count: int = 0


@dataclass(frozen=True)
class BehaviorFeedbackContextItem:
    item_id: str
    item_type: str
    text: str
    speaker: str = ""
    source: str = ""


@dataclass(frozen=True)
class BehaviorFeedbackContext:
    references: list[BehaviorReferenceCandidate]
    timeline_items: list[BehaviorFeedbackContextItem]


@dataclass(frozen=True)
class PendingBehaviorAnalysisBatch:
    session_id: str
    messages: list[BaseMessage]


@dataclass(frozen=True)
class PendingBehaviorFeedbackState:
    references: list[BehaviorReferenceCandidate]
    created_at: datetime
    reference_text: str = ""
    attempts: int = 0

    @property
    def behavior_ids(self) -> set[int]:
        return {
            reference.behavior_id
            for reference in self.references
            if reference.behavior_id > 0
        }


class BehaviorLearningBatchGate:
    def __init__(self, max_count: int = DEFAULT_MAX_CONCURRENT_LEARNERS) -> None:
        self.max_count = max_count
        self._lock = anyio.Lock()
        self._active_session_ids: set[str] = set()

    async def acquire(self, session_id: str) -> BehaviorLearningAcquireResult:
        if self.max_count <= 0:
            return BehaviorLearningAcquireResult(
                False,
                "max_expression_learner <= 0",
                0,
                self.max_count,
            )

        async with self._lock:
            active_count = len(self._active_session_ids)
            if session_id in self._active_session_ids:
                return BehaviorLearningAcquireResult(
                    False,
                    "session_busy",
                    active_count,
                    self.max_count,
                )
            if active_count >= self.max_count:
                return BehaviorLearningAcquireResult(
                    False,
                    "global_limit",
                    active_count,
                    self.max_count,
                )
            self._active_session_ids.add(session_id)
            return BehaviorLearningAcquireResult(
                True,
                active_count=active_count + 1,
                max_count=self.max_count,
            )

    async def release(self, session_id: str) -> None:
        async with self._lock:
            self._active_session_ids.discard(session_id)


def _coerce_source_ids(raw_value: Any) -> list[str]:
    if isinstance(raw_value, list):
        raw_items = raw_value
    elif raw_value is None:
        raw_items = []
    else:
        raw_items = [raw_value]

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
    return source_ids


def _coerce_actor_type(raw_value: Any) -> str:
    normalized_value = str(raw_value or "").strip().lower()
    if normalized_value in {
        ACTOR_OTHER_USER,
        ACTOR_GROUP_COLLECTIVE,
        ACTOR_MAIBOT_SELF,
        "unknown",
    }:
        return normalized_value
    return ""


def _coerce_learning_type(raw_value: Any) -> str:
    normalized_value = str(raw_value or "").strip().lower()
    if normalized_value in {LEARNING_OBSERVED, LEARNING_SELF_REFLECTION}:
        return normalized_value
    return ""


def _coerce_bool(raw_value: Any) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, (int, float)):
        return raw_value != 0
    normalized_value = str(raw_value or "").strip().lower()
    return normalized_value in {"1", "true", "yes", "y", "adopted", "used"}


def _coerce_feedback_status(raw_value: Any) -> str:
    normalized_value = str(raw_value or "").strip().lower()
    if normalized_value in ALLOWED_FEEDBACK_STATUSES:
        return normalized_value
    if normalized_value in {"succeeded", "completed", "positive"}:
        return FEEDBACK_STATUS_SUCCESS
    if normalized_value in {"partial", "partially_successful", "weak_success"}:
        return FEEDBACK_STATUS_PARTIAL_SUCCESS
    if normalized_value in {"blocked", "abandoned", "negative", "failure"}:
        return FEEDBACK_STATUS_FAILED
    return FEEDBACK_STATUS_NEUTRAL


def _coerce_score_delta(raw_value: Any, *, status: str) -> float:
    try:
        score_delta = float(raw_value)
    except (TypeError, ValueError):
        if status == FEEDBACK_STATUS_SUCCESS:
            score_delta = 0.6
        elif status == FEEDBACK_STATUS_PARTIAL_SUCCESS:
            score_delta = 0.25
        elif status == FEEDBACK_STATUS_FAILED:
            score_delta = -0.6
        else:
            score_delta = 0.0
    if status == FEEDBACK_STATUS_SUCCESS:
        return max(0.1, min(1.0, abs(score_delta)))
    if status == FEEDBACK_STATUS_PARTIAL_SUCCESS:
        return max(0.05, min(0.35, abs(score_delta)))
    if status == FEEDBACK_STATUS_FAILED:
        return -max(0.1, min(1.0, abs(score_delta)))
    return 0.0


def _compact_log_text(text: str, *, max_length: int = 1200) -> str:
    compacted_text = " ".join((text or "").split()).strip()
    if len(compacted_text) <= max_length:
        return compacted_text
    return compacted_text[:max_length].rstrip() + "..."


def _normalize_text(text: str, *, max_length: int = 240) -> str:
    normalized = clean_text(text)
    if len(normalized) <= max_length:
        return normalized
    return normalized[:max_length].rstrip()


def _coerce_message_datetime(raw_value: Any) -> datetime | None:
    if isinstance(raw_value, datetime):
        return raw_value
    if not raw_value:
        return None
    try:
        return datetime.fromisoformat(str(raw_value).strip())
    except ValueError:
        return None


def _normalize_record_message(message: BaseMessage) -> BehaviorMessageRecord | None:
    if isinstance(message, HumanMessage) and isinstance(
        raw := message.additional_kwargs.get("raw"),
        UserMessage,
    ):
        content = clean_text(raw.message.get_msgcode())
        if not content:
            return None
        return BehaviorMessageRecord(
            speaker="USER",
            content=content,
            name=clean_text(raw.user),
            timestamp=raw.timestamp.strftime("%H:%M:%S"),
            occurred_at=raw.timestamp,
        )

    raw_content = content_to_text(message.content)
    if not raw_content.strip():
        return None

    if isinstance(message, HumanMessage):
        stripped = raw_content.strip()
        if match := _USER_TAG_PATTERN.match(stripped):
            content = clean_text(match.group(1))
            if content:
                return BehaviorMessageRecord(
                    speaker="USER",
                    content=content,
                    name="未知用户",
                    timestamp="unknown",
                    occurred_at=_coerce_message_datetime(
                        message.additional_kwargs.get("history_timestamp")
                    ),
                )
        if match := _BOT_TAG_PATTERN.match(stripped):
            content = clean_text(match.group(1))
            if content:
                return BehaviorMessageRecord(
                    speaker="SELF",
                    content=content,
                    name=BOT_NAME,
                    timestamp="unknown",
                    occurred_at=_coerce_message_datetime(
                        message.additional_kwargs.get("history_timestamp")
                    ),
                )
        return None

    if isinstance(message, AIMessage):
        content = clean_text(raw_content)
        if content:
            occurred_at = _coerce_message_datetime(
                message.additional_kwargs.get("history_timestamp")
            )
            return BehaviorMessageRecord(
                speaker="SELF",
                content=content,
                name=BOT_NAME,
                timestamp=(
                    occurred_at.strftime("%H:%M:%S")
                    if occurred_at is not None
                    else "unknown"
                ),
                occurred_at=occurred_at,
            )
    return None


def _parse_behavior_item(raw_item: Any) -> BehaviorCandidate | None:
    if not isinstance(raw_item, dict):
        return None
    action = str(raw_item.get("action") or "").strip()
    outcome = str(raw_item.get("outcome") or "").strip()
    source_ids = _coerce_source_ids(raw_item.get("source_ids"))
    segment_id = str(
        raw_item.get("segment_id") or raw_item.get("scene_id") or ""
    ).strip()
    actor_type = _coerce_actor_type(raw_item.get("actor_type"))
    learning_type = _coerce_learning_type(raw_item.get("learning_type"))
    if not action or not outcome or not actor_type or not learning_type:
        return None
    if actor_type == ACTOR_MAIBOT_SELF and learning_type != LEARNING_SELF_REFLECTION:
        return None
    if actor_type != ACTOR_MAIBOT_SELF and learning_type != LEARNING_OBSERVED:
        return None
    return BehaviorCandidate(
        action=action,
        outcome=outcome,
        source_ids=source_ids,
        segment_id=segment_id,
        actor_type=actor_type,
        learning_type=learning_type,
    )


def parse_behavior_response_with_diagnostics(
    response: str,
    *,
    scene_start: str,
) -> BehaviorParseResult:
    normalized_response = strip_json_code_fence(response or "")
    normalized_scene_start = scene_start.strip()
    if not normalized_response:
        return BehaviorParseResult(
            candidates=[],
            diagnostics=BehaviorParseDiagnostics(
                normalized_response="",
                empty_output=True,
            ),
        )
    if not normalized_scene_start:
        return BehaviorParseResult(
            candidates=[],
            diagnostics=BehaviorParseDiagnostics(
                normalized_response=normalized_response,
                missing_scene_start=True,
            ),
        )

    try:
        parsed_response = json.loads(repair_json(normalized_response))
    except Exception as exc:  # noqa: BLE001
        return BehaviorParseResult(
            candidates=[],
            diagnostics=BehaviorParseDiagnostics(
                normalized_response=normalized_response,
                parse_error=str(exc),
            ),
        )

    if not isinstance(parsed_response, list):
        return BehaviorParseResult(
            candidates=[],
            diagnostics=BehaviorParseDiagnostics(
                normalized_response=normalized_response,
                non_list_output=True,
            ),
        )

    candidates: list[BehaviorCandidate] = []
    invalid_item_count = 0
    for raw_item in parsed_response:
        candidate = _parse_behavior_item(raw_item)
        if candidate is not None:
            candidates.append(candidate)
        else:
            invalid_item_count += 1

    return BehaviorParseResult(
        candidates=candidates,
        diagnostics=BehaviorParseDiagnostics(
            normalized_response=normalized_response,
            parsed_item_count=len(parsed_response),
            accepted_item_count=len(candidates),
            invalid_item_count=invalid_item_count,
        ),
    )


def parse_behavior_feedback_response(response: str) -> list[BehaviorFeedbackCandidate]:
    normalized_response = strip_json_code_fence(response or "")
    if not normalized_response:
        return []

    try:
        parsed_response = json.loads(repair_json(normalized_response))
    except Exception:  # noqa: BLE001
        logger.warning("行为路径反馈结果解析失败: %r", normalized_response)
        return []

    if isinstance(parsed_response, dict):
        raw_items = (
            parsed_response.get("feedback") or parsed_response.get("items") or []
        )
    else:
        raw_items = parsed_response
    if isinstance(raw_items, dict):
        raw_items = [raw_items]
    if not isinstance(raw_items, list):
        return []

    feedback_items: list[BehaviorFeedbackCandidate] = []
    used_behavior_ids: set[int] = set()
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            continue
        try:
            behavior_id = int(raw_item.get("behavior_id") or raw_item.get("id") or 0)
        except (TypeError, ValueError):
            behavior_id = 0
        if behavior_id <= 0 or behavior_id in used_behavior_ids:
            continue
        adopted = _coerce_bool(raw_item.get("adopted"))
        status = _coerce_feedback_status(raw_item.get("status"))
        score_delta = _coerce_score_delta(raw_item.get("score_delta"), status=status)
        reason = str(raw_item.get("reason") or "").strip()
        outcome = str(raw_item.get("outcome") or "").strip()
        source_ids = _coerce_source_ids(raw_item.get("source_ids"))
        if (
            not adopted
            or status == FEEDBACK_STATUS_NEUTRAL
            or abs(score_delta) <= 0.0001
            or not reason
            or not source_ids
        ):
            continue
        used_behavior_ids.add(behavior_id)
        feedback_items.append(
            BehaviorFeedbackCandidate(
                behavior_id=behavior_id,
                adopted=adopted,
                status=status,
                score_delta=score_delta,
                reason=reason,
                outcome=outcome,
                source_ids=source_ids,
            )
        )
    return feedback_items


def _validate_behavior_feedback_evidence(
    feedback_item: BehaviorFeedbackCandidate,
    feedback_context: BehaviorFeedbackContext,
) -> tuple[bool, str, list[str]]:
    item_by_id = {
        item.item_id: item
        for item in feedback_context.timeline_items
        if item.item_type == "chat_message" and item.item_id
    }
    valid_source_ids: list[str] = []
    cited_items: list[BehaviorFeedbackContextItem] = []
    for source_id in feedback_item.source_ids:
        item = item_by_id.get(source_id)
        if item is None:
            continue
        valid_source_ids.append(source_id)
        cited_items.append(item)
    if not valid_source_ids:
        return False, "invalid_source_ids", []
    has_self_adoption_evidence = any(item.speaker == "SELF" for item in cited_items)
    if not has_self_adoption_evidence:
        return False, "missing_self_adoption_evidence", valid_source_ids
    return True, "", valid_source_ids


def _build_evidence_item(
    *,
    action: str,
    outcome: str,
    source_ids: Sequence[str],
    actor_type: str,
    learning_type: str,
    profile_tag_distribution: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    evidence_item: dict[str, Any] = {
        "action": action,
        "outcome": outcome,
        "source_ids": normalize_source_ids(source_ids),
        "actor_type": actor_type,
        "learning_type": learning_type,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    if profile_tag_distribution:
        evidence_item["profile_tag_distribution"] = list(profile_tag_distribution)
    return evidence_item


async def upsert_behavior_pattern(
    db: BehaviorDatabase,
    *,
    action: str,
    outcome: str,
    source_ids: Sequence[str],
    session_id: str,
    scenario_profile: BehaviorScenarioProfile,
    scene_start: str,
    actor_type: str = ACTOR_OTHER_USER,
    learning_type: str = LEARNING_OBSERVED,
) -> BehaviorExperiencePath | None:
    normalized_action = _normalize_text(action, max_length=240)
    normalized_outcome = _normalize_text(outcome, max_length=240)
    normalized_source_ids = normalize_source_ids(source_ids)
    normalized_actor_type = _coerce_actor_type(actor_type) or ACTOR_UNKNOWN
    normalized_learning_type = (
        str(learning_type or "").strip().lower()
        if str(learning_type or "").strip().lower() in ALLOWED_LEARNING_TYPES
        else (
            LEARNING_SELF_REFLECTION
            if normalized_actor_type == ACTOR_MAIBOT_SELF
            else LEARNING_OBSERVED
        )
    )
    if not normalized_action or not normalized_outcome:
        logger.warning(
            "跳过写入行为经验路径：归一化后字段为空 action=%r outcome=%r",
            normalized_action,
            normalized_outcome,
        )
        return None

    try:
        async with db.session() as session:
            graph_refs = await upsert_behavior_graph_refs(
                session=session,
                session_id=session_id,
                profile=scenario_profile,
                scene_start=scene_start,
                action=normalized_action,
                outcome=normalized_outcome,
            )
            if graph_refs is None:
                logger.warning(
                    "跳过写入行为经验路径：场景簇引用生成失败 session_id=%s action=%s outcome=%s",
                    session_id,
                    normalized_action,
                    normalized_outcome,
                )
                return None

            profile_tag_distribution = build_profile_tag_distribution(
                scenario_profile,
                tag_lookup=await load_tag_cluster_lookup(session),
            )
            now = datetime.now()
            evidence_item = _build_evidence_item(
                action=normalized_action,
                outcome=normalized_outcome,
                source_ids=normalized_source_ids,
                actor_type=normalized_actor_type,
                learning_type=normalized_learning_type,
                profile_tag_distribution=profile_tag_distribution,
            )

            statement = (
                select(BehaviorExperiencePath)
                .where(BehaviorExperiencePath.session_id == session_id)
                .where(
                    BehaviorExperiencePath.scene_cluster_id
                    == graph_refs.scene_cluster_id
                )
                .where(BehaviorExperiencePath.action_id == graph_refs.action_id)
                .where(BehaviorExperiencePath.outcome_id == graph_refs.outcome_id)
                .where(BehaviorExperiencePath.actor_type == normalized_actor_type)
                .where(BehaviorExperiencePath.learning_type == normalized_learning_type)
            )
            path = (await session.scalars(statement)).first()
            if path is None:
                path = BehaviorExperiencePath(
                    session_id=session_id,
                    scene_cluster_id=graph_refs.scene_cluster_id,
                    action_id=graph_refs.action_id,
                    outcome_id=graph_refs.outcome_id,
                    actor_type=normalized_actor_type,
                    learning_type=normalized_learning_type,
                    evidence_list=dump_json_list([evidence_item]),
                    feedback_list=dump_json_list([]),
                    count=1,
                    activation_count=0,
                    success_count=0,
                    failure_count=0,
                    score=0.0,
                    enabled=True,
                    last_active_time=now,
                    create_time=now,
                    update_time=now,
                )
            else:
                evidence_items = load_json_list(path.evidence_list)
                evidence_items.append(evidence_item)
                path.evidence_list = dump_json_list(
                    evidence_items[-EVIDENCE_HISTORY_LIMIT:]
                )
                path.count += 1
                path.last_active_time = now
                path.update_time = now

            session.add(path)
            await session.flush()
            await session.refresh(path)
            session.expunge(path)
            return path
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "写入行为经验路径失败: session_id=%s action=%s outcome=%s source_ids=%s error=%s",
            session_id,
            normalized_action,
            normalized_outcome,
            normalized_source_ids,
            exc,
        )
        return None


async def apply_behavior_scene_feedback(
    db: BehaviorDatabase,
    *,
    experience_path_id: int,
    score_delta: float,
    status: str,
) -> None:
    del score_delta
    del status
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
            "更新行为场景簇反馈失败: experience_id=%s error=%s", experience_path_id, exc
        )


async def apply_behavior_feedback(
    db: BehaviorDatabase,
    *,
    pattern_id: int,
    score_delta: float,
    status: str,
    reason: str,
    outcome: str,
    session_id: str,
    source_ids: Sequence[str] = (),
) -> BehaviorExperiencePath | None:
    normalized_status = str(status or "").strip().lower()
    normalized_reason = _normalize_text(reason, max_length=300)
    normalized_outcome = _normalize_text(outcome, max_length=240)
    normalized_source_ids = normalize_source_ids(source_ids)
    now = datetime.now()
    try:
        async with db.session() as session:
            path = await session.get(BehaviorExperiencePath, pattern_id)
            if path is None:
                return None
            feedback_items = load_json_list(path.feedback_list)
            feedback_items.append(
                {
                    "score_delta": float(score_delta),
                    "status": normalized_status,
                    "reason": normalized_reason,
                    "outcome": normalized_outcome,
                    "session_id": session_id,
                    "source_ids": normalized_source_ids,
                    "created_at": now.isoformat(timespec="seconds"),
                }
            )
            path.feedback_list = dump_json_list(
                feedback_items[-FEEDBACK_HISTORY_LIMIT:]
            )
            path.score = clamp_score(float(path.score or 0.0) + float(score_delta))
            path.last_feedback_time = now
            path.update_time = now
            if normalized_status in POSITIVE_FEEDBACK_STATUSES:
                path.success_count += 1
            elif normalized_status in PARTIAL_POSITIVE_FEEDBACK_STATUSES:
                pass
            elif normalized_status in NEGATIVE_FEEDBACK_STATUSES:
                path.failure_count += 1
            if path.score <= MIN_BEHAVIOR_SCORE and path.failure_count >= 3:
                path.enabled = False

            session.add(path)
            await session.flush()
            await session.refresh(path)
            session.expunge(path)
            feedback_path = path
    except Exception as exc:  # noqa: BLE001
        logger.error("写入行为经验路径反馈失败: id=%s error=%s", pattern_id, exc)
        return None
    await apply_behavior_scene_feedback(
        db,
        experience_path_id=pattern_id,
        score_delta=score_delta,
        status=normalized_status,
    )
    return feedback_path


class BehaviorLearnerMiddleware(BaseDaemonMiddleware[PendingBehaviorAnalysisBatch]):
    state_schema = ManagerState

    def __init__(
        self,
        analyze_model: BaseChatModel,
        *,
        scene_model: BaseChatModel | None = None,
        feedback_model: BaseChatModel | None = None,
        db_url: str = DEFAULT_DB_URL,
        engine: AsyncEngine | None = None,
        max_sessions: int = 100,
        max_batches: int = 3,
        max_retries: int = 3,
        min_messages_for_extraction: int = 10,
        max_concurrent_learners: int = DEFAULT_MAX_CONCURRENT_LEARNERS,
        max_feedback_attempts: int = DEFAULT_MAX_FEEDBACK_ATTEMPTS,
        behavior_group_resolver: Callable[[str], set[str] | tuple[set[str], bool]]
        | None = None,
    ) -> None:
        super().__init__(
            max_sessions=max_sessions,
            max_retries=max_retries,
            max_batch_window_size=max_batches,
            logger_config={
                "worker_name": "BehaviorLearner",
                "job_name": "behavior learning",
                "process_failure_log": "Failed to analyze behavior entries",
                "retry_exhausted_label": "Behavior learning",
            },
        )
        self.analyze_model = analyze_model
        self.scene_model = scene_model or analyze_model
        self.feedback_model = feedback_model or analyze_model
        self.min_messages_for_extraction = min_messages_for_extraction
        self.max_feedback_attempts = max_feedback_attempts
        self._db = BehaviorDatabase(engine=engine, db_url=db_url)
        self._maintenance = BehaviorPatternMaintenanceService(self._db)
        self._selector = BehaviorPatternSelector(
            self._db,
            self._maintenance,
            group_resolver=behavior_group_resolver,
        )
        self._learning_gate = BehaviorLearningBatchGate(max_concurrent_learners)
        self._turn_selection_cache: dict[
            str, BehaviorPatternRetrievalResult | None
        ] = {}
        self._pending_feedback: dict[str, list[PendingBehaviorFeedbackState]] = {}

    @override
    async def abefore_agent(
        self,
        state: ManagerState,
        runtime: Runtime[ManagerContext],
    ) -> dict[str, Any] | None:
        del state
        self._turn_selection_cache.pop(runtime.context["session_id"], None)
        return None

    @override
    async def aafter_agent(
        self,
        state: ManagerState,
        runtime: Runtime[ManagerContext],
    ) -> dict[str, Any] | None:
        if messages := cast(
            "list[BaseMessage] | None", state.get("summary_pruned_messages")
        ):
            session_id = runtime.context["session_id"]
            await self._enqueue_batch(
                session_id,
                PendingBehaviorAnalysisBatch(
                    session_id=session_id,
                    messages=list(messages),
                ),
            )
        self._turn_selection_cache.pop(runtime.context["session_id"], None)
        return None

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest[ManagerContext],
        handler: Callable[
            [ModelRequest[ManagerContext]], Awaitable[ModelResponse[Any]]
        ],
    ) -> ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]:
        session_id = request.runtime.context["session_id"]
        selection = self._turn_selection_cache.get(session_id)
        if selection is None:
            visible_messages = self._extract_visible_behavior_messages(request.messages)
            context_text = self._build_behavior_selector_context_text(visible_messages)
            if context_text:
                selection = await self._selector.retrieve_for_planner(
                    session_id=session_id,
                    scenario_agent_runner=lambda system_prompt: self._run_selection_scene_prompt(
                        visible_messages,
                        system_prompt,
                    ),
                    context_text=context_text,
                    include_context_in_prompt=False,
                )
                if selection.references:
                    self._enqueue_pending_feedback(
                        session_id,
                        selection.references,
                        selection.reference_text,
                    )
            else:
                selection = BehaviorPatternRetrievalResult()
            self._turn_selection_cache[session_id] = selection

        if selection and selection.reference_text:
            request = request.override(
                messages=[
                    HumanMessage(
                        content=selection.reference_text,
                        additional_kwargs={"lc_source": "behavior_pattern"},
                    ),
                    *request.messages,
                ]
            )
        return await handler(request)

    @override
    async def process_batches(
        self,
        session_id: str,
        batches: tuple[PendingBehaviorAnalysisBatch, ...],
    ) -> SessionProcessOutput:
        records = self._extract_pruned_messages(batches)
        if not records:
            return len(batches)

        task_errors: list[Exception] = []
        written_behavior_ids: set[int] = set()
        evaluated_behavior_ids: set[int] = set()

        async def run_learning() -> None:
            try:
                await self._learn_from_session_messages(
                    records,
                    learning_session_id=session_id,
                )
            except Exception as exc:
                task_errors.append(exc)

        run_learning_task = len(records) >= self.min_messages_for_extraction
        if not run_learning_task:
            logger.debug(
                "%s 行为学习消息不足: 可学习=%s 阈值=%s",
                session_id,
                len(records),
                self.min_messages_for_extraction,
            )

        pending_states = list(self._pending_feedback.get(session_id) or [])
        run_feedback_task = bool(pending_states)

        async def run_feedback() -> None:
            nonlocal evaluated_behavior_ids, written_behavior_ids
            try:
                for pending_state in pending_states:
                    feedback_context = self._build_behavior_feedback_context(
                        pending_state=pending_state,
                        records=records,
                    )
                    if feedback_context is None:
                        continue
                    evaluated_behavior_ids.update(pending_state.behavior_ids)
                    written_behavior_ids.update(
                        await self._evaluate_behavior_feedback(
                            session_id,
                            feedback_context,
                        )
                    )
            except Exception as exc:
                task_errors.append(exc)

        if not run_learning_task and not run_feedback_task:
            return len(batches)

        async with anyio.create_task_group() as task_group:
            if run_learning_task:
                task_group.start_soon(run_learning)
            if run_feedback_task:
                task_group.start_soon(run_feedback)

        if run_feedback_task:
            self._reconcile_pending_feedback(
                session_id,
                evaluated_behavior_ids=evaluated_behavior_ids,
                written_behavior_ids=written_behavior_ids,
            )

        for task_error in task_errors:
            logger.exception("行为中间件任务执行失败", exc_info=task_error)
        return len(batches)

    @override
    async def on_close(self) -> None:
        self._turn_selection_cache.clear()
        self._pending_feedback.clear()
        await self._db.dispose()

    def _extract_visible_behavior_messages(
        self,
        messages: Sequence[BaseMessage],
    ) -> list[BehaviorMessageRecord]:
        return [
            record
            for message in messages
            if (record := _normalize_record_message(message)) is not None
        ]

    def _extract_pruned_messages(
        self,
        batches: Sequence[PendingBehaviorAnalysisBatch],
    ) -> list[BehaviorMessageRecord]:
        records: list[BehaviorMessageRecord] = []
        for batch in batches:
            for message in batch.messages:
                if record := _normalize_record_message(message):
                    records.append(record)
        return records

    @staticmethod
    def _append_behavior_selector_context_item(
        context_items: list[str],
        *,
        text: str,
        seen_texts: set[str],
    ) -> None:
        normalized_text = clean_text(text)
        if not normalized_text or normalized_text in seen_texts:
            return
        seen_texts.add(normalized_text)
        context_items.append(normalized_text)

    def _build_behavior_selector_context_text(
        self,
        records: Sequence[BehaviorMessageRecord],
    ) -> str:
        context_items: list[str] = []
        seen_texts: set[str] = set()
        for record in records[-BEHAVIOR_SELECTOR_CONTEXT_MESSAGE_LIMIT:]:
            self._append_behavior_selector_context_item(
                context_items,
                text=record.content,
                seen_texts=seen_texts,
            )
        context_text = "\n".join(
            context_items[-BEHAVIOR_SELECTOR_CONTEXT_MESSAGE_LIMIT:]
        )
        if len(context_text) <= BEHAVIOR_SELECTOR_CONTEXT_TEXT_LIMIT:
            return context_text
        return context_text[-BEHAVIOR_SELECTOR_CONTEXT_TEXT_LIMIT:]

    async def _run_selection_scene_prompt(
        self,
        records: Sequence[BehaviorMessageRecord],
        system_prompt: str,
    ) -> str:
        messages = self._build_scene_analysis_messages(records, system_prompt)
        response = await cast(
            "Any",
            self.scene_model.bind(temperature=BEHAVIOR_SCENE_TEMPERATURE),
        ).ainvoke(
            messages,
            config={"metadata": {"lc_source": "behavior_scene_analyzer"}},
        )
        return content_to_text(response.content)

    @staticmethod
    def _build_scene_analysis_messages(
        records: Sequence[BehaviorMessageRecord],
        system_prompt: str,
    ) -> list[BaseMessage]:
        scene_messages: list[BaseMessage] = [
            SystemMessage(
                content=(
                    f"{system_prompt}\n\n"
                    "注意：聊天记录会在后续多条 user message 中给出。每条消息内的 source_id "
                    "是本轮场景概括的来源编号；speaker=SELF 表示这条真实聊天消息由麦麦发出。"
                )
            )
        ]
        for index, record in enumerate(records, start=1):
            content = record.content or "[空消息]"
            scene_messages.append(
                HumanMessage(
                    content="\n".join(
                        [
                            f"[source_id:{index}]",
                            f"[speaker:{record.speaker}]",
                            f"[name:{record.name or ('未知用户' if record.speaker == 'USER' else BOT_NAME)}]",
                            f"[time:{record.timestamp or 'unknown'}]",
                            "[content]",
                            content,
                        ]
                    )
                )
            )
        scene_messages.append(
            HumanMessage(content="请根据以上真实聊天消息输出场景片段 JSON。")
        )
        return scene_messages

    @staticmethod
    def _build_learning_messages(
        records: Sequence[BehaviorMessageRecord],
        system_prompt: str,
    ) -> list[BaseMessage]:
        learning_messages: list[BaseMessage] = [
            SystemMessage(
                content=(
                    f"{system_prompt}\n\n"
                    "注意：聊天记录会在后续多条 user message 中给出。每条消息内的 source_id "
                    "是本轮学习的来源编号；speaker=SELF 的消息可以作为行为链的一部分，"
                    "如果行为主体是 speaker=SELF，请用 actor_type=maibot_self 与 "
                    "learning_type=self_reflection 表示；但 action/outcome 不要直接写 SELF 或具体昵称。"
                )
            )
        ]
        for index, record in enumerate(records, start=1):
            content = record.content or "[空消息]"
            learning_messages.append(
                HumanMessage(
                    content="\n".join(
                        [
                            f"[source_id:{index}]",
                            f"[speaker:{record.speaker}]",
                            f"[name:{record.name or ('未知用户' if record.speaker == 'USER' else BOT_NAME)}]",
                            f"[time:{record.timestamp or 'unknown'}]",
                            "[content]",
                            content,
                        ]
                    )
                )
            )
        learning_messages.append(HumanMessage(content="请根据以上聊天消息输出 JSON。"))
        return learning_messages

    @staticmethod
    def _build_learning_context_text(records: Sequence[BehaviorMessageRecord]) -> str:
        context_lines: list[str] = []
        for index, record in enumerate(records, start=1):
            content = record.content or "[空消息]"
            if len(content) > 300:
                content = content[:300].rstrip() + "..."
            context_lines.append(
                "\n".join(
                    [
                        f"[source_id:{index}]",
                        f"[speaker:{record.speaker}]",
                        f"[time:{record.timestamp or 'unknown'}]",
                        "[content]",
                        content,
                    ]
                )
            )
        return "\n\n".join(context_lines).strip()

    def _build_behavior_feedback_context(
        self,
        *,
        pending_state: PendingBehaviorFeedbackState,
        records: Sequence[BehaviorMessageRecord],
    ) -> BehaviorFeedbackContext | None:
        if not pending_state.references:
            return None
        filtered_records = [
            record
            for record in records
            if record.occurred_at is not None
            and record.occurred_at > pending_state.created_at
        ]
        if not filtered_records:
            return None
        timeline_items: list[BehaviorFeedbackContextItem] = []
        timeline_items.append(
            BehaviorFeedbackContextItem(
                item_id="ref1",
                item_type="behavior_reference",
                text=(
                    pending_state.reference_text.strip()
                    or "以上行为参考已在此时间点前提供给模型；以下均为其后的真实聊天。"
                ),
                source="behavior_pattern",
            )
        )
        for index, record in enumerate(filtered_records, start=1):
            timeline_items.append(
                BehaviorFeedbackContextItem(
                    item_id=f"m{index}",
                    item_type="chat_message",
                    text=record.content or "[空消息]",
                    speaker=record.speaker,
                    source="trimmed_history",
                )
            )
        if not timeline_items:
            return None
        return BehaviorFeedbackContext(
            references=list(pending_state.references),
            timeline_items=timeline_items,
        )

    def _enqueue_pending_feedback(
        self,
        session_id: str,
        references: Sequence[BehaviorReferenceCandidate],
        reference_text: str,
    ) -> None:
        queue = self._pending_feedback.setdefault(session_id, [])
        queued_behavior_ids = {
            behavior_id for state in queue for behavior_id in state.behavior_ids
        }
        new_references: list[BehaviorReferenceCandidate] = []
        for reference in references:
            if (
                reference.behavior_id <= 0
                or reference.behavior_id in queued_behavior_ids
            ):
                continue
            queued_behavior_ids.add(reference.behavior_id)
            new_references.append(reference)
        if not new_references:
            return
        queue.append(
            PendingBehaviorFeedbackState(
                references=new_references,
                created_at=datetime.now(),
                reference_text=reference_text,
            )
        )

    def _reconcile_pending_feedback(
        self,
        session_id: str,
        *,
        evaluated_behavior_ids: set[int],
        written_behavior_ids: set[int],
    ) -> None:
        states = self._pending_feedback.get(session_id)
        if not states:
            return
        next_states: list[PendingBehaviorFeedbackState] = []
        for state in states:
            remaining_references = [
                reference
                for reference in state.references
                if reference.behavior_id not in written_behavior_ids
            ]
            if not remaining_references:
                continue
            if not (state.behavior_ids & evaluated_behavior_ids):
                next_states.append(
                    PendingBehaviorFeedbackState(
                        references=remaining_references,
                        created_at=state.created_at,
                        reference_text=state.reference_text,
                        attempts=state.attempts,
                    )
                )
                continue
            next_attempts = state.attempts + 1
            if next_attempts >= self.max_feedback_attempts:
                continue
            next_states.append(
                PendingBehaviorFeedbackState(
                    references=remaining_references,
                    created_at=state.created_at,
                    reference_text=state.reference_text,
                    attempts=next_attempts,
                )
            )
        if next_states:
            self._pending_feedback[session_id] = next_states
        else:
            self._pending_feedback.pop(session_id, None)

    async def _learn_from_session_messages(
        self,
        records: list[BehaviorMessageRecord],
        *,
        learning_session_id: str,
    ) -> bool:
        acquire_result = await self._learning_gate.acquire(learning_session_id)
        if not acquire_result.acquired:
            if acquire_result.reason == "session_busy":
                logger.info(
                    "%s 已有行为学习批次正在运行，放弃新的批次", learning_session_id
                )
            elif acquire_result.reason == "global_limit":
                logger.info(
                    "行为学习全局并发已满，放弃新的批次: active=%s, max=%s, session_id=%s",
                    acquire_result.active_count,
                    acquire_result.max_count,
                    learning_session_id,
                )
            else:
                logger.warning(
                    "行为学习并发配置无效，放弃新的批次: max_expression_learner=%s, session_id=%s",
                    acquire_result.max_count,
                    learning_session_id,
                )
            return False
        try:
            return await self._run_learning_batch(
                records, learning_session_id=learning_session_id
            )
        finally:
            await self._learning_gate.release(learning_session_id)

    async def _run_learning_batch(
        self,
        records: list[BehaviorMessageRecord],
        *,
        learning_session_id: str,
    ) -> bool:
        scene_segments = await self._analyze_learning_scene_segments(
            records,
            learning_session_id=learning_session_id,
        )
        if not scene_segments:
            logger.debug(
                "%s 行为学习未形成可用场景片段，跳过本批次", learning_session_id
            )
            return False
        scene_start_by_segment_id = {
            segment.segment_id: segment.profile.tag_cluster_text()
            for segment in scene_segments
            if segment.profile.tag_cluster_text()
        }
        primary_segment = scene_segments[0]
        scene_start = scene_start_by_segment_id.get(primary_segment.segment_id, "")
        if not scene_start:
            logger.debug(
                "%s 行为学习未形成可用 tag 场景，跳过本批次", learning_session_id
            )
            return False

        prompt = BEHAVIOR_LEARN_PROMPT.format(
            bot_name=BOT_NAME,
            chat_str="聊天记录将在后续多条 user message 中给出；请以每条消息中的 source_id 作为来源行编号。",
            scene_profile=self._format_scene_segments_for_prompt(scene_segments),
        )
        try:
            learning_messages = self._build_learning_messages(records, prompt)
            response = await cast(
                "Any",
                self.analyze_model.bind(temperature=BEHAVIOR_LEARN_TEMPERATURE),
            ).ainvoke(
                learning_messages,
                config={"metadata": {"lc_source": "learn_behavior"}},
            )
            response_text = content_to_text(response.content)
        except Exception as exc:  # noqa: BLE001
            logger.error("学习行为表现失败: %s", exc)
            return False

        parse_result = parse_behavior_response_with_diagnostics(
            response_text,
            scene_start=scene_start,
        )
        self._log_parse_diagnostics(
            learning_session_id=learning_session_id,
            response=response_text,
            parse_result=parse_result,
        )
        filter_result = self._filter_behavior_candidates(
            parse_result.candidates, records
        )
        behavior_candidates = filter_result.candidates
        logger.info(
            "%s 行为学习过滤概览: 解析候选=%s 有效候选=%s 跳过原因=%s",
            learning_session_id,
            len(parse_result.candidates),
            len(behavior_candidates),
            filter_result.skipped_reasons,
        )
        if not behavior_candidates:
            logger.info(
                "%s 行为学习未抽取到有效候选: 模型输出预览=%r",
                learning_session_id,
                _compact_log_text(
                    parse_result.diagnostics.normalized_response, max_length=1600
                ),
            )
            return False

        wrote_pattern = False
        write_success_count = 0
        write_failed_count = 0
        for candidate in behavior_candidates[:12]:
            matched_segment = self._select_segment_for_candidate(
                candidate, scene_segments
            )
            candidate_scene_start = scene_start_by_segment_id.get(
                matched_segment.segment_id,
                scene_start,
            )
            logger.info(
                "%s 准备写入行为经验路径: segment_id=%s action=%s outcome=%s actor_type=%s learning_type=%s source_ids=%s",
                learning_session_id,
                matched_segment.segment_id,
                candidate.action,
                candidate.outcome,
                candidate.actor_type,
                candidate.learning_type,
                candidate.source_ids,
            )
            path = await upsert_behavior_pattern(
                self._db,
                action=candidate.action,
                outcome=candidate.outcome,
                source_ids=candidate.source_ids,
                session_id=learning_session_id,
                scenario_profile=matched_segment.profile,
                scene_start=candidate_scene_start,
                actor_type=candidate.actor_type,
                learning_type=candidate.learning_type,
            )
            if path is None:
                write_failed_count += 1
                logger.warning(
                    "%s 行为经验路径写入未成功: segment_id=%s action=%s outcome=%s actor_type=%s learning_type=%s source_ids=%s",
                    learning_session_id,
                    matched_segment.segment_id,
                    candidate.action,
                    candidate.outcome,
                    candidate.actor_type,
                    candidate.learning_type,
                    candidate.source_ids,
                )
                continue
            wrote_pattern = True
            write_success_count += 1
            logger.info(
                "学习到行为经验路径 [ID: %s]: 场景片段=%s 场景=%s 主体=%s 类型=%s 行为=%s 结果=%s",
                path.id,
                matched_segment.segment_id,
                candidate_scene_start,
                candidate.actor_type,
                candidate.learning_type,
                candidate.action,
                candidate.outcome,
            )

        logger.info(
            "%s 行为学习写入概览: 有效候选=%s 尝试写入=%s 成功=%s 失败=%s",
            learning_session_id,
            len(behavior_candidates),
            min(len(behavior_candidates), 12),
            write_success_count,
            write_failed_count,
        )

        if wrote_pattern:
            maintenance_result = await self._maintenance.maybe_maintain_session(
                session_id=learning_session_id,
                force=True,
            )
            if maintenance_result.changed:
                logger.info(
                    "%s 行为表现已完成学习后维护: 衰减=%s 禁用=%s 合并=%s",
                    learning_session_id,
                    maintenance_result.decayed_count,
                    maintenance_result.disabled_count,
                    maintenance_result.merged_count,
                )
        return wrote_pattern

    def _log_parse_diagnostics(
        self,
        *,
        learning_session_id: str,
        response: str,
        parse_result: BehaviorParseResult,
    ) -> None:
        diagnostics = parse_result.diagnostics
        response_preview = _compact_log_text(
            diagnostics.normalized_response or response,
            max_length=1600,
        )
        logger.info(
            "%s 行为学习解析概览: 原始长度=%s 规范化长度=%s 数组项=%s 解析候选=%s 无效项=%s 空输出=%s 缺少场景=%s 非数组=%s 解析错误=%s 输出预览=%r",
            learning_session_id,
            len(response or ""),
            len(diagnostics.normalized_response),
            diagnostics.parsed_item_count,
            diagnostics.accepted_item_count,
            diagnostics.invalid_item_count,
            diagnostics.empty_output,
            diagnostics.missing_scene_start,
            diagnostics.non_list_output,
            diagnostics.parse_error or "无",
            response_preview,
        )
        for index, candidate in enumerate(parse_result.candidates[:12], start=1):
            logger.info(
                "%s 行为学习解析候选[%s]: segment_id=%s actor_type=%s learning_type=%s action=%s outcome=%s source_ids=%s",
                learning_session_id,
                index,
                candidate.segment_id or "auto",
                candidate.actor_type,
                candidate.learning_type,
                candidate.action,
                candidate.outcome,
                candidate.source_ids,
            )

    @staticmethod
    def _format_scene_segments_for_prompt(
        segments: Sequence[BehaviorScenarioSegment],
    ) -> str:
        return json.dumps(
            {"segments": [segment.to_prompt_payload() for segment in segments]},
            ensure_ascii=False,
            indent=2,
        )

    @staticmethod
    def _select_segment_for_candidate(
        candidate: BehaviorCandidate,
        segments: Sequence[BehaviorScenarioSegment],
    ) -> BehaviorScenarioSegment:
        if not segments:
            return BehaviorScenarioSegment(
                segment_id="s1",
                title="主场景",
                profile=BehaviorScenarioProfile(),
            )
        if candidate.segment_id:
            for segment in segments:
                if segment.segment_id == candidate.segment_id:
                    return segment
        candidate_source_ids = set(candidate.source_ids)
        best_segment = segments[0]
        best_overlap = -1
        for segment in segments:
            segment_source_ids = set(segment.source_ids)
            overlap = len(candidate_source_ids & segment_source_ids)
            if overlap > best_overlap:
                best_overlap = overlap
                best_segment = segment
        return best_segment

    async def _analyze_learning_scene_segments(
        self,
        records: list[BehaviorMessageRecord],
        *,
        learning_session_id: str,
    ) -> list[BehaviorScenarioSegment]:
        context_text = self._build_learning_context_text(records)
        if not context_text:
            return []

        async def run_scene_prompt(prompt: str) -> str:
            scene_messages = self._build_scene_analysis_messages(records, prompt)
            response = await cast(
                "Any",
                self.scene_model.bind(temperature=BEHAVIOR_SCENE_TEMPERATURE),
            ).ainvoke(
                scene_messages,
                config={"metadata": {"lc_source": "behavior_scene_analyzer"}},
            )
            return content_to_text(response.content)

        segments = await behavior_scenario_analyzer.analyze_segments(
            context_text=context_text,
            sub_agent_runner=run_scene_prompt,
        )
        if segments:
            logger.info(
                "%s 行为学习场景片段分析完成: 片段数=%s 片段=%s",
                learning_session_id,
                len(segments),
                [
                    {
                        "id": segment.segment_id,
                        "sources": segment.source_ids,
                        "title": segment.title,
                    }
                    for segment in segments
                ],
            )
            return segments

        profile = await behavior_scenario_analyzer.analyze(
            context_text=context_text,
            sub_agent_runner=run_scene_prompt,
        )
        if not profile.has_signal:
            return []
        return [
            BehaviorScenarioSegment(
                segment_id="s1",
                title=profile.summary or "主场景",
                source_ids=[str(index) for index in range(1, len(records) + 1)],
                profile=profile,
            )
        ]

    @staticmethod
    def _format_feedback_references_for_system_prompt(
        references: Sequence[BehaviorReferenceCandidate],
    ) -> str:
        formatted_references: list[str] = []
        for index, reference in enumerate(references, start=1):
            formatted_references.append(
                "\n".join(
                    [
                        f"路径 {index}",
                        f"- behavior_id: {reference.behavior_id}",
                        f"- actor_type: {reference.actor_type or 'unknown'}",
                        f"- learning_type: {reference.learning_type or 'unknown'}",
                        f"- session_id: {reference.session_id or 'unknown'}",
                        f"- 采用行为: {reference.action or '[空]'}",
                        f"- 预期结果: {reference.outcome or '[空]'}",
                    ]
                )
            )
        return "\n\n".join(formatted_references)

    @staticmethod
    def _format_feedback_timeline_message(item: BehaviorFeedbackContextItem) -> str:
        return "\n".join(
            [
                "[timeline_item]",
                f"[item_id:{item.item_id}]",
                f"[type:{item.item_type}]",
                f"[speaker:{item.speaker or 'unknown'}]",
                f"[source:{item.source or 'unknown'}]",
                "[content]",
                _compact_log_text(item.text, max_length=900) or "[空]",
            ]
        )

    def _build_behavior_feedback_messages(
        self,
        feedback_context: BehaviorFeedbackContext,
    ) -> list[BaseMessage]:
        prompt = BEHAVIOR_FEEDBACK_PROMPT.format(
            bot_name=BOT_NAME,
            behavior_references=self._format_feedback_references_for_system_prompt(
                feedback_context.references
            ),
        )
        feedback_messages: list[BaseMessage] = [
            SystemMessage(
                content=(
                    f"{prompt}\n\n"
                    "注意：候选行为路径已经在本 system prompt 中列出。"
                    "后续聊天时间线会在后续多条 user message 中给出；每条时间线消息包含 item_id，"
                    "source_ids 必须引用时间线消息中的 item_id。"
                )
            ),
            HumanMessage(content="以下是后续聊天时间线。"),
        ]
        for item in feedback_context.timeline_items:
            feedback_messages.append(
                HumanMessage(content=self._format_feedback_timeline_message(item))
            )
        feedback_messages.append(
            HumanMessage(content="请根据以上行为参考和后续聊天时间线输出反馈 JSON。")
        )
        return feedback_messages

    async def _evaluate_behavior_feedback(
        self,
        session_id: str,
        feedback_context: BehaviorFeedbackContext,
    ) -> set[int]:
        reference_by_id = {
            reference.behavior_id: reference
            for reference in feedback_context.references
        }
        feedback_messages = self._build_behavior_feedback_messages(feedback_context)
        try:
            response = await cast(
                "Any",
                self.feedback_model.bind(temperature=BEHAVIOR_FEEDBACK_TEMPERATURE),
            ).ainvoke(
                feedback_messages,
                config={"metadata": {"lc_source": "evaluate_behavior_feedback"}},
            )
            response_text = content_to_text(response.content)
        except Exception as exc:  # noqa: BLE001
            logger.error("行为路径反馈评估失败: %s", exc)
            return set()

        feedback_items = parse_behavior_feedback_response(response_text)
        if not feedback_items:
            logger.debug("行为路径反馈评估未产生可写入反馈")
            return set()

        wrote_count = 0
        written_behavior_ids: set[int] = set()
        skipped_reasons: Counter[str] = Counter()
        for feedback_item in feedback_items:
            reference = reference_by_id.get(feedback_item.behavior_id)
            if reference is None:
                skipped_reasons["unknown_behavior_id"] += 1
                continue
            evidence_valid, invalid_reason, valid_source_ids = (
                _validate_behavior_feedback_evidence(
                    feedback_item,
                    feedback_context,
                )
            )
            if not evidence_valid:
                skipped_reasons[invalid_reason] += 1
                continue
            feedback_path = await apply_behavior_feedback(
                self._db,
                pattern_id=feedback_item.behavior_id,
                score_delta=feedback_item.score_delta,
                status=feedback_item.status,
                reason=feedback_item.reason,
                outcome=feedback_item.outcome,
                session_id=reference.session_id or session_id,
                source_ids=valid_source_ids,
            )
            if feedback_path is None:
                skipped_reasons["write_failed"] += 1
                continue
            wrote_count += 1
            written_behavior_ids.add(feedback_item.behavior_id)
            logger.info(
                "行为路径反馈已写入: behavior_id=%s status=%s score_delta=%s source_ids=%s",
                feedback_item.behavior_id,
                feedback_item.status,
                feedback_item.score_delta,
                valid_source_ids,
            )
        logger.info(
            "行为路径反馈写入概览: 候选=%s 成功=%s 跳过原因=%s",
            len(feedback_items),
            wrote_count,
            dict(skipped_reasons),
        )
        return written_behavior_ids

    def _filter_behavior_candidates(
        self,
        candidates: list[BehaviorCandidate],
        records: list[BehaviorMessageRecord],
    ) -> BehaviorFilterResult:
        filtered_candidates: list[BehaviorCandidate] = []
        skipped_reasons: Counter[str] = Counter()
        for candidate in candidates:
            if "SELF" in candidate.action or "SELF" in candidate.outcome:
                skipped_reasons["contains_self_literal"] += 1
                logger.info(
                    "跳过包含 SELF 字面量的行为表现: action=%s, outcome=%s",
                    candidate.action,
                    candidate.outcome,
                )
                continue
            valid_source_ids: list[str] = []
            for source_id in candidate.source_ids:
                source_id_str = source_id.strip()
                if not source_id_str.isdigit():
                    continue
                line_index = int(source_id_str) - 1
                if line_index < 0 or line_index >= len(records):
                    continue
                if source_id_str not in valid_source_ids:
                    valid_source_ids.append(source_id_str)
            if not valid_source_ids:
                skipped_reasons["invalid_source_ids"] += 1
                logger.info(
                    "跳过来源无效的行为表现 action=%s outcome=%s source_ids=%s",
                    candidate.action,
                    candidate.outcome,
                    candidate.source_ids,
                )
                continue
            has_source_text = any(
                records[int(source_id) - 1].content.strip()
                for source_id in valid_source_ids
            )
            if not has_source_text:
                skipped_reasons["empty_source_text"] += 1
                logger.info(
                    "跳过来源为空的行为表现 action=%s outcome=%s source_ids=%s",
                    candidate.action,
                    candidate.outcome,
                    valid_source_ids,
                )
                continue
            filtered_candidates.append(
                BehaviorCandidate(
                    action=candidate.action.strip(),
                    outcome=candidate.outcome.strip(),
                    source_ids=valid_source_ids,
                    segment_id=candidate.segment_id.strip(),
                    actor_type=candidate.actor_type,
                    learning_type=candidate.learning_type,
                )
            )
        return BehaviorFilterResult(
            candidates=filtered_candidates,
            skipped_reasons=dict(skipped_reasons),
        )


__all__ = ["BehaviorLearnerMiddleware"]
