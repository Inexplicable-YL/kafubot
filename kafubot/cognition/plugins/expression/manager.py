# ruff: noqa: TC002, TC003, DTZ005, TRY400, TRY401
from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import anyio
import jieba
from json_repair import repair_json
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    or_,
    select,
)
from sqlalchemy import (
    Enum as SQLEnum,
)
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from kafubot.cognition.utils import content_to_text

from .constants import (
    AI_REVIEW_EVENT,
    DEFAULT_DB_URL,
    EXPRESSION_REVIEW_LOG_PATH,
    EXPRESSION_SELECTOR_SOURCE,
    MANUAL_RESCUE_EVENT,
    MAX_PRECISE_SELECTED_EXPRESSIONS,
    MAX_VISIBLE_CONTEXT_MESSAGES,
    MIN_CANDIDATE_POOL_SIZE,
)
from .prompt import build_expression_selector_prompt
from .utils import (
    ExpressionRuntimeConfig,
    clean_text,
    load_json_list,
    normalize_expression_runtime_config,
    normalize_expression_scope,
)

logger = logging.getLogger(__name__)

_review_log_lock = anyio.Lock()


class ModifiedBy(str, Enum):
    AI = "AI"
    USER = "USER"


class ExpressionMessageRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    speaker: str
    content: str
    name: str = ""
    timestamp: str = ""


class ExpressionSelectionResult(BaseModel):
    expression_habits: str = ""
    selected_expression_ids: list[int] = Field(default_factory=list)
    selected_expressions: list[dict[str, Any]] = Field(default_factory=list)


class ExpressionEntry(BaseModel):
    situation: str
    style: str
    content: list[str]
    count: int
    last_active_time: datetime
    create_time: datetime
    item_id: int | None = None
    session_id: str | None = None
    checked: bool = False
    modified_by: ModifiedBy | None = None

    @classmethod
    def from_db_instance(cls, db_record: ExpressionRecord) -> ExpressionEntry:
        content_list = load_json_list(db_record.content_list)
        normalized_content = [
            str(item) for item in content_list if isinstance(item, str)
        ]
        return cls(
            item_id=db_record.id,
            situation=db_record.situation,
            style=db_record.style,
            content=normalized_content,
            count=db_record.count,
            last_active_time=db_record.last_active_time,
            create_time=db_record.create_time,
            session_id=db_record.session_id,
            checked=db_record.checked,
            modified_by=db_record.modified_by,
        )


class _ExpressionBase(DeclarativeBase):
    pass


class ExpressionRecord(_ExpressionBase):
    __tablename__ = "expressions"

    id: Mapped[int | None] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        nullable=False,
    )
    situation: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    style: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    content_list: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_active_time: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.now,
        index=True,
    )
    create_time: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.now,
    )
    session_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        default=None,
        index=True,
    )
    checked: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )
    modified_by: Mapped[ModifiedBy | None] = mapped_column(
        SQLEnum(ModifiedBy),
        nullable=True,
        default=None,
    )


class ExpressionEffectRecord(_ExpressionBase):
    __tablename__ = "expression_effects"

    expression_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    observations: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    reward_sum: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    last_confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class ExpressionDatabase:
    def __init__(
        self,
        *,
        engine: AsyncEngine | None = None,
        db_url: str = DEFAULT_DB_URL,
    ) -> None:
        self.engine = engine or create_async_engine(
            self._normalize_async_db_url(db_url)
        )
        self._sessionmaker = async_sessionmaker(self.engine, expire_on_commit=False)
        self._schema_ready = False
        self._schema_lock = anyio.Lock()

    @classmethod
    def _normalize_async_db_url(cls, db_url: str) -> str:
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
                await conn.run_sync(_ExpressionBase.metadata.create_all)
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


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


async def read_review_log_entries() -> list[dict[str, Any]]:
    log_path = anyio.Path(EXPRESSION_REVIEW_LOG_PATH)
    if not await log_path.exists():
        return []

    try:
        content = await log_path.read_text(encoding="utf-8")
        parsed = json.loads(content)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError as exc:
        logger.warning("表达方式审核日志 JSON 解析失败，已忽略当前文件: %s", exc)
        return []

    if not isinstance(parsed, list):
        logger.warning("表达方式审核日志格式异常，应为 JSON 数组")
        return []
    return [entry for entry in parsed if isinstance(entry, dict)]


async def _append_log_entry(entry: dict[str, Any]) -> None:
    async with _review_log_lock:
        entries = await read_review_log_entries()
        entries.append(entry)
        log_path = anyio.Path(EXPRESSION_REVIEW_LOG_PATH)
        await log_path.parent.mkdir(parents=True, exist_ok=True)
        await log_path.write_text(
            json.dumps(entries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


async def append_ai_review_log(
    *,
    session_id: str,
    situation: str,
    style: str,
    passed: bool,
    reason: str,
    source: str,
    expression_id: int | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": uuid.uuid4().hex,
        "event": AI_REVIEW_EVENT,
        "created_at": _now_iso(),
        "expression_id": expression_id,
        "session_id": str(session_id or "").strip(),
        "passed": bool(passed),
        "reason": str(reason or "").strip(),
        "situation": str(situation or "").strip(),
        "style": str(style or "").strip(),
        "source": source,
    }
    if error:
        entry["error"] = str(error)

    await _append_log_entry(entry)
    logger.info(
        "表达方式 AI 审核记录已写入 %s: passed=%s session_id=%s source=%s reason=%s",
        EXPRESSION_REVIEW_LOG_PATH,
        entry["passed"],
        entry["session_id"],
        source,
        entry["reason"],
    )
    return entry


async def append_manual_rescue_log(
    *,
    review_log_id: str,
    expression_id: int,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": uuid.uuid4().hex,
        "event": MANUAL_RESCUE_EVENT,
        "created_at": _now_iso(),
        "review_log_id": str(review_log_id),
        "expression_id": int(expression_id),
    }
    await _append_log_entry(entry)
    return entry


def _parse_created_at(value: Any) -> datetime:
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return datetime.fromtimestamp(0, tz=UTC)
    return datetime.fromtimestamp(0, tz=UTC)


def _rescue_by_review_id(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    rescue_entries = [
        entry
        for entry in entries
        if entry.get("event") == MANUAL_RESCUE_EVENT
        and str(entry.get("review_log_id") or "").strip()
    ]
    rescue_entries.sort(
        key=lambda entry: _parse_created_at(entry.get("created_at")),
        reverse=True,
    )
    rescues: dict[str, dict[str, Any]] = {}
    for rescue_entry in rescue_entries:
        review_log_id = str(rescue_entry.get("review_log_id") or "").strip()
        rescues.setdefault(review_log_id, rescue_entry)
    return rescues


def _with_rescue_state(
    entry: dict[str, Any],
    rescue_entry: dict[str, Any] | None,
) -> dict[str, Any]:
    enriched_entry = dict(entry)
    enriched_entry["rescued"] = rescue_entry is not None
    enriched_entry["rescued_expression_id"] = (
        rescue_entry.get("expression_id") if rescue_entry else None
    )
    enriched_entry["rescued_at"] = (
        rescue_entry.get("created_at") if rescue_entry else None
    )
    return enriched_entry


async def get_recent_ai_review_logs(
    *,
    limit: int = 50,
    passed: bool | None = None,
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    entries = await read_review_log_entries()
    rescues = _rescue_by_review_id(entries)
    normalized_session_id = str(session_id or "").strip()
    review_entries = [
        entry
        for entry in entries
        if entry.get("event", AI_REVIEW_EVENT) == AI_REVIEW_EVENT
        and str(entry.get("id") or "").strip()
    ]

    if passed is not None:
        review_entries = [
            entry for entry in review_entries if bool(entry.get("passed")) is passed
        ]
    if normalized_session_id:
        review_entries = [
            entry
            for entry in review_entries
            if str(entry.get("session_id") or "").strip() == normalized_session_id
        ]

    review_entries.sort(
        key=lambda entry: _parse_created_at(entry.get("created_at")),
        reverse=True,
    )
    normalized_limit = max(1, int(limit))
    return [
        _with_rescue_state(entry, rescues.get(str(entry.get("id"))))
        for entry in review_entries[:normalized_limit]
    ]


async def get_ai_review_log(review_log_id: str) -> dict[str, Any] | None:
    normalized_id = str(review_log_id or "").strip()
    if not normalized_id:
        return None

    entries = await read_review_log_entries()
    rescues = _rescue_by_review_id(entries)
    for entry in entries:
        if entry.get("event", AI_REVIEW_EVENT) != AI_REVIEW_EVENT:
            continue
        if str(entry.get("id") or "").strip() == normalized_id:
            return _with_rescue_state(entry, rescues.get(normalized_id))
    return None


class ExpressionSelector:
    def __init__(
        self,
        db: ExpressionDatabase,
        selection_model: BaseChatModel | None,
        *,
        expression_checked_only: bool,
        enable_precise_expression_selection: bool,
        group_resolver: (
            Callable[[str], set[str] | tuple[set[str], bool]] | None
        ) = None,
        config_resolver: Callable[[str], tuple[bool, bool]] | None = None,
    ) -> None:
        self._db = db
        self._selection_model = selection_model
        self._expression_checked_only = expression_checked_only
        self._enable_precise_expression_selection = enable_precise_expression_selection
        self._group_resolver = group_resolver
        self._config_resolver = config_resolver

    def _get_runtime_config(self, session_id: str) -> ExpressionRuntimeConfig:
        if self._config_resolver is None:
            return ExpressionRuntimeConfig()
        try:
            resolved = self._config_resolver(session_id)
        except Exception:
            logger.exception("检查表达方式使用开关失败: session_id=%s", session_id)
            return ExpressionRuntimeConfig()
        return normalize_expression_runtime_config(resolved)

    def _can_use_expressions(self, session_id: str) -> bool:
        return self._get_runtime_config(session_id).use_expression

    def _resolve_expression_group_scope(
        self,
        session_id: str,
    ) -> tuple[set[str], bool]:
        if self._group_resolver is None or not session_id:
            return normalize_expression_scope(session_id, None)
        try:
            resolved = self._group_resolver(session_id)
        except Exception:
            logger.exception(
                "Failed to resolve expression group scope for %s",
                session_id,
            )
            return normalize_expression_scope(session_id, None)
        return normalize_expression_scope(session_id, resolved)

    async def _load_expression_candidates(
        self,
        session_id: str,
        context_text: str,
    ) -> list[dict[str, Any]]:
        related_session_ids, has_global_share = self._resolve_expression_group_scope(
            session_id
        )
        async with self._db.session(auto_commit=False) as session:
            statement = select(ExpressionRecord)
            if not has_global_share:
                statement = statement.where(
                    or_(
                        ExpressionRecord.session_id.in_(related_session_ids),
                        ExpressionRecord.session_id.is_(None),
                    )
                )
            if self._expression_checked_only:
                statement = statement.where(
                    ExpressionRecord.checked.is_(True),
                    ExpressionRecord.modified_by == ModifiedBy.USER,
                )
            expressions = list((await session.scalars(statement)).all())
            expression_ids = [
                expression.id for expression in expressions if expression.id is not None
            ]
            effect_rows = {
                row.expression_id: row
                for row in (
                    await session.scalars(
                        select(ExpressionEffectRecord).where(
                            ExpressionEffectRecord.expression_id.in_(expression_ids)
                        )
                    )
                ).all()
            }

        all_candidates = [
            {
                "id": expression.id,
                "situation": expression.situation,
                "style": expression.style,
                "count": expression.count if expression.count is not None else 1,
                "effect_score": (
                    effect_rows[expression.id].reward_sum
                    / max(1, effect_rows[expression.id].observations)
                    if expression.id in effect_rows
                    else 0.0
                ),
                "context_score": 0.0,
            }
            for expression in expressions
            if expression.id is not None and expression.situation and expression.style
        ]
        if not all_candidates:
            return []
        context_tokens = {
            token.casefold().strip()
            for token in jieba.lcut(clean_text(context_text))
            if len(token.strip()) >= 2
        }
        for candidate in all_candidates:
            candidate_tokens = {
                token.casefold().strip()
                for token in jieba.lcut(
                    f"{candidate['situation']} {candidate['style']}"
                )
                if len(token.strip()) >= 2
            }
            union = context_tokens | candidate_tokens
            candidate["context_score"] = (
                len(context_tokens & candidate_tokens) / len(union) if union else 0.0
            )
        all_candidates.sort(
            key=lambda item: (
                float(item.get("context_score", 0.0)),
                float(item.get("effect_score", 0.0)),
                int(item.get("count", 1)),
                -int(item.get("id", 0)),
            ),
            reverse=True,
        )
        return all_candidates[: max(MIN_CANDIDATE_POOL_SIZE, 12)]

    @classmethod
    def _format_candidate_preview(
        cls,
        candidates: list[dict[str, Any]],
    ) -> str:
        preview_items = [
            "id={id_}, situation={situation!r}, style={style!r}, count={count}".format(
                id_=candidate.get("id"),
                situation=str(candidate.get("situation") or "").strip(),
                style=str(candidate.get("style") or "").strip(),
                count=candidate.get("count"),
            )
            for candidate in candidates[:5]
        ]
        return "; ".join(preview_items)

    @classmethod
    def build_expression_habits_block(
        cls,
        selected_expressions: list[dict[str, Any]],
    ) -> str:
        if not selected_expressions:
            return ""
        lines = [
            f"""- 当"{expression["situation"]}"时，可以用"{expression["style"]}"来表达。"""
            for expression in selected_expressions
        ]
        return "【表达习惯参考，请视情况自然的使用】\n" + "\n".join(lines)

    @classmethod
    def _normalize_history_line(
        cls,
        message: ExpressionMessageRecord,
    ) -> str:
        content = clean_text(message.content)
        if len(content) > 120:
            content = content[:120] + "..."
        return f"- {message.timestamp} {message.speaker}: {content}".strip()

    def _build_selector_prompt(
        self,
        *,
        chat_history: list[ExpressionMessageRecord],
        target_message: str,
        reply_reason: str,
        candidates: list[dict[str, Any]],
    ) -> str:
        history_lines = [
            self._normalize_history_line(message)
            for message in chat_history[-MAX_VISIBLE_CONTEXT_MESSAGES:]
            if clean_text(message.content)
        ]
        history_block = "\n".join(history_lines) if history_lines else "- 无可用上下文"
        candidate_lines = [
            "{}: 情景={} | 风格={} | count={}".format(
                candidate["id"],
                candidate["situation"],
                candidate["style"],
                candidate["count"],
            )
            for candidate in candidates
        ]
        return build_expression_selector_prompt(
            history_block=history_block,
            target_text=target_message,
            reply_reason=reply_reason,
            candidate_lines=candidate_lines,
        )

    @classmethod
    def _parse_selected_ids(
        cls,
        raw_response: str,
        candidates: list[dict[str, Any]],
    ) -> list[int]:
        if not raw_response.strip():
            return []
        try:
            parsed_result = json.loads(repair_json(raw_response))
        except Exception:
            logger.warning("表达方式选择结果解析失败: %r", raw_response)
            return []

        raw_selected_ids = (
            parsed_result.get("selected_ids", [])
            if isinstance(parsed_result, dict)
            else []
        )
        if not isinstance(raw_selected_ids, list):
            return []

        candidate_map = {
            candidate["id"]: candidate
            for candidate in candidates
            if isinstance(candidate.get("id"), int)
        }
        selected_ids: list[int] = []
        for candidate_id in raw_selected_ids:
            try:
                normalized_candidate_id = int(candidate_id)
            except (TypeError, ValueError):
                continue
            if (
                normalized_candidate_id not in candidate_map
                or normalized_candidate_id in selected_ids
            ):
                continue
            selected_ids.append(normalized_candidate_id)
            if len(selected_ids) >= MAX_PRECISE_SELECTED_EXPRESSIONS:
                break
        return selected_ids

    async def _build_direct_selection_result(
        self,
        *,
        session_id: str,
        candidates: list[dict[str, Any]],
    ) -> ExpressionSelectionResult:
        selected_ids = [
            candidate["id"]
            for candidate in candidates[:MAX_PRECISE_SELECTED_EXPRESSIONS]
            if isinstance(candidate.get("id"), int)
        ]
        selected_expressions = [
            candidate for candidate in candidates if candidate.get("id") in selected_ids
        ]
        logger.debug(
            "表达方式直接注入：session_id=%s 已选数=%s selected_ids=%r 已选预览=%s",
            session_id,
            len(selected_ids),
            selected_ids,
            self._format_candidate_preview(selected_expressions),
        )
        return ExpressionSelectionResult(
            expression_habits=self.build_expression_habits_block(selected_expressions),
            selected_expression_ids=selected_ids,
            selected_expressions=list(selected_expressions),
        )

    async def _build_selection_result_from_ids(
        self,
        *,
        candidates: list[dict[str, Any]],
        selected_ids: list[int],
    ) -> ExpressionSelectionResult:
        candidate_map = {
            candidate["id"]: candidate
            for candidate in candidates
            if isinstance(candidate.get("id"), int)
        }
        selected_expressions = [
            candidate_map[expression_id]
            for expression_id in selected_ids
            if expression_id in candidate_map
        ]
        return ExpressionSelectionResult(
            expression_habits=self.build_expression_habits_block(selected_expressions),
            selected_expression_ids=selected_ids,
            selected_expressions=list(selected_expressions),
        )

    async def _build_default_selection_result(
        self,
        *,
        session_id: str,
        chat_history: list[ExpressionMessageRecord],
        target_message: str,
        reply_reason: str,
        candidates: list[dict[str, Any]],
    ) -> ExpressionSelectionResult:
        if not self._enable_precise_expression_selection:
            return await self._build_direct_selection_result(
                session_id=session_id,
                candidates=candidates,
            )
        if self._selection_model is None:
            logger.info("精细表达选择已跳过：缺少子代理执行器，回退为直接注入")
            return await self._build_direct_selection_result(
                session_id=session_id,
                candidates=candidates,
            )

        selector_prompt = self._build_selector_prompt(
            chat_history=chat_history,
            target_message=target_message,
            reply_reason=reply_reason,
            candidates=candidates,
        )
        try:
            response = await self._selection_model.ainvoke(
                selector_prompt,
                config={"metadata": {"lc_source": EXPRESSION_SELECTOR_SOURCE}},
            )
            raw_response = content_to_text(response.content)
        except Exception as exc:
            logger.warning("精细表达选择子代理执行失败，回退为直接注入: %s", exc)
            return await self._build_direct_selection_result(
                session_id=session_id,
                candidates=candidates,
            )

        selected_ids = self._parse_selected_ids(raw_response, candidates)
        logger.debug(
            "精细表达选择完成：session_id=%s selected_ids=%r 候选预览=%s",
            session_id,
            selected_ids,
            self._format_candidate_preview(candidates),
        )
        return await self._build_selection_result_from_ids(
            candidates=candidates,
            selected_ids=selected_ids,
        )

    async def _update_last_active_time(self, selected_ids: list[int]) -> None:
        if not selected_ids:
            return
        async with self._db.session() as session:
            statement = select(ExpressionRecord).where(
                ExpressionRecord.id.in_(selected_ids)
            )
            expressions = list((await session.scalars(statement)).all())
            now = datetime.now()
            for expression in expressions:
                expression.last_active_time = now
                session.add(expression)

    async def select_for_reply(
        self,
        *,
        session_id: str,
        chat_history: list[ExpressionMessageRecord],
        target_message: str = "",
        reply_reason: str = "",
    ) -> ExpressionSelectionResult:
        if not session_id:
            logger.info("表达方式选择已跳过：缺少 session_id")
            return ExpressionSelectionResult()
        if not self._can_use_expressions(session_id):
            logger.info(
                "表达方式选择已跳过：当前会话未启用表达方式，session_id=%s",
                session_id,
            )
            return ExpressionSelectionResult()

        context_text = "\n".join(
            message.content for message in chat_history[-MAX_VISIBLE_CONTEXT_MESSAGES:]
        )
        candidates = await self._load_expression_candidates(session_id, context_text)
        if not candidates:
            logger.info("表达方式选择已跳过：本地候选不足，session_id=%s", session_id)
            return ExpressionSelectionResult()

        return await self._build_default_selection_result(
            session_id=session_id,
            chat_history=chat_history,
            target_message=target_message,
            reply_reason=reply_reason,
            candidates=candidates,
        )


__all__ = [
    "ExpressionDatabase",
    "ExpressionEffectRecord",
    "ExpressionEntry",
    "ExpressionMessageRecord",
    "ExpressionRecord",
    "ExpressionSelectionResult",
    "ExpressionSelector",
    "ModifiedBy",
    "append_ai_review_log",
    "append_manual_rescue_log",
    "get_ai_review_log",
    "get_recent_ai_review_logs",
    "logger",
]
