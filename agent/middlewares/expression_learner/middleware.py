# ruff: noqa: TC002, TC003, DTZ005, TRY400, TRY401
from __future__ import annotations

import difflib
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast
from typing_extensions import override

import anyio
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
)

from agent.base import ManagerContext, ManagerState, UserMessage
from agent.middlewares.base import BaseDaemonMiddleware, SessionProcessOutput
from agent.prompts.manager import BOT_NAME
from agent.utils import content_to_text

from .constants import (
    DEFAULT_DB_URL,
    DEFAULT_ENABLE_PRECISE_EXPRESSION_SELECTION,
    DEFAULT_EXPRESSION_CHECKED_ONLY,
    DEFAULT_EXPRESSION_SELF_REFLECT,
    DEFAULT_MAX_CONCURRENT_LEARNERS,
    EXPRESSION_EVALUATION_SOURCE,
    EXPRESSION_LEARN_SOURCE,
    EXPRESSION_REPLY_SOURCE,
    EXPRESSION_SUMMARY_SOURCE,
    MAX_LEARNED_EXPRESSIONS_PER_BATCH,
    SIMILARITY_THRESHOLD,
)
from .manager import (
    ExpressionDatabase,
    ExpressionEntry,
    ExpressionMessageRecord,
    ExpressionRecord,
    ExpressionSelector,
    ModifiedBy,
    append_ai_review_log,
    logger,
)
from .prompt import EXPRESSION_EVALUATION_PROMPT, EXPRESSION_LEARN_PROMPT
from .utils import (
    clean_text,
    dump_json_list,
    normalize_expression_runtime_config,
    normalize_expression_scope,
    parse_evaluation_response,
    parse_expression_response,
)

_USER_TAG_PATTERN = re.compile(r"(?is)^<user-message\b[^>]*>(.*?)</user-message>$")
_BOT_TAG_PATTERN = re.compile(r"(?is)^<bot-message\b[^>]*>(.*?)</bot-message>$")


@dataclass(frozen=True)
class PendingExpressionAnalysisBatch:
    session_id: str
    messages: list[BaseMessage]


@dataclass(frozen=True)
class ExpressionLearningAcquireResult:
    acquired: bool
    reason: str = ""
    active_count: int = 0
    max_count: int = 0


class ExpressionLearningBatchGate:
    def __init__(self, max_count: int = DEFAULT_MAX_CONCURRENT_LEARNERS) -> None:
        self.max_count = max_count
        self._lock = anyio.Lock()
        self._active_session_ids: set[str] = set()

    async def acquire(self, session_id: str) -> ExpressionLearningAcquireResult:
        if self.max_count <= 0:
            return ExpressionLearningAcquireResult(
                False,
                "max_expression_learner <= 0",
                0,
                self.max_count,
            )

        async with self._lock:
            active_count = len(self._active_session_ids)
            if session_id in self._active_session_ids:
                return ExpressionLearningAcquireResult(
                    False,
                    "session_busy",
                    active_count,
                    self.max_count,
                )
            if active_count >= self.max_count:
                return ExpressionLearningAcquireResult(
                    False,
                    "global_limit",
                    active_count,
                    self.max_count,
                )
            self._active_session_ids.add(session_id)
            return ExpressionLearningAcquireResult(
                True,
                active_count=active_count + 1,
                max_count=self.max_count,
            )

    async def release(self, session_id: str) -> None:
        async with self._lock:
            self._active_session_ids.discard(session_id)


def _normalize_record_message(message: BaseMessage) -> ExpressionMessageRecord | None:
    if isinstance(message, HumanMessage) and isinstance(
        raw := message.additional_kwargs.get("raw"),
        UserMessage,
    ):
        content = clean_text(raw.message.get_msgcode())
        if not content:
            return None
        return ExpressionMessageRecord(
            speaker="USER",
            content=content,
            name=clean_text(raw.user),
            timestamp=raw.timestamp.strftime("%H:%M:%S"),
        )

    raw_content = content_to_text(message.content)
    if not raw_content.strip():
        return None

    if isinstance(message, HumanMessage):
        stripped = raw_content.strip()
        if match := _USER_TAG_PATTERN.match(stripped):
            content = clean_text(match.group(1))
            if content:
                return ExpressionMessageRecord(
                    speaker="USER",
                    content=content,
                    name="未知用户",
                    timestamp="unknown",
                )
        if match := _BOT_TAG_PATTERN.match(stripped):
            content = clean_text(match.group(1))
            if content:
                return ExpressionMessageRecord(
                    speaker="SELF",
                    content=content,
                    name=BOT_NAME,
                    timestamp="unknown",
                )
        return None

    if isinstance(message, AIMessage):
        content = clean_text(raw_content)
        if content:
            return ExpressionMessageRecord(
                speaker="SELF",
                content=content,
                name=BOT_NAME,
                timestamp="unknown",
            )
    return None


class ExpressionLearnerMiddleware(BaseDaemonMiddleware[PendingExpressionAnalysisBatch]):
    state_schema = ManagerState

    def __init__(
        self,
        analyze_model: BaseChatModel,
        *,
        selection_model: BaseChatModel | None = None,
        review_model: BaseChatModel | None = None,
        summary_model: BaseChatModel | None = None,
        db_url: str = DEFAULT_DB_URL,
        engine: AsyncEngine | None = None,
        max_sessions: int = 100,
        max_batches: int = 3,
        max_retries: int = 3,
        min_messages_for_extraction: int = 10,
        max_concurrent_learners: int = DEFAULT_MAX_CONCURRENT_LEARNERS,
        expression_checked_only: bool = DEFAULT_EXPRESSION_CHECKED_ONLY,
        expression_self_reflect: bool = DEFAULT_EXPRESSION_SELF_REFLECT,
        enable_precise_expression_selection: bool = DEFAULT_ENABLE_PRECISE_EXPRESSION_SELECTION,
        expression_group_resolver: (
            Callable[[str], set[str] | tuple[set[str], bool]] | None
        ) = None,
        expression_config_resolver: Callable[[str], tuple[bool, bool]] | None = None,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
    ) -> None:
        super().__init__(
            max_sessions=max_sessions,
            max_retries=max_retries,
            max_batch_window_size=max_batches,
            logger_config={
                "worker_name": "ExpressionLearner",
                "job_name": "expression learning",
                "process_failure_log": "Failed to analyze expression entries",
                "retry_exhausted_label": "Expression learning",
            },
        )
        self.analyze_model = analyze_model
        self.review_model = review_model or analyze_model
        self.summary_model = summary_model or analyze_model
        self.min_messages_for_extraction = min_messages_for_extraction
        self.expression_self_reflect = expression_self_reflect
        self._expression_group_resolver = expression_group_resolver
        self._expression_config_resolver = expression_config_resolver
        self._similarity_threshold = similarity_threshold
        self._db = ExpressionDatabase(engine=engine, db_url=db_url)
        self._selector = ExpressionSelector(
            self._db,
            selection_model or analyze_model,
            expression_checked_only=expression_checked_only,
            enable_precise_expression_selection=enable_precise_expression_selection,
            group_resolver=expression_group_resolver,
            config_resolver=expression_config_resolver,
        )
        self._learning_gate = ExpressionLearningBatchGate(max_concurrent_learners)

    def _get_expression_config(self, session_id: str) -> tuple[bool, bool]:
        if self._expression_config_resolver is None:
            return True, True
        try:
            resolved = self._expression_config_resolver(session_id)
        except Exception:
            logger.exception("读取表达方式会话配置失败: session_id=%s", session_id)
            return True, True
        config = normalize_expression_runtime_config(resolved)
        return config.use_expression, config.enable_learning

    def _resolve_expression_group_scope(
        self,
        session_id: str,
    ) -> tuple[set[str], bool]:
        if self._expression_group_resolver is None or not session_id:
            return normalize_expression_scope(session_id, None)
        try:
            resolved = self._expression_group_resolver(session_id)
        except Exception:
            logger.exception("Failed to resolve expression group scope for %s", session_id)
            return normalize_expression_scope(session_id, None)
        return normalize_expression_scope(session_id, resolved)

    @override
    async def abefore_agent(
        self,
        state: ManagerState,
        runtime: Runtime[ManagerContext],
    ) -> dict[str, Any] | None:
        session_id = runtime.context["session_id"]
        use_expression, _ = self._get_expression_config(session_id)
        if not use_expression:
            return None

        records = self._extract_visible_expression_messages(
            cast("Sequence[BaseMessage]", state.get("messages") or [])
        )
        if not records:
            return None

        selection = await self._selector.select_for_reply(
            session_id=session_id,
            chat_history=records,
        )
        if not selection.expression_habits:
            return None
        return {
            "reply_bottom_messages": [
                HumanMessage(
                    content=selection.expression_habits,
                    additional_kwargs={"lc_source": EXPRESSION_REPLY_SOURCE},
                )
            ]
        }

    @override
    async def aafter_agent(
        self,
        state: ManagerState,
        runtime: Runtime[ManagerContext],
    ) -> dict[str, Any] | None:
        session_id = runtime.context["session_id"]
        _, enable_learning = self._get_expression_config(session_id)
        if not enable_learning:
            return None

        if messages := cast(
            "list[BaseMessage] | None",
            state.get("summary_pruned_messages"),
        ):
            await self._enqueue_batch(
                session_id,
                PendingExpressionAnalysisBatch(
                    session_id=session_id,
                    messages=list(messages),
                ),
            )
        return None

    @override
    async def process_batches(
        self,
        session_id: str,
        batches: tuple[PendingExpressionAnalysisBatch, ...],
    ) -> SessionProcessOutput:
        records = self._extract_pruned_messages(batches)
        if not records:
            return len(batches)
        if len(records) < self.min_messages_for_extraction:
            logger.debug(
                "%s 表达学习消息不足: 可学习=%s 阈值=%s",
                session_id,
                len(records),
                self.min_messages_for_extraction,
            )
            return len(batches)
        await self._learn_from_session_messages(records, learning_session_id=session_id)
        return len(batches)

    @override
    async def on_close(self) -> None:
        await self._db.dispose()

    def _extract_visible_expression_messages(
        self,
        messages: Sequence[BaseMessage],
    ) -> list[ExpressionMessageRecord]:
        return [
            record
            for message in messages
            if (record := _normalize_record_message(message)) is not None
        ]

    def _extract_pruned_messages(
        self,
        batches: Sequence[PendingExpressionAnalysisBatch],
    ) -> list[ExpressionMessageRecord]:
        records: list[ExpressionMessageRecord] = []
        for batch in batches:
            records.extend(
                [
                    record
                    for message in batch.messages
                    if (record := _normalize_record_message(message))
                ]
            )

        return records

    async def _learn_from_session_messages(
        self,
        records: list[ExpressionMessageRecord],
        *,
        learning_session_id: str,
    ) -> bool:
        acquire_result = await self._learning_gate.acquire(learning_session_id)
        if not acquire_result.acquired:
            if acquire_result.reason == "session_busy":
                logger.info(
                    "%s 已有表达学习批次正在运行，放弃新的批次", learning_session_id
                )
            elif acquire_result.reason == "global_limit":
                logger.info(
                    "表达学习全局并发已满，放弃新的批次: active=%s, max=%s, session_id=%s",
                    acquire_result.active_count,
                    acquire_result.max_count,
                    learning_session_id,
                )
            else:
                logger.warning(
                    "表达学习并发配置无效，放弃新的批次: max_expression_learner=%s, session_id=%s",
                    acquire_result.max_count,
                    learning_session_id,
                )
            return False
        try:
            return await self._run_learning_batch(
                records,
                learning_session_id=learning_session_id,
            )
        finally:
            await self._learning_gate.release(learning_session_id)

    async def _run_learning_batch(
        self,
        records: list[ExpressionMessageRecord],
        *,
        learning_session_id: str,
    ) -> bool:
        prompt = EXPRESSION_LEARN_PROMPT.format(
            chat_str="聊天记录将在后续多条 user message 中给出；请以每条消息中的 source_id 作为来源行编号。"
        )
        try:
            learning_messages = self._build_learning_messages(records, prompt)
            response = await self.analyze_model.ainvoke(
                learning_messages,
                config={"metadata": {"lc_source": EXPRESSION_LEARN_SOURCE}},
            )
            response_text = content_to_text(response.content)
        except Exception as exc:
            logger.error("学习表达方式失败: %s", exc)
            return False

        expressions = parse_expression_response(response_text)
        if len(expressions) > MAX_LEARNED_EXPRESSIONS_PER_BATCH:
            logger.info("表达方式数量超过20: %s", len(expressions))
            expressions = []
        if not expressions:
            logger.info("没有可学习的表达方式")
            return False

        learnt_expressions = self._filter_expressions(expressions, records)
        if not learnt_expressions:
            logger.info("没有可学习的表达方式通过过滤")
            return False

        learnt_expressions_str = "\n".join(
            f"{situation}->{style}" for situation, style in learnt_expressions
        )
        expression_log_title = (
            "待优化的表达方式" if self.expression_self_reflect else "学习到的表达"
        )
        logger.info(
            "[%s] %s：\n%s",
            learning_session_id,
            expression_log_title,
            learnt_expressions_str,
        )

        wrote_expression = False
        for situation, style in learnt_expressions:
            if (
                self.expression_self_reflect
                and not await self._check_expression_before_upsert(
                    situation,
                    style,
                    session_id=learning_session_id,
                )
            ):
                continue
            expression = await self._upsert_expression_to_db(
                situation,
                style,
                session_id=learning_session_id,
                checked=False,
                modified_by=ModifiedBy.AI if self.expression_self_reflect else None,
            )
            wrote_expression = wrote_expression or expression is not None
        return wrote_expression

    @staticmethod
    def _build_learning_messages(
        records: list[ExpressionMessageRecord],
        system_prompt: str,
    ) -> list[BaseMessage]:
        learning_messages: list[BaseMessage] = [
            SystemMessage(
                content=(
                    f"{system_prompt}\n\n"
                    "注意：聊天记录会在后续多条 user message 中给出。每条消息内的 source_id "
                    "是本轮学习的来源编号；speaker=SELF 的消息只作为上下文，不要从 SELF 的发言中学习。"
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

    def _filter_expressions(
        self,
        expressions: list[tuple[str, str, str]],
        records: list[ExpressionMessageRecord],
    ) -> list[tuple[str, str]]:
        filtered_expressions: list[tuple[str, str]] = []
        banned_casefold = {BOT_NAME.casefold()}

        for situation, style, source_id in expressions:
            source_id_str = source_id.strip()
            if not source_id_str.isdigit():
                continue
            line_index = int(source_id_str) - 1
            if line_index < 0 or line_index >= len(records):
                continue
            current_record = records[line_index]
            if current_record.speaker == "SELF":
                continue
            context = current_record.content.strip()
            if not context:
                continue
            if "SELF" in situation or "SELF" in style or "SELF" in context:
                logger.info(
                    "跳过包含 SELF 的表达方式：situation=%s, style=%s, source_id=%s",
                    situation,
                    style,
                    source_id,
                )
                continue
            normalized_style = style.strip()
            if normalized_style and normalized_style.casefold() in banned_casefold:
                logger.debug(
                    "跳过 style 与机器人名称重复的表达方式：situation=%s, style=%s, source_id=%s",
                    situation,
                    style,
                    source_id,
                )
                continue
            if "[表情包" in situation or "[表情包" in style or "[表情包" in context:
                logger.info(
                    "跳过包含表情标记的表达方式：situation=%s, style=%s, source_id=%s",
                    situation,
                    style,
                    source_id,
                )
                continue
            if "[图片" in situation or "[图片" in style or "[图片" in context:
                logger.info(
                    "跳过包含图片标记的表达方式：situation=%s, style=%s, source_id=%s",
                    situation,
                    style,
                    source_id,
                )
                continue
            filtered_expressions.append((situation, style))
        return filtered_expressions

    async def _upsert_expression_to_db(
        self,
        situation: str,
        style: str,
        *,
        session_id: str,
        checked: bool = False,
        modified_by: ModifiedBy | None = None,
    ) -> ExpressionEntry | None:
        result = await self._find_similar_expression(situation, session_id=session_id)
        expr, similarity = result or (None, 0.0)
        if expr is not None:
            use_llm_summary = similarity < 1.0
            return await self._update_existing_expression(
                expr,
                situation,
                use_llm_summary=use_llm_summary,
                checked=checked,
                modified_by=modified_by,
            )
        return await self._create_expression(
            situation,
            style,
            session_id=session_id,
            checked=checked,
            modified_by=modified_by,
        )

    async def _create_expression(
        self,
        situation: str,
        style: str,
        *,
        session_id: str,
        checked: bool = False,
        modified_by: ModifiedBy | None = None,
    ) -> ExpressionEntry | None:
        content_list = [situation]
        try:
            async with self._db.session() as session:
                new_expr = ExpressionRecord(
                    situation=situation,
                    style=style,
                    content_list=json.dumps(content_list, ensure_ascii=False),
                    count=1,
                    session_id=session_id,
                    last_active_time=datetime.now(),
                    checked=checked,
                    modified_by=modified_by,
                )
                session.add(new_expr)
                await session.flush()
                await session.refresh(new_expr)
                return ExpressionEntry.from_db_instance(new_expr)
        except Exception as exc:
            logger.error("创建表达方式失败: %s", exc)
        return None

    async def _update_existing_expression(
        self,
        expr: ExpressionEntry,
        situation: str,
        *,
        use_llm_summary: bool = True,
        checked: bool = False,
        modified_by: ModifiedBy | None = None,
    ) -> ExpressionEntry | None:
        expr.content.append(situation)
        expr.count += 1
        expr.checked = checked
        expr.modified_by = modified_by
        expr.last_active_time = datetime.now()

        if use_llm_summary:
            new_situation = await self._compose_situation_text(expr.content)
            if new_situation:
                expr.situation = new_situation

        try:
            async with self._db.session() as session:
                if expr.item_id is None:
                    msg = "表达方式对象缺少 item_id，无法更新数据库记录"
                    raise ValueError(msg)
                db_expr = await session.get(ExpressionRecord, expr.item_id)
                if db_expr is None:
                    logger.warning(
                        "表达方式 ID %s 在数据库中未找到，无法更新", expr.item_id
                    )
                    return None
                db_expr.content_list = dump_json_list(expr.content)
                db_expr.count = expr.count
                db_expr.checked = expr.checked
                db_expr.modified_by = expr.modified_by
                db_expr.last_active_time = expr.last_active_time
                db_expr.situation = expr.situation
                session.add(db_expr)
                return expr
        except Exception as exc:
            logger.error("更新表达方式失败: %s", exc)
        return None

    async def _check_expression_before_upsert(
        self,
        situation: str,
        style: str,
        *,
        session_id: str,
    ) -> bool:
        suitable, reason, error = await self._check_expression_suitability(
            situation,
            style,
        )
        if error:
            await append_ai_review_log(
                session_id=session_id,
                situation=situation,
                style=style,
                passed=False,
                reason=reason or error,
                source="learn_before_upsert",
                error=error,
            )
            logger.error("检查表达方式时发生错误: %s", error)
            return False

        await append_ai_review_log(
            session_id=session_id,
            situation=situation,
            style=style,
            passed=suitable,
            reason=reason,
            source="learn_before_upsert",
        )
        status = "通过" if suitable else "不通过"
        logger.info(
            "表达方式检查 - %s | Situation: %s | Style: %s || Reason: %s...",
            status,
            situation,
            style,
            reason[:100] if reason else "无",
        )
        return suitable

    async def _check_expression_suitability(
        self,
        situation: str,
        style: str,
    ) -> tuple[bool, str, str | None]:
        base_criteria = [
            "表达方式或言语风格是否与使用条件或使用情景匹配",
            "允许部分语法错误或口语化或缺省出现",
            "表达方式不能太过特指，需要具有泛用性",
            "一般不涉及具体的人名或名称",
        ]
        criteria_list = "\n".join(
            f"{index + 1}. {criterion}" for index, criterion in enumerate(base_criteria)
        )
        prompt = EXPRESSION_EVALUATION_PROMPT.format(
            situation=situation,
            style=style,
            criteria_list=criteria_list,
        )

        logger.info("正在评估表达方式: situation=%s, style=%s", situation, style)
        try:
            response = await self.review_model.ainvoke(
                prompt,
                config={"metadata": {"lc_source": EXPRESSION_EVALUATION_SOURCE}},
            )
            response_text = content_to_text(response.content)
        except Exception as exc:
            return False, f"评估表达方式时发生错误: {exc}", str(exc)

        logger.debug("评估结果: %s", response_text)
        try:
            evaluation = parse_evaluation_response(response_text)
        except Exception as exc:
            return False, f"评估表达方式时发生错误: {exc}", str(exc)

        try:
            suitable = bool(evaluation.get("suitable", False))
            reason = str(evaluation.get("reason", "未提供理由")).strip()
            logger.debug("评估结果: %s", "通过" if suitable else "不通过")

        except Exception as exc:
            return False, f"评估结果格式错误: {exc}", str(exc)
        else:
            return suitable, reason, None

    async def _compose_situation_text(
        self,
        content_list: list[str],
    ) -> str | None:
        texts = [content.strip() for content in content_list if content.strip()]
        if not texts:
            return None
        description = "\n".join(f"- {text}" for text in texts[-10:])
        prompt = (
            "请阅读以下多个聊天情境描述，并将它们概括成一句简短的话，长度不超过20个字，保留共同特点：\n"
            f"{description}\n"
            "只输出概括内容。"
        )
        try:
            response = await self.summary_model.ainvoke(
                prompt,
                config={"metadata": {"lc_source": EXPRESSION_SUMMARY_SOURCE}},
            )
            summary = content_to_text(response.content).strip()
        except Exception as exc:
            logger.error("使用 LLM 生成表达方式概括失败: %s", exc)
        else:
            return summary or None
        return None

    async def _find_similar_expression(
        self,
        situation: str,
        *,
        session_id: str,
    ) -> tuple[ExpressionEntry, float] | None:
        try:
            related_session_ids, has_global_share = self._resolve_expression_group_scope(
                session_id
            )
            async with self._db.session(auto_commit=False) as session:
                statement = select(ExpressionRecord)
                if not has_global_share:
                    statement = statement.where(
                        ExpressionRecord.session_id.in_(related_session_ids)
                    )
                expressions = list((await session.scalars(statement)).all())

            best_match: ExpressionEntry | None = None
            best_similarity = 0.0
            for db_expression in expressions:
                expression = ExpressionEntry.from_db_instance(db_expression)
                candidate_situations = [expression.situation, *expression.content]
                for candidate_situation in candidate_situations:
                    normalized_candidate = candidate_situation.strip()
                    if not normalized_candidate:
                        continue
                    similarity = difflib.SequenceMatcher(
                        None,
                        situation,
                        normalized_candidate,
                    ).ratio()
                    if (
                        similarity > self._similarity_threshold
                        and similarity > best_similarity
                    ):
                        best_similarity = similarity
                        best_match = expression

            if best_match is not None:
                logger.debug(
                    "找到相似表达方式情景 [ID: %s]，相似度: %.2f",
                    best_match.item_id,
                    best_similarity,
                )
                return best_match, best_similarity
        except Exception as exc:
            logger.error("查找相似表达方式失败: %s", exc)
        return None


__all__ = ["ExpressionLearnerMiddleware"]
