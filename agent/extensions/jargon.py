import json
import logging
import random
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast

import anyio
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from json_repair import repair_json
from langchain.agents.middleware import AgentMiddleware
from langchain.tools import ToolRuntime, tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore, SearchItem
from pydantic import BaseModel, Field

from agent.base import ManagerContext, ManagerState, UserMessage
from agent.prompts.manager import BOT_NAME
from agent.utils import content_to_text

if TYPE_CHECKING:
    from anyio.abc import TaskGroup

logger = logging.getLogger(__name__)

JARGON_CREATED_BY_AI = "ai"
JARGON_CREATED_BY_MANUAL = "manual"
JARGON_INFERENCE_THRESHOLDS = (4, 8, 25, 100)
JARGON_SAMPLE_RAW_CONTENT_THRESHOLDS = {25}
JARGON_PREVIOUS_MEANING_THRESHOLDS = {25, 100}

JARGON_EXTRACTION_PROMPT = """
你正在从聊天记录中提取“疑似黑话/梗/圈内简称/缩写/代称/有特殊义项的词条”。

请严格按下面要求返回 JSON 数组，除此之外不要输出任何其他内容：
[
  {{
    "content": "词条原文",
    "source_id": "对应消息的 message_id"
  }}
]

要求：
1. 只提取词或短语，不提取完整句子。
2. `content` 必须是聊天记录里真实出现过的原文。
3. `source_id` 必须来自聊天记录中的 `<user-message message_id=...>`。
4. 普通常见词、常规成语、明显的人名地名、单个汉字、单个英文字母、单个数字、纯标点、纯表情、URL，不要提取。
5. 如果一个词只是字面义，没有明显特殊含义，也不要提取。
6. 允许同一个 `content` 对应多个 `source_id`，请分别输出多项。
7. 机器人自己的发言可能有误，不应作为唯一依据。

聊天记录如下：
{messages}
""".strip()

JARGON_INFERENCE_WITH_CONTEXT_PROMPT = """
**词条内容**
{content}
**词条出现的上下文。其中的{bot_name}的发言内容是你自己的发言**
{raw_content_list}
{previous_meaning_section}

请根据上下文，推断"{content}"这个词条的含义。
- 如果这是一个黑话、俚语或网络用语，请推断其含义
- 如果含义明确（常规词汇），也请说明
- {bot_name} 的发言内容可能包含错误，请不要参考其发言内容
- 如果上下文信息不足，无法推断含义，请设置 no_info 为 true
{previous_meaning_instruction}

以 JSON 格式输出：
{{
  "meaning": "详细含义说明（包含使用场景、来源、具体解释等）",
  "no_info": false
}}
注意：如果信息不足无法推断，请设置 "no_info": true，此时 meaning 可以为空字符串
""".strip()

JARGON_INFERENCE_CONTENT_ONLY_PROMPT = """
**词条内容**
{content}

请仅根据这个词条本身，推断其含义。
- 如果这是一个黑话、俚语或网络用语，请推断其含义
- 如果含义明确（常规词汇），也请说明

以 JSON 格式输出：
{{
  "meaning": "详细含义说明（包含使用场景、来源、具体解释等）"
}}
""".strip()

JARGON_COMPARE_INFERENCE_PROMPT = """
**推断结果1（基于上下文）**
{inference1}

**推断结果2（仅基于词条）**
{inference2}

请比较这两个推断结果，判断它们是否相同或类似。
请忽略细微的差别，关注主要含义是否相似

以 JSON 格式输出：
{{
  "is_similar": true/false,
  "reason": "判断理由"
}}
""".strip()


class SearchJargonInput(BaseModel):
    keyword: str = Field(description="搜索关键词。")
    limit: int = Field(default=10, ge=1, le=20, description="返回结果数量限制。")
    case_sensitive: bool = Field(
        default=False,
        description="是否大小写敏感，默认 False（不敏感）。",
    )
    fuzzy: bool = Field(
        default=True,
        description="是否模糊搜索，默认 True（使用 contains 匹配）。",
    )


class JargonEntry(TypedDict):
    content: str
    raw_content: set[str]


class JargonRecord(TypedDict):
    content: str
    raw_content: list[str]
    session_id_dict: dict[str, int]
    session_id: str
    is_global: bool
    other_match_count: int
    count: int
    meaning: str
    is_jargon: bool
    is_complete: bool
    last_inference_count: int
    created_by: Literal["ai", "manual"]
    created_at: str
    updated_at: str


@dataclass(slots=True)
class PendingJargonAnalysisBatch:
    session_id: str
    messages: list[BaseMessage]


class JargonMiddleware(AgentMiddleware[ManagerState, ManagerContext]):
    state_schema = ManagerState

    def __init__(
        self,
        analyze_model: BaseChatModel,
        *,
        jargon_context_batches: int = 3,
        queue_size: int = 100,
        max_message_chars: int = 600,
        max_extract_input_chars: int = 18000,
        max_raw_content_chars: int = 360,
        retry_max_attempts: int = 3,
        retry_backoff_seconds: float = 5.0,
        to_global_threshold: int | None = None,
        store: BaseStore | None = None,
        namespace_root: str = "jargon",
    ) -> None:
        super().__init__()
        self.analyze_model = analyze_model
        self.jargon_context_batches = jargon_context_batches
        self.queue_size = queue_size
        self.max_message_chars = max_message_chars
        self.max_extract_input_chars = max_extract_input_chars
        self.max_raw_content_chars = max_raw_content_chars
        self.retry_max_attempts = retry_max_attempts
        self.retry_backoff_seconds = retry_backoff_seconds
        self.to_global_threshold = to_global_threshold
        self.namespace_root = _clean_text(namespace_root).replace(".", "_") or "jargon"
        self._store = store

        self._pending_batches: dict[str, deque[PendingJargonAnalysisBatch]] = (
            defaultdict(deque)
        )
        self._scheduled_sessions: set[str] = set()
        self._delayed_retry_sessions: set[str] = set()
        self._retry_attempts: dict[str, int] = defaultdict(int)

        self._start_lock = anyio.Lock()
        self._task_group: TaskGroup | None = None
        self._send_stream: MemoryObjectSendStream[str] | None = None
        self._receive_stream: MemoryObjectReceiveStream[str] | None = None

        self.tools = [
            tool(
                "search_jargon",
                description="搜索 jargon，支持大小写不敏感和模糊搜索。适合查询当前聊天里已经学到并成功解释过的黑话含义。",
                args_schema=SearchJargonInput,
            )(self._query_jargon)
        ]

    async def aafter_agent(
        self,
        state: ManagerState,
        runtime: Runtime[ManagerContext],
    ) -> dict[str, Any] | None:
        if runtime.store is not None:
            self._store = self._store or runtime.store
        if messages := cast(
            "list[BaseMessage] | None", state.get("summary_pruned_messages")
        ):
            await self._ensure_service()
            self._pending_batches[runtime.context["session_id"]].append(
                PendingJargonAnalysisBatch(
                    session_id=runtime.context["session_id"],
                    messages=list(messages),
                )
            )
            await self._queue_jargon_job(runtime.context["session_id"])
        return None

    async def aclose(self) -> None:
        if self._send_stream is not None:
            await self._send_stream.aclose()
        if self._receive_stream is not None:
            await self._receive_stream.aclose()
        if self._task_group is not None:
            self._task_group.cancel_scope.cancel()
            await self._task_group.__aexit__(None, None, None)

        self._send_stream = None
        self._receive_stream = None
        self._task_group = None
        self._store = None
        self._pending_batches.clear()
        self._scheduled_sessions.clear()
        self._delayed_retry_sessions.clear()
        self._retry_attempts.clear()

    async def _ensure_service(self) -> None:
        if self._send_stream is not None:
            return

        async with self._start_lock:
            if self._send_stream is not None:
                return

            send_stream, receive_stream = anyio.create_memory_object_stream[str](
                self.queue_size
            )
            task_group = anyio.create_task_group()
            await task_group.__aenter__()
            task_group.start_soon(self._jargon_worker, receive_stream)

            self._send_stream = send_stream
            self._receive_stream = receive_stream
            self._task_group = task_group

    async def _queue_jargon_job(self, session_id: str) -> None:
        if (
            self._send_stream is None
            or self._task_group is None
            or session_id in self._scheduled_sessions
        ):
            return

        self._scheduled_sessions.add(session_id)
        try:
            self._send_stream.send_nowait(session_id)
        except anyio.WouldBlock:
            pass
        except (anyio.BrokenResourceError, anyio.ClosedResourceError):
            self._scheduled_sessions.discard(session_id)
            logger.exception("Failed to queue jargon job")
            return
        else:
            return

        try:
            self._task_group.start_soon(
                self._send_jargon_job,
                self._send_stream,
                session_id,
            )
        except RuntimeError:
            self._scheduled_sessions.discard(session_id)
            logger.exception("Failed to schedule jargon job")

    async def _send_jargon_job(
        self,
        send_stream: MemoryObjectSendStream[str],
        session_id: str,
    ) -> None:
        try:
            await send_stream.send(session_id)
        except (anyio.BrokenResourceError, anyio.ClosedResourceError):
            self._scheduled_sessions.discard(session_id)
            logger.exception("Failed to send jargon job")

    async def _jargon_worker(
        self,
        receive_stream: MemoryObjectReceiveStream[str],
    ) -> None:
        async with receive_stream:
            async for session_id in receive_stream:
                try:
                    progressed, failed = await self._analyze_session(session_id)
                except Exception:
                    logger.exception("Failed to analyze jargon batch")
                    progressed = False
                    failed = True
                finally:
                    self._scheduled_sessions.discard(session_id)

                if failed:
                    await self._schedule_retry(session_id)
                    continue

                self._retry_attempts.pop(session_id, None)
                if progressed and self._pending_batches.get(session_id):
                    await self._queue_jargon_job(session_id)

    async def _analyze_session(self, session_id: str) -> tuple[bool, bool]:
        if self._store is None:
            logger.warning(
                "Skipping jargon analysis for session %s because store is unavailable",
                session_id,
            )
            return False, False

        progressed = False
        failed = False
        while pending_batches := self._pending_batches.get(session_id):
            batch_count, session_id, messages_text, source_map = (
                self._build_extraction_input(pending_batches)
            )
            if not messages_text:
                for _ in range(min(batch_count, len(pending_batches))):
                    pending_batches.popleft()
                if not pending_batches:
                    self._pending_batches.pop(session_id, None)
                progressed = True
                continue

            entries = await self._extract_entries(messages_text, source_map)
            if entries is None:
                failed = True
                break

            await self._process_entries(
                session_id=session_id,
                entries=entries,
                store=self._store,
            )
            for _ in range(min(batch_count, len(pending_batches))):
                pending_batches.popleft()
            if not pending_batches:
                self._pending_batches.pop(session_id, None)
            progressed = True

        return progressed, failed

    def _build_extraction_input(
        self,
        pending_batches: deque[PendingJargonAnalysisBatch],
    ) -> tuple[int, str, str, dict[str, str]]:
        selected_lines: list[str] = []
        source_map: dict[str, str] = {}
        batch_count = 0
        session_id = ""

        for batch in list(pending_batches)[: self.jargon_context_batches]:
            session_id = batch.session_id
            batch_lines: list[str] = []
            batch_sources: dict[str, str] = {}

            for message in batch.messages:
                if isinstance(message, HumanMessage) and isinstance(
                    raw := message.additional_kwargs.get("raw"), UserMessage
                ):
                    batch_lines.append(
                        _truncate(raw.as_content(), self.max_message_chars)
                    )
                    plain = _clean_text(raw.as_plain_content())
                    if raw.message_id and plain:
                        batch_sources[raw.message_id] = _truncate(
                            plain,
                            self.max_raw_content_chars,
                        )
                    continue

                text = _clean_text(content_to_text(message.content))
                if not text:
                    continue

                if isinstance(message, AIMessage):
                    batch_lines.append(
                        _truncate(
                            f"<bot-message user={BOT_NAME}>\n{text}\n</bot-message>",
                            self.max_message_chars,
                        )
                    )
                else:
                    batch_lines.append(
                        _truncate(
                            f"<user-message>\n{text}\n</user-message>",
                            self.max_message_chars,
                        )
                    )

            if not batch_lines:
                batch_count += 1
                continue

            candidate = "\n".join([*selected_lines, *batch_lines]).strip()
            if selected_lines and len(candidate) > self.max_extract_input_chars:
                break

            selected_lines.extend(batch_lines)
            source_map.update(batch_sources)
            batch_count += 1

            if len(candidate) > self.max_extract_input_chars:
                return (
                    batch_count,
                    session_id,
                    _truncate(candidate, self.max_extract_input_chars),
                    source_map,
                )

        return batch_count, session_id, "\n".join(selected_lines).strip(), source_map

    async def _extract_entries(
        self,
        messages_text: str,
        source_map: dict[str, str],
    ) -> list[JargonEntry] | None:
        try:
            response = await self.analyze_model.ainvoke(
                JARGON_EXTRACTION_PROMPT.format(messages=messages_text),
                config={"metadata": {"lc_source": "jargon_extract"}},
            )
        except Exception:
            logger.exception("Failed to extract jargon entries")
            return None

        parsed = _parse_result(content_to_text(response.content))
        raw_entries = parsed.get("entries", []) if isinstance(parsed, dict) else parsed
        if not isinstance(raw_entries, list):
            logger.warning("Failed to parse jargon extraction result")
            return None

        merged_entries: dict[str, set[str]] = defaultdict(set)
        for item in raw_entries:
            if not isinstance(item, dict):
                continue
            content = _clean_text(item.get("content"))
            source_id = _clean_text(item.get("source_id"))
            if content and source_id and source_id in source_map:
                merged_entries[content].add(source_map[source_id])

        return [
            {"content": content, "raw_content": raw_content}
            for content, raw_content in merged_entries.items()
            if raw_content
        ]

    async def _process_entries(  # noqa: PLR0915
        self,
        *,
        session_id: str,
        entries: list[JargonEntry],
        store: BaseStore,
    ) -> tuple[int, int]:
        merged_entries: dict[str, set[str]] = defaultdict(set)
        for entry in entries:
            content = _clean_text(entry["content"])
            if not content or is_single_char_jargon(content):
                continue
            for raw_content in entry["raw_content"]:
                raw_content = _truncate(
                    _clean_text(raw_content), self.max_raw_content_chars
                )
                if raw_content:
                    merged_entries[content].add(raw_content)

        if not merged_entries:
            return 0, 0

        namespace_prefix = (self.namespace_root,)
        saved = 0
        updated = 0
        pending_inference: list[tuple[tuple[str, ...], str, JargonRecord]] = []
        all_items: list[SearchItem] = []
        offset = 0

        while True:
            items = await store.asearch(
                namespace_prefix,
                query=None,
                limit=100,
                offset=offset,
            )
            if not items:
                break
            all_items.extend(items)
            if len(items) < 100:
                break
            offset += 100

        for content, raw_content in merged_entries.items():
            namespace = (self.namespace_root, "sessions", session_id)
            key = content
            matched_item: SearchItem | None = None
            matched_record: JargonRecord | None = None
            matched_rank: tuple[int, int, int] | None = None

            for item in all_items:
                record = _normalize_record(item.value)
                if record is None or record["content"] != content:
                    continue
                rank = (
                    0
                    if session_id in record["session_id_dict"]
                    else 1
                    if record["is_global"]
                    else 2,
                    0 if record["created_by"] == JARGON_CREATED_BY_MANUAL else 1,
                    -record["count"],
                )
                if matched_rank is None or rank < matched_rank:
                    matched_item = item
                    matched_record = record
                    matched_rank = rank

            now = datetime.now(UTC).isoformat()

            if matched_record is None:
                record = cast(
                    "JargonRecord",
                    {
                        "content": content,
                        "raw_content": sorted(raw_content),
                        "session_id_dict": {session_id: 1},
                        "session_id": session_id,
                        "is_global": False,
                        "other_match_count": 0,
                        "count": 1,
                        "meaning": "",
                        "is_jargon": False,
                        "is_complete": False,
                        "last_inference_count": 0,
                        "created_by": JARGON_CREATED_BY_AI,
                        "created_at": now,
                        "updated_at": now,
                    },
                )
                saved += 1
            elif matched_item is not None:
                record = matched_record
                namespace = matched_item.namespace
                key = matched_item.key
                if record["created_by"] == JARGON_CREATED_BY_MANUAL:
                    continue
                record["count"] += 1
                record["raw_content"] = sorted(
                    set(record["raw_content"]).union(raw_content)
                )
                record["session_id_dict"][session_id] = (
                    record["session_id_dict"].get(session_id, 0) + 1
                )
                record["is_global"] = (
                    record["is_global"] or len(record["session_id_dict"]) > 1
                )
                record["updated_at"] = now
                updated += 1
            else:
                continue

            await store.aput(
                namespace,
                key,
                dict(record),
                index=["content", "meaning", "raw_content[*]"],
            )
            if self._should_infer_meaning(record):
                pending_inference.append((namespace, key, record))

        logger.info("[%s]疑似黑话: %s", session_id, ",".join(merged_entries))
        for namespace, key, record in pending_inference:
            await self._infer_meaning(namespace, key, record, store)

        return saved, updated

    async def _infer_meaning(
        self,
        namespace: tuple[str, ...],
        key: str,
        record: JargonRecord,
        store: BaseStore,
    ) -> None:
        if record["created_by"] == JARGON_CREATED_BY_MANUAL:
            return

        raw_content_list = [item for item in record["raw_content"] if _clean_text(item)]
        if not raw_content_list:
            logger.warning(
                "Jargon '%s' has no raw_content; skip inference", record["content"]
            )
            return

        raw_content_text = "\n".join(raw_content_list)
        if (
            record["count"] in JARGON_SAMPLE_RAW_CONTENT_THRESHOLDS
            and len(raw_content_list) > 1
        ):
            raw_content_list = random.sample(
                raw_content_list,
                max(1, len(raw_content_list) // 2),
            )

        previous_meaning_section = ""
        previous_meaning_instruction = ""
        if (
            record["count"] in JARGON_PREVIOUS_MEANING_THRESHOLDS
            and record["meaning"].strip()
        ):
            previous_meaning_section = (
                f"\n**上一次推断的含义（仅供参考）**\n{record['meaning']}"
            )
            previous_meaning_instruction = "- 请参考上一次推断的含义，结合新的上下文信息，给出更准确或更新的推断结果"

        async def ask(prompt: str, source: str) -> dict[str, Any] | None:
            try:
                response = await self.analyze_model.ainvoke(
                    prompt,
                    config={"metadata": {"lc_source": f"jargon_{source}"}},
                )
            except Exception:
                logger.exception("Failed jargon inference call: %s", source)
                return None
            return _parse_inference_result(content_to_text(response.content))

        inference1 = await ask(
            JARGON_INFERENCE_WITH_CONTEXT_PROMPT.format(
                bot_name=BOT_NAME,
                content=record["content"],
                raw_content_list=raw_content_text,
                previous_meaning_section=previous_meaning_section,
                previous_meaning_instruction=previous_meaning_instruction,
            ),
            "inference_with_context",
        )
        if not inference1:
            return

        if inference1.get("no_info") or not _clean_text(inference1.get("meaning")):
            record["last_inference_count"] = record["count"]
            record["updated_at"] = datetime.now(UTC).isoformat()
            await store.aput(
                namespace,
                key,
                dict(record),
                index=["content", "meaning", "raw_content[*]"],
            )
            return

        inference2 = await ask(
            JARGON_INFERENCE_CONTENT_ONLY_PROMPT.format(content=record["content"]),
            "inference_content_only",
        )
        if not inference2:
            return

        comparison = await ask(
            JARGON_COMPARE_INFERENCE_PROMPT.format(
                inference1=json.dumps(inference1, ensure_ascii=False),
                inference2=json.dumps(inference2, ensure_ascii=False),
            ),
            "inference_compare",
        )
        if not comparison:
            return

        record["is_jargon"] = not bool(comparison.get("is_similar"))
        record["meaning"] = (
            _clean_text(inference1.get("meaning")) if record["is_jargon"] else ""
        )
        record["last_inference_count"] = record["count"]
        record["is_complete"] = record["count"] >= JARGON_INFERENCE_THRESHOLDS[-1]
        record["updated_at"] = datetime.now(UTC).isoformat()
        await store.aput(
            namespace,
            key,
            dict(record),
            index=["content", "meaning", "raw_content[*]"],
        )

        if record["is_jargon"]:
            logger.info(
                "[黑话]%s的含义是 %s",
                record["content"],
                record["meaning"] or "无详细说明",
            )
        else:
            logger.info("[%s]%s 不是黑话", record["session_id"], record["content"])

    async def _query_jargon(
        self,
        runtime: ToolRuntime[ManagerContext, ManagerState],
        keyword: str,
        limit: int = 10,
        case_sensitive: bool = False,
        fuzzy: bool = True,
    ) -> list[dict[str, str]]:
        keyword = _clean_text(keyword)
        store = runtime.store or self._store
        if not keyword or store is None:
            return []

        namespace = (self.namespace_root,)
        session_id = runtime.context["session_id"]
        keyword_cmp = keyword if case_sensitive else keyword.lower()
        results: list[tuple[int, SearchItem, JargonRecord]] = []
        offset = 0
        while True:
            items = await store.asearch(namespace, query=None, limit=100, offset=offset)
            if not items:
                break

            for item in items:
                record = _normalize_record(item.value)
                if record is None or not record["meaning"].strip():
                    continue
                content_cmp = (
                    record["content"] if case_sensitive else record["content"].lower()
                )
                if (fuzzy and keyword_cmp in content_cmp) or (
                    not fuzzy and keyword_cmp == content_cmp
                ):
                    scope = (
                        0
                        if session_id in record["session_id_dict"]
                        else 1
                        if record["is_global"]
                        else 2
                    )
                    if scope == 2 and content_cmp == keyword_cmp:
                        record["other_match_count"] += 1
                        threshold = self.to_global_threshold
                        if (
                            threshold is not None
                            and record["other_match_count"] >= threshold
                        ):
                            record["is_global"] = True
                        record["updated_at"] = datetime.now(UTC).isoformat()
                        await store.aput(
                            item.namespace,
                            item.key,
                            dict(record),
                            index=["content", "meaning", "raw_content[*]"],
                        )
                        if record["is_global"]:
                            scope = 1
                    results.append((scope, item, record))

            if len(items) < 100:
                break
            offset += 100

        deduped: dict[str, tuple[int, SearchItem, JargonRecord]] = {}
        for scope, item, record in results:
            current = deduped.get(record["content"])
            rank = (
                scope,
                0 if record["created_by"] == JARGON_CREATED_BY_MANUAL else 1,
                -record["count"],
            )
            if current is None:
                deduped[record["content"]] = (scope, item, record)
                continue
            current_scope, _, current_record = current
            current_rank = (
                current_scope,
                0 if current_record["created_by"] == JARGON_CREATED_BY_MANUAL else 1,
                -current_record["count"],
            )
            if rank < current_rank:
                deduped[record["content"]] = (scope, item, record)

        sorted_results = sorted(
            deduped.values(),
            key=lambda item: (
                item[0],
                item[2]["created_by"] != JARGON_CREATED_BY_MANUAL,
                -item[2]["count"],
            ),
        )
        response: list[dict[str, str]] = []
        for scope, _, record in sorted_results[:limit]:
            meaning = record["meaning"]
            if scope == 2:
                meaning = f"[非当前会话] {meaning}"
            response.append({"content": record["content"], "meaning": meaning})
        return response

    async def _schedule_retry(self, session_id: str) -> None:
        if not self._pending_batches.get(session_id):
            self._retry_attempts.pop(session_id, None)
            return

        attempt = self._retry_attempts.get(session_id, 0) + 1
        if attempt > self.retry_max_attempts:
            logger.error(
                "Jargon retries exhausted for %s; pending batches=%s",
                session_id,
                len(self._pending_batches.get(session_id, ())),
            )
            return

        self._retry_attempts[session_id] = attempt
        if session_id in self._delayed_retry_sessions or self._task_group is None:
            return

        self._delayed_retry_sessions.add(session_id)
        self._task_group.start_soon(
            self._retry_jargon_job_after_delay,
            session_id,
            self.retry_backoff_seconds * attempt,
        )

    async def _retry_jargon_job_after_delay(
        self,
        session_id: str,
        delay_seconds: float,
    ) -> None:
        try:
            await anyio.sleep(delay_seconds)
        finally:
            self._delayed_retry_sessions.discard(session_id)

        if not self._pending_batches.get(session_id):
            self._retry_attempts.pop(session_id, None)
            return
        await self._queue_jargon_job(session_id)

    @staticmethod
    def _should_infer_meaning(record: JargonRecord) -> bool:
        if (
            record["created_by"] == JARGON_CREATED_BY_MANUAL
            or record["is_complete"]
            or record["count"] < JARGON_INFERENCE_THRESHOLDS[0]
            or record["count"] <= record["last_inference_count"]
        ):
            return False

        next_threshold = next(
            (
                threshold
                for threshold in JARGON_INFERENCE_THRESHOLDS
                if threshold > record["last_inference_count"]
            ),
            None,
        )
        return next_threshold is not None and record["count"] >= next_threshold


def _normalize_record(value: Any) -> JargonRecord | None:
    if not isinstance(value, dict):
        return None

    content = _clean_text(value.get("content"))
    if not content:
        return None

    raw_content = value.get("raw_content") or []
    if isinstance(raw_content, str):
        try:
            raw_content = json.loads(raw_content)
        except json.JSONDecodeError:
            raw_content = [raw_content]

    session_id_dict = value.get("session_id_dict") or {}
    if isinstance(session_id_dict, str):
        try:
            session_id_dict = json.loads(session_id_dict)
        except json.JSONDecodeError:
            session_id_dict = {}

    return {
        "content": content,
        "raw_content": sorted(
            {_clean_text(item) for item in raw_content if _clean_text(item)}
        ),
        "session_id_dict": {
            str(key): int(raw or 0) for key, raw in dict(session_id_dict).items()
        },
        "session_id": str(value.get("session_id") or ""),
        "is_global": bool(value.get("is_global", False))
        or len(dict(session_id_dict)) > 1,
        "other_match_count": int(value.get("other_match_count") or 0),
        "count": int(value.get("count") or 0),
        "meaning": _clean_text(value.get("meaning")),
        "is_jargon": bool(value.get("is_jargon", False)),
        "is_complete": bool(value.get("is_complete", False)),
        "last_inference_count": int(value.get("last_inference_count") or 0),
        "created_by": cast(
            "Literal['ai', 'manual']",
            str(value.get("created_by") or JARGON_CREATED_BY_AI).lower(),
        ),
        "created_at": str(value.get("created_at") or ""),
        "updated_at": str(value.get("updated_at") or ""),
    }


def _parse_result(text: str) -> Any | None:
    raw = text.strip()
    if match := re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL | re.IGNORECASE):
        raw = match[1].strip()
    else:
        raw = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    candidates = [raw]
    for left, right in (("{", "}"), ("[", "]")):
        start = raw.find(left)
        end = raw.rfind(right)
        if start != -1 and end != -1 and start < end:
            candidate = raw[start : end + 1].strip()
            if candidate not in candidates:
                candidates.append(candidate)

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            logger.debug("Failed to parse jargon JSON candidate directly")

        try:
            repaired = repair_json(candidate)
            if isinstance(repaired, tuple):
                repaired = repaired[0]
            if not isinstance(repaired, str):
                repaired = json.dumps(repaired, ensure_ascii=False)
            return json.loads(repaired)
        except Exception:
            logger.debug("Failed to repair jargon JSON candidate", exc_info=True)

    return None


def _parse_inference_result(text: str) -> dict[str, Any] | None:
    raw = text.strip()
    try:
        result = json.loads(raw)
    except Exception:
        try:
            repaired = repair_json(raw)
            if isinstance(repaired, tuple):
                repaired = repaired[0]
            if not isinstance(repaired, str):
                repaired = json.dumps(repaired, ensure_ascii=False)
            result = json.loads(repaired)
        except Exception:
            logger.exception("Failed to parse jargon inference result")
            return None

    if not isinstance(result, dict):
        logger.warning("Jargon inference result is not a JSON object")
        return None
    return result


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def is_single_char_jargon(content: str) -> bool:
    return len(content) == 1 and (
        "\u4e00" <= content <= "\u9fff"
        or "a" <= content <= "z"
        or "A" <= content <= "Z"
        or "0" <= content <= "9"
    )


__all__ = ["JargonMiddleware"]
