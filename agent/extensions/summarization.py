"""会话摘要中间件。

该中间件不负责主动裁剪消息，而是消费上游已经剪掉的历史消息，
在后台异步生成会话摘要，并在后续模型调用前把摘要重新注入上下文。
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import anyio
from cachetools import LRUCache
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.messages.utils import get_buffer_string

from agent.base import ManagerContext, ManagerState
from agent.utils import content_to_text

if TYPE_CHECKING:
    from anyio.abc import TaskGroup
    from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
    from langchain.agents.middleware import ModelRequest
    from langchain_core.language_models.chat_models import BaseChatModel
    from langgraph.runtime import Runtime
    from langgraph.store.base import BaseStore

logger = logging.getLogger(__name__)


DEFAULT_SUMMARY_PROMPT = """<role>
Context Extraction Assistant
</role>

<primary_objective>
Your sole objective in this task is to extract the highest quality/most relevant context from the pruned conversation history below.
</primary_objective>

<objective_information>
The pruned conversation history below has already been removed from the live context window.
The context you extract in this step will overwrite the existing session summary and be injected back into future model calls.
Because of this, ensure the context you extract is only the most important information needed to continue the conversation effectively.
</objective_information>

<instructions>
The pruned conversation history below will be replaced with the context you extract in this step.
You want to ensure that future replies do not lose important progress or repeat work, so the context you extract should focus only on the most important information.

You should structure your summary using the following sections. Each section acts as a checklist. You must populate it with relevant information or explicitly state "None" if there is nothing to report for that section:

## SESSION INTENT
What is the user's primary ongoing goal, request, or discussion thread? This should be concise but complete enough to understand the purpose of the session.

## SUMMARY
Extract and record the most important durable context from the pruned messages. Include important choices, conclusions, strategies, constraints, and facts worth preserving. Keep only information that improves future replies.

## ARTIFACTS
What files, resources, tool results, external facts, or references were created, modified, accessed, or learned in these messages? If none, say "None."

## NEXT STEPS
What unresolved questions, pending tasks, likely follow-ups, or commitments should the agent remember?

</instructions>

Carefully read both the current session summary and the newly pruned messages, then produce a new session summary that fully replaces the previous one.
Do not keep filler, repeated wording, transient jokes, raw tool noise, or formatting noise.
Do not invent facts that are not supported by the messages.
Respond ONLY with the extracted context. Do not include any additional information, or text before or after the extracted context.

<existing_summary>
Current session summary:
{current_summary}
</existing_summary>

<messages>
Messages to summarize:
{messages}
</messages>"""  # noqa: E501


SESSION_SUMMARY_INJECTION_PROMPT = """
Here is a summary of the conversation to date:

<system-summary>
{summary}
</system-summary>
""".strip()


DEFAULT_NAMESPACE_ROOT = "session_memory"

# 默认的空摘要文本，确保 summary 字段始终有内容，避免模型调用时注入完全空的 system message。
_DEFAULT_EMPTY_SUMMARY = "None."
_SUMMARY_SOURCE = "session_summary"
_DEFAULT_RETRY_BACKOFF = 5.0


class SummarizationMiddleware(AgentMiddleware[ManagerState, ManagerContext]):
    """为会话提供异步摘要能力。

    整体流程分成三段：

    1. `aafter_agent` 接收由上游裁剪出的历史消息，并按 session 入队。
    2. 后台 worker 串行处理同一 session 的待总结批次，生成并持久化摘要。
    3. `awrap_model_call` 在主模型调用前把摘要作为额外的 system message 注入。

    设计上它只关心“已经被裁掉的旧消息”，不会干预上游如何裁剪，也不会
    主动改写当前轮的 live messages。
    """

    state_schema = ManagerState

    def __init__(
        self,
        summary_model: BaseChatModel,
        *,
        max_sessions: int = 100,
        max_input_chars: int = 24000,
        max_output_chars: int = 3000,
        max_batches: int = 3,
        max_retries: int = 3,
        summary_prompt: str = DEFAULT_SUMMARY_PROMPT,
        store: BaseStore | None = None,
        namespace_root: str = DEFAULT_NAMESPACE_ROOT,
    ) -> None:
        """初始化摘要中间件。

        Args:
            summary_model: 专门用于生成摘要的模型。
            max_sessions: 整个系统最多同时持有的 session 数量（LRU 驱逐）。
            max_input_chars: 单次摘要输入的最大字符数。
            max_output_chars: 最终摘要文本的最大字符数。
            max_batches: 单次摘要时最多合并多少个待处理批次。
            max_retries: 单个 session 摘要失败后的最大重试次数。
            summary_prompt: 摘要模型使用的提示词模板。
            store: 可选持久化存储；为空时只保存在进程内。
            namespace_root: 持久化存储命名空间的根路径。
        """
        super().__init__()
        self.summary_model = summary_model
        self.max_input_chars = max_input_chars
        self.max_output_chars = max_output_chars
        self.max_batches = max_batches
        self.max_retries = max_retries
        self.summary_prompt = summary_prompt

        self._store = store
        self._queue_size = max(64, max_sessions * 2)
        self._max_batches_per_session = max(16, max_sessions * 2)
        self._max_message_chars = max(1, max_input_chars // max_batches)
        self.namespace = (
            str(namespace_root).strip().replace(".", "_") or DEFAULT_NAMESPACE_ROOT,
        )

        self._pending_batches: LRUCache[str, deque[list[BaseMessage]]] = LRUCache(
            maxsize=max_sessions
        )
        self._session_summaries: LRUCache[str, str] = LRUCache(maxsize=max_sessions)
        self._loaded_sessions: LRUCache[str, bool] = LRUCache(maxsize=max_sessions)
        self._retry_attempts: LRUCache[str, int] = LRUCache(maxsize=max_sessions)

        self._scheduled_sessions: set[str] = set()
        self._delayed_retry_sessions: set[str] = set()

        self._start_lock = anyio.Lock()
        self._task_group: TaskGroup | None = None
        self._send_stream: MemoryObjectSendStream[str] | None = None
        self._receive_stream: MemoryObjectReceiveStream[str] | None = None

    async def abefore_agent(
        self,
        state: ManagerState,
        runtime: Runtime[ManagerContext],
    ) -> dict[str, Any] | None:
        """在 agent 启动前预热当前 session 的摘要缓存。"""
        _ = state
        if runtime.store is not None:
            self._store = self._store or runtime.store
        await self._ensure_session_summary_loaded(runtime.context["session_id"])
        return None

    async def aafter_agent(
        self,
        state: ManagerState,
        runtime: Runtime[ManagerContext],
    ) -> dict[str, Any] | None:
        """接收本轮被上游裁掉的历史消息，并把它们投递到后台摘要队列。"""
        if runtime.store is not None:
            self._store = self._store or runtime.store
        if messages := cast(
            "list[BaseMessage] | None", state.get("summary_pruned_messages")
        ):
            await self._ensure_service()
            session_id = runtime.context["session_id"]
            if session_id not in self._pending_batches:
                self._pending_batches[session_id] = deque(
                    maxlen=self._max_batches_per_session
                )
            self._pending_batches[session_id].append(list(messages))
            await self._queue_job(session_id)
        return None

    async def awrap_model_call(
        self,
        request: ModelRequest[ManagerContext],
        handler,
    ) -> Any:
        """在主模型调用前注入 session summary。

        这里刻意不去改写主 system prompt，而是额外插入一条 system message，
        让摘要成为独立的会话背景层，便于和主提示词职责分离。
        """
        if request.runtime.store is not None:
            self._store = self._store or request.runtime.store
        session_id = request.runtime.context["session_id"]
        await self._ensure_session_summary_loaded(session_id)
        if not (summary := self._session_summaries.get(session_id)):
            return await handler(request)

        return await handler(
            request.override(
                messages=[
                    SystemMessage(
                        content=SESSION_SUMMARY_INJECTION_PROMPT.format(
                            summary=summary
                        ),
                        additional_kwargs={"lc_source": _SUMMARY_SOURCE},
                    ),
                    *request.messages,
                ]
            )
        )

    async def aclose(self) -> None:
        """关闭后台服务并清空进程内缓存。"""
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
        self._loaded_sessions.clear()
        self._scheduled_sessions.clear()
        self._delayed_retry_sessions.clear()
        self._pending_batches.clear()
        self._session_summaries.clear()
        self._retry_attempts.clear()

    async def _ensure_service(self) -> None:
        """确保后台摘要 worker 已启动。

        该服务是惰性启动的：只有第一次真的收到 pruned messages 时才创建。
        """
        if self._send_stream is not None:
            return

        async with self._start_lock:
            if self._send_stream is not None:
                return

            send_stream, receive_stream = anyio.create_memory_object_stream[str](
                self._queue_size
            )
            task_group = anyio.create_task_group()
            await task_group.__aenter__()
            task_group.start_soon(self._save_worker, receive_stream)

            self._send_stream = send_stream
            self._receive_stream = receive_stream
            self._task_group = task_group

    async def _save_worker(
        self,
        receive_stream: MemoryObjectReceiveStream[str],
    ) -> None:
        """异常隔离与自动重启包装器。

        当 _worker 因任何未捕获异常退出时，自动等待 1s 后重启，
        直到 aclose() 将 receive_stream 置空。
        """
        async with receive_stream:
            while self._receive_stream is not None:
                try:
                    await self._worker(receive_stream)
                except Exception:
                    if self._receive_stream is None:
                        break
                    logger.exception("Summary worker crashed, restarting in 1s")
                    await anyio.sleep(1.0)

    async def _queue_job(self, session_id: str) -> None:
        """为指定 session 投递一次摘要任务。

        `_scheduled_sessions` 用于去重，保证同一 session 在后台最多只挂一个
        待执行任务；真正的批次合并在 worker 内部完成。
        """
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
            logger.exception("Failed to queue session summary job")
            return
        else:
            return

        async def _send_job(
            send_stream: MemoryObjectSendStream[str],
        ) -> None:
            try:
                await send_stream.send(session_id)
            except (anyio.BrokenResourceError, anyio.ClosedResourceError):
                self._scheduled_sessions.discard(session_id)
                logger.exception("Failed to send session summary job")

        try:
            self._task_group.start_soon(
                _send_job,
                self._send_stream,
            )
        except RuntimeError:
            self._scheduled_sessions.discard(session_id)
            logger.exception("Failed to schedule session summary job")

    async def _worker(
        self,
        receive_stream: MemoryObjectReceiveStream[str],
    ) -> None:
        """后台摘要 worker。

        它按 session 维度串行推进摘要，失败时触发延迟重试，成功时继续消费
        同一 session 剩余的待处理批次。
        """
        async for session_id in receive_stream:
            try:
                progressed, failed = await self._summarize_session(session_id)
            except Exception:
                logger.exception("Failed to update session summary")
                progressed = False
                failed = True
            finally:
                self._scheduled_sessions.discard(session_id)

            if failed:
                await self._schedule_retry(session_id)
                continue

            self._retry_attempts.pop(session_id, None)
            if progressed and self._pending_batches.get(session_id):
                await self._queue_job(session_id)

    async def _summarize_session(self, session_id: str) -> tuple[bool, bool]:
        """尽可能推进某个 session 的摘要状态。

        Returns:
            `(progressed, failed)`：
            - `progressed=True` 表示至少消费掉了一部分待处理批次。
            - `failed=True` 表示本轮处理中遇到了需要后续重试的错误。
        """
        progressed = False
        failed = False

        while pending_batches := self._pending_batches.get(session_id):
            batch_count, messages_text = self._build_summary_input(pending_batches)
            if batch_count <= 0:
                break

            if not messages_text:
                for _ in range(min(batch_count, len(pending_batches))):
                    pending_batches.popleft()
                if not pending_batches:
                    self._pending_batches.pop(session_id, None)
                progressed = True
                continue

            summary = await self._create_summary(
                current_summary=self._session_summaries.get(
                    session_id,
                    _DEFAULT_EMPTY_SUMMARY,
                ),
                messages_text=messages_text,
            )
            if summary is None or not await self._save_session_summary(
                session_id, summary
            ):
                failed = True
                break

            for _ in range(min(batch_count, len(pending_batches))):
                pending_batches.popleft()
            if not pending_batches:
                self._pending_batches.pop(session_id, None)
            progressed = True

        return progressed, failed

    async def _create_summary(
        self,
        *,
        current_summary: str,
        messages_text: str,
    ) -> str | None:
        """调用摘要模型生成新摘要。

        这里采用“当前摘要 + 新剪枝消息”的增量覆盖模式，每次生成的结果都
        会完整替换旧摘要。
        """
        try:
            response = await self.summary_model.ainvoke(
                self.summary_prompt.format(
                    current_summary=current_summary,
                    messages=messages_text,
                ).rstrip(),
                config={"metadata": {"lc_source": _SUMMARY_SOURCE}},
            )
        except Exception:
            logger.exception("Failed to generate session summary")
            return None

        summary = content_to_text(response.content).strip()
        if not summary:
            logger.warning("Summary model returned an empty session summary")
            return None
        return summary

    def _build_summary_input(
        self,
        pending_batches: deque[list[BaseMessage]],
    ) -> tuple[int, str]:
        """从待处理批次中拼出一次摘要调用的输入。

        策略是按顺序合并前若干个批次：
        - 单条消息先做字符级截断，避免异常长消息污染输入。
        - 一旦累计文本超过 `max_summary_input_chars`，就停止继续扩张。
        - 如果超长发生在第一批，则保留该批并整体截断，确保任务还能推进。
        """
        selected_messages: list[BaseMessage] = []
        batch_count = 0

        for batch in list(pending_batches)[: self.max_batches]:
            batch_messages = [
                message.model_copy(
                    update={"content": _truncate(text, self._max_message_chars)}
                )
                for message in batch
                if (text := content_to_text(message.content).strip())
            ]
            if not batch_messages:
                batch_count += 1
                continue

            candidate_text = get_buffer_string(
                [*selected_messages, *batch_messages]
            ).strip()
            if selected_messages and len(candidate_text) > self.max_input_chars:
                break

            selected_messages.extend(batch_messages)
            batch_count += 1

            if len(candidate_text) > self.max_input_chars:
                return batch_count, _truncate(candidate_text, self.max_input_chars)

        return (
            batch_count,
            get_buffer_string(selected_messages).strip() if selected_messages else "",
        )

    async def _ensure_session_summary_loaded(self, session_id: str) -> None:
        """把指定 session 的摘要从 store 懒加载到进程内缓存。"""
        if (
            session_id in self._loaded_sessions
            and session_id in self._session_summaries
        ):
            return
        if session_id in self._loaded_sessions:
            del self._loaded_sessions[session_id]
        if self._store is None:
            self._loaded_sessions[session_id] = True
            return

        try:
            item = await self._store.aget(self.namespace, session_id)
        except Exception:
            logger.exception("Failed to load session summary from store")
            return

        self._loaded_sessions[session_id] = True
        if item is None:
            return

        summary = str(item.value.get("summary") or "").strip()
        if summary:
            self._session_summaries[session_id] = _truncate(
                summary,
                self.max_output_chars,
            )

    async def _save_session_summary(self, session_id: str, summary: str) -> bool:
        """保存摘要到缓存，并在有 store 时写入持久层。"""
        summary = _truncate(summary.strip(), self.max_output_chars)
        if not summary:
            return False

        if self._store is None:
            self._session_summaries[session_id] = summary
            self._loaded_sessions[session_id] = True
            return True

        created_at = datetime.now(UTC).isoformat()
        try:
            item = await self._store.aget(self.namespace, session_id)
            if item is not None and isinstance(item.value.get("created_at"), str):
                created_at = item.value["created_at"]

            await self._store.aput(
                self.namespace,
                session_id,
                {
                    "id": session_id,
                    "session_id": session_id,
                    "summary": summary,
                    "created_at": created_at,
                    "updated_at": datetime.now(UTC).isoformat(),
                },
                index=False,
            )
        except Exception:
            logger.exception("Failed to persist session summary")
            return False

        self._session_summaries[session_id] = summary
        self._loaded_sessions[session_id] = True
        return True

    async def _schedule_retry(self, session_id: str) -> None:
        """为失败的 session 安排一次延迟重试。

        同一个 session 在等待重试期间只允许存在一个延迟任务，避免错误高峰时
        产生无意义的重试风暴。
        """
        if not self._pending_batches.get(session_id):
            self._retry_attempts.pop(session_id, None)
            return

        attempt = self._retry_attempts.get(session_id, 0) + 1
        if attempt > self.max_retries:
            logger.error(
                "Session summary retries exhausted for %s; pending batches=%s",
                session_id,
                len(self._pending_batches.get(session_id, ())),
            )
            return

        self._retry_attempts[session_id] = attempt
        if session_id in self._delayed_retry_sessions or self._task_group is None:
            return

        self._delayed_retry_sessions.add(session_id)
        self._task_group.start_soon(
            self._retry_summary_job_after_delay,
            session_id,
            _DEFAULT_RETRY_BACKOFF * attempt,
        )

    async def _retry_summary_job_after_delay(
        self,
        session_id: str,
        delay_seconds: float,
    ) -> None:
        """等待一段时间后，把失败 session 重新投回正常队列。"""
        try:
            await anyio.sleep(delay_seconds)
        finally:
            self._delayed_retry_sessions.discard(session_id)

        if not self._pending_batches.get(session_id):
            self._retry_attempts.pop(session_id, None)
            return

        await self._queue_job(session_id)


def _truncate(text: str, limit: int) -> str:
    """按字符数截断文本，并尽量保留省略号语义。"""
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..." if limit > 3 else text[:limit]


__all__ = ["SummarizationMiddleware"]
