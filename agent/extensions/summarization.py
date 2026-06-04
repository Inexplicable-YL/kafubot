"""会话摘要中间件。

该中间件不负责主动裁剪消息，而是消费上游已经剪掉的历史消息，在后台异步生成
会话摘要，并在后续模型调用前把摘要重新注入上下文。

整体流程分为三段：

1. `abefore_agent` 在主代理开始执行前，预热当前会话的摘要缓存。
2. `aafter_agent` 接收上游已经裁剪掉的历史消息，并按 `session_id` 投递到后台队列。
3. 后台 worker 串行更新摘要，`awrap_model_call` 再把最新摘要注入后续模型调用。

与“把所有历史消息一直保留在 live context”相比，这种设计的核心价值在于：

1. 历史上下文可以被压缩成稳定、低成本的会话记忆层。
2. 摘要生成发生在主对话链路之外，不直接增加当前轮回复延迟。
3. 摘要内容可以持续覆盖更新，避免旧消息无限累积。

该模块的职责边界也比较明确：它只关心“已经被上游裁剪掉的消息如何沉淀成摘要”，
不决定何时裁剪、裁剪多少，也不改写当前轮正在参与推理的 live messages。
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import anyio
from cachetools import LRUCache
from langchain.agents.middleware import AgentMiddleware
from langchain.messages import HumanMessage
from langchain_core.messages.utils import get_buffer_string

from agent.base import ManagerContext, ManagerState
from agent.utils import content_to_text

if TYPE_CHECKING:
    from anyio.abc import TaskGroup
    from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
    from langchain.agents.middleware import ModelRequest
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import BaseMessage
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
_DEFAULT_EMPTY_SUMMARY = "None."
_SUMMARY_SOURCE = "session_summary"
_DEFAULT_RETRY_BACKOFF = 5.0


class SummarizationMiddleware(AgentMiddleware[ManagerState, ManagerContext]):
    """为代理提供异步会话摘要能力。

    这个中间件的核心设计思想，是把“长期上下文压缩”作为一个独立的后台子系统来
    维护，而不是在主调用链里即时总结。这样既能持续保留会话进展，又能避免把大段
    历史消息反复塞回 live context。

    内部流程可以概括为三个层次：

    1. 收集层：`aafter_agent` 接收被上游裁剪掉的历史消息，按会话写入待处理队列。
    2. 工作层：后台 worker 串行处理同一会话的批次，生成并持久化最新摘要。
    3. 注入层：`awrap_model_call` 在主模型调用前读取缓存，把摘要作为额外 system
       message 注入。

    与普通缓存不同，这里的摘要缓存既承担性能职责，也承担语义职责：

    1. 性能上，它避免每一轮都从 store 反复读取同一摘要。
    2. 语义上，它代表“当前会话已压缩沉淀出的最重要上下文”。

    Attributes:
        namespace: 当前中间件写入 store 时使用的命名空间。
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
            summary_model: 专门用于生成会话摘要的模型。
            max_sessions: 进程内最多同时维护多少个活跃会话的队列、缓存和重试状态。
                超出后由 `LRUCache` 自动驱逐较久未访问的会话状态。
            max_input_chars: 单次摘要模型调用允许输入的最大字符数。
            max_output_chars: 单条摘要最终允许保留的最大字符数。
            max_batches: 单次后台处理时，最多合并多少个待处理批次。
            max_retries: 同一会话摘要任务失败后的最大自动重试次数。
            summary_prompt: 用于生成摘要的提示词模板。调用方可以覆盖默认模板，但
                仍应保持“基于旧摘要与新消息，产出完整新摘要”的语义。
            store: 可选的持久化存储。若未显式传入，会在运行时优先使用
                `runtime.store`。
            namespace_root: 持久化命名空间根名称。若包含点号，会被转换成下划线，
                以避免不同 store 后端在命名空间解析上的差异。
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
        """在主代理开始执行前预热当前会话的摘要缓存。

        该钩子的目标很直接：尽量在真正进入模型调用前，就把当前会话的摘要从
        store 惰性加载到进程内缓存。这样一来，`awrap_model_call` 通常只需要做一
        次字典读取，而不必在主调用链上临时访问外部存储。

        Args:
            state: 当前代理状态。这里不直接读取其中内容，但保留参数以满足中间件
                接口约定。
            runtime: 运行时上下文，用于获取 `session_id` 和可用的 store。

        Returns:
            始终返回 `None`，因为该钩子只做缓存预热，不改写状态。
        """
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
        """接收本轮被裁剪掉的历史消息，并把它们投递到后台摘要队列。

        该钩子只关心 `summary_pruned_messages`。这意味着：

        1. 当前还留在 live context 中的消息不会被重复总结。
        2. 摘要系统天然与“消息裁剪策略”解耦，只消费其产物。
        3. 摘要生成延迟到主代理执行之后，不阻塞当前轮响应。

        Args:
            state: 当前代理状态。这里会读取 `summary_pruned_messages`。
            runtime: 运行时上下文，用于获取 `session_id` 和 store。

        Returns:
            始终返回 `None`，因为该钩子只负责排队副作用。
        """
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
        """在主模型调用前注入当前会话摘要。

        这里刻意不去修改原始 system prompt，而是额外插入一条新的 `SystemMessage`。
        这样做的好处是：

        1. 摘要作为独立上下文层存在，更容易与角色提示分离。
        2. 上游若需要观察或调试注入内容，可以明确识别 `_SUMMARY_SOURCE`。
        3. 未来如果调整摘要注入策略，不必侵入主提示词模板。

        Args:
            request: 当前模型调用请求。
            handler: 下一个处理器。

        Returns:
            下游处理器返回的模型调用结果。
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
                    HumanMessage(
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
        """关闭后台摘要服务并清空进程内状态。

        该方法只会清理当前进程持有的资源与缓存，不删除已经写入外部 store 的持久化
        摘要数据。
        """
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

        该后台服务采用惰性启动策略：只有真正收到待总结消息时才创建内存流与
        `TaskGroup`。这可以避免在完全不需要摘要能力的会话中平白持有后台资源。
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
        """作为外层监护循环运行摘要 worker。

        真正的业务处理由 `_worker` 完成；这里的职责是隔离未捕获异常，并在 worker
        崩溃后自动等待一小段时间再重启，避免偶发错误直接让整个摘要子系统永久停摆。

        Args:
            receive_stream: 接收待处理 `session_id` 的内存流。
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
        """为指定会话投递一次摘要任务。

        `_scheduled_sessions` 用于去重，保证同一 `session_id` 在任意时刻最多只存在
        一个“已进入调度体系但尚未完成”的后台任务。至于这个任务内部会合并多少个批
        次，则由 `_worker` 和 `_build_summary_input` 决定。

        Args:
            session_id: 需要安排摘要处理的会话 ID。
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
        """后台摘要 worker 主循环。

        它按会话维度串行推进摘要更新：某个会话失败时进入延迟重试，成功时若该会话
        仍然残留待处理批次，则继续重新排队，直到当前积压被尽可能消费完。

        Args:
            receive_stream: 从 `_queue_job` 投递过来的 `session_id` 流。
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
        """尽可能推进某个会话的摘要状态。

        该方法会反复尝试消费当前会话积压的待处理批次。每一轮都执行：

        1. 取若干批次并拼成一次摘要输入。
        2. 调用摘要模型生成新的完整摘要。
        3. 将结果写回缓存和 store。
        4. 成功后从队列中移除对应批次。

        Args:
            session_id: 需要推进摘要状态的会话 ID。

        Returns:
            一个二元组 `(progressed, failed)`：
            `progressed=True` 表示至少消费掉了一个批次或空批次；
            `failed=True` 表示本轮遇到了需要进入重试流程的错误。
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
        """调用摘要模型生成新的完整摘要。

        这里采用“旧摘要 + 新裁剪消息 -> 新摘要”的覆盖式更新模式。也就是说，模型
        每次输出的都应当是可以直接替换旧摘要的完整结果，而不是一段增量补丁。

        Args:
            current_summary: 当前会话已存在的摘要文本。
            messages_text: 本轮新纳入总结范围的历史消息文本。

        Returns:
            生成成功时返回新的摘要文本；若模型调用失败或返回空结果，则返回 `None`。
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
        """从待处理批次中构建一次摘要模型调用的输入。

        输入拼装策略与后台摘要的吞吐和成本直接相关，因此这里做了三层约束：

        1. 单条消息先按 `_max_message_chars` 截断，防止极长消息污染整批输入。
        2. 单次最多合并 `max_batches` 个批次，限制单轮处理时延。
        3. 总文本超过 `max_input_chars` 后停止扩展；若超限发生在首批，则保留首批并
           做整体截断，确保任务仍然能够继续前进，而不是被一条超长历史永久卡住。

        Args:
            pending_batches: 当前会话积压的待处理批次队列。

        Returns:
            一个二元组 `(batch_count, messages_text)`，其中 `batch_count` 表示本轮纳入
            的批次数，`messages_text` 是格式化后的摘要输入文本。
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
        """把指定会话的摘要从 store 惰性加载到进程内缓存。

        `_loaded_sessions` 的语义不是“该会话一定有摘要”，而是“该会话已经尝试过装载”。
        这样可以区分：

        1. 还没查过 store。
        2. 查过，但确实没有摘要。
        3. 查过，并成功装入了摘要内容。

        Args:
            session_id: 需要装载摘要的会话 ID。
        """
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
        """保存摘要到进程缓存，并在可用时写入持久化 store。

        该方法会统一负责：

        1. 按 `max_output_chars` 截断最终摘要。
        2. 在纯内存模式下更新缓存。
        3. 在持久化模式下保留既有 `created_at`，只刷新 `updated_at`。

        Args:
            session_id: 需要保存摘要的会话 ID。
            summary: 待保存的摘要文本。

        Returns:
            保存成功返回 `True`，否则返回 `False`。
        """
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
        """为失败的会话摘要任务安排一次延迟重试。

        与 `jargon_learner` 不同，这里的重试上限耗尽后不会主动丢弃批次，而是停止当前
        这一轮自动重试。这样做更保守：摘要失败不会直接造成历史上下文丢失，只是暂时
        不再继续自动推进，等待后续新的调度机会。

        Args:
            session_id: 需要重试的会话 ID。
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
        """在指定延迟后，把失败会话重新投递回正常处理队列。

        Args:
            session_id: 需要重新排队的会话 ID。
            delay_seconds: 本轮重试前需要等待的秒数。
        """
        try:
            await anyio.sleep(delay_seconds)
        finally:
            self._delayed_retry_sessions.discard(session_id)

        if not self._pending_batches.get(session_id):
            self._retry_attempts.pop(session_id, None)
            return

        await self._queue_job(session_id)


def _truncate(text: str, limit: int) -> str:
    """按字符上限截断文本，并尽量保留省略语义。

    Args:
        text: 待截断的文本。
        limit: 允许保留的最大字符数。

    Returns:
        若原文本未超限则原样返回；否则尽量在尾部附加 `...`。当 `limit <= 3` 时，
        直接返回硬截断结果，避免出现负索引语义混乱。
    """
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..." if limit > 3 else text[:limit]


__all__ = ["SummarizationMiddleware"]
