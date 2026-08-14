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

import logging
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from typing import Any, cast
from typing_extensions import override

from cachetools import LRUCache
from langchain.agents.middleware.types import (
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langchain.messages import HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages.utils import get_buffer_string
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore

from agent.base import ManagerContext, ManagerState
from agent.middlewares.base import (
    BaseDaemonMiddleware,
    SessionProcessOutput,
)
from agent.utils import content_to_text

logger = logging.getLogger(__name__)


DEFAULT_SUMMARY_PROMPT = """<role>
你是多人互联网聊天的结构化上下文压缩器。旧消息将离开实时窗口，你需要生成可追溯、说话者归属明确的完整替代摘要。
</role>

<required-format>
严格使用以下小节；无内容写“无”：

## 活动话题与线程
分别记录并行话题、参与者、最新进展、热度变化，不要把不同话题合并。

## 未完成互动
记录尚未回答的问题、承诺、等待中的回应，并写出相关 message_id。

## 说话者归属的信息
使用“用户A说/认为/玩笑称……”而不是把个人说法压成客观事实；保留冲突观点和来源。

## 关系、情绪与边界
只记录有消息证据的可观察变化，如互相调侃、安慰、冲突或修复；不要诊断人格和心理。

## 群内梗、称呼与表达
保留正在形成或反复出现的梗、别称、黑话、使用者、语境、含义假设和来源。短暂玩笑在群聊里可能是重要状态，不得仅因短暂而删除。

## 机器人行动与群体影响
记录机器人说过什么、回应了哪个话题、群友如何反应，以及是否可能打断、误解或修复。
</required-format>

<rules>
- 合并现有摘要与新消息，输出能完全替代旧摘要的新版本。
- 事实、推断、玩笑必须区分；低置信度写明“可能”。
- 不得把私聊内容推断到群聊，也不得丢失姓名/用户与 message_id 的来源关系。
- 删除无社会意义的重复、工具噪声和格式噪声，但保留会影响关系、指代、群梗和后续接话的内容。
- 仅输出摘要，不添加说明。
</rules>

<existing_summary>{current_summary}</existing_summary>
<messages>{messages}</messages>"""


SESSION_SUMMARY_INJECTION_PROMPT = """
Here is a summary of the conversation to date:

<system-summary>
{summary}
</system-summary>
""".strip()


DEFAULT_NAMESPACE_ROOT = "session_memory"
_DEFAULT_EMPTY_SUMMARY = "None."
_SUMMARY_SOURCE = "session_summary"


class SummarizationMiddleware(BaseDaemonMiddleware[list["BaseMessage"]]):
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
        super().__init__(
            max_sessions=max_sessions,
            max_retries=max_retries,
            max_batch_window_size=max_batches,
            logger_config={
                "worker_name": "Summary",
                "job_name": "session summary",
                "process_failure_log": "Failed to update session summary",
                "retry_exhausted_label": "Session summary",
            },
        )
        self.summary_model = summary_model
        self.max_input_chars = max_input_chars
        self.max_output_chars = max_output_chars
        self.summary_prompt = summary_prompt

        self._store = store
        self._max_message_chars = max(1, max_input_chars // max_batches)
        self.namespace = (
            str(namespace_root).strip().replace(".", "_") or DEFAULT_NAMESPACE_ROOT,
        )

        self._session_summaries: LRUCache[str, str] = LRUCache(maxsize=max_sessions)
        self._loaded_sessions: LRUCache[str, bool] = LRUCache(maxsize=max_sessions)

    @override
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
        if summary := self._session_summaries.get(runtime.context["session_id"]):
            return {
                "reply_top_messages": [
                    HumanMessage(
                        content=SESSION_SUMMARY_INJECTION_PROMPT.format(
                            summary=summary
                        ),
                        additional_kwargs={"lc_source": _SUMMARY_SOURCE},
                    )
                ]
            }
        return None

    @override
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
            await self._enqueue_batch(runtime.context["session_id"], list(messages))
        return None

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest[ManagerContext],
        handler: Callable[
            [ModelRequest[ManagerContext]], Awaitable[ModelResponse[Any]]
        ],
    ) -> ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]:
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

    @override
    async def process_batches(
        self,
        session_id: str,
        batches: tuple[list[BaseMessage], ...],
    ) -> SessionProcessOutput:
        batch_count, messages_text = self._build_summary_input(batches)
        if batch_count <= 0:
            return None

        if not messages_text:
            return batch_count

        summary = await self._create_summary(
            current_summary=self._session_summaries.get(
                session_id,
                _DEFAULT_EMPTY_SUMMARY,
            ),
            messages_text=messages_text,
        )
        if summary is None or not await self._save_session_summary(session_id, summary):
            return 0, True

        return batch_count

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
        batches: Sequence[list[BaseMessage]],
    ) -> tuple[int, str]:
        """从待处理批次中构建一次摘要模型调用的输入。

        输入拼装策略与后台摘要的吞吐和成本直接相关，因此这里做了三层约束：

        1. 单条消息先按 `_max_message_chars` 截断，防止极长消息污染整批输入。
        2. 父类会先按固定窗口大小切好批次，子类这里只处理当前窗口。
        3. 总文本超过 `max_input_chars` 后停止扩展；若超限发生在首批，则保留首批并
           做整体截断，确保任务仍然能够继续前进，而不是被一条超长历史永久卡住。

        Args:
            batches: 父类已经切好的当前批次窗口。

        Returns:
            一个二元组 `(batch_count, messages_text)`，其中 `batch_count` 表示本轮纳入
            的批次数，`messages_text` 是格式化后的摘要输入文本。
        """
        selected_messages: list[BaseMessage] = []
        batch_count = 0

        for batch in batches:
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

    async def clear_session(self, session_id: str) -> None:
        """Delete both cached and persisted summary state for one QQ session."""
        self._session_summaries.pop(session_id, None)
        self._loaded_sessions.pop(session_id, None)
        if self._store is not None:
            await self._store.adelete(self.namespace, session_id)

    @override
    async def on_close(self) -> None:
        self._store = None
        self._loaded_sessions.clear()
        self._session_summaries.clear()


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
