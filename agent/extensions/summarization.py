from __future__ import annotations

import logging
from collections import defaultdict, deque
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import anyio
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.messages.utils import get_buffer_string

from agent.base import ManagerContext, ManagerState
from agent.utils import content_to_text

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

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

{summary}
""".strip()


DEFAULT_EMPTY_SUMMARY = "None."
SUMMARY_SOURCE = "session_summary"


class SummarizationMiddleware(AgentMiddleware[ManagerState, ManagerContext]):
    state_schema = ManagerState

    def __init__(
        self,
        summary_model: BaseChatModel,
        *,
        summary_context_batches: int = 3,
        queue_size: int = 100,
        max_message_chars: int = 6000,
        max_summary_input_chars: int = 24000,
        max_summary_chars: int = 3000,
        summary_prompt: str = DEFAULT_SUMMARY_PROMPT,
        store: BaseStore | None = None,
        namespace_root: str = "session_memory",
    ) -> None:
        super().__init__()
        self.summary_model = summary_model
        self.summary_context_batches = self._validate_positive_int(
            summary_context_batches,
            "summary_context_batches",
        )
        self.queue_size = self._validate_positive_int(queue_size, "queue_size")
        self.max_message_chars = self._validate_positive_int(
            max_message_chars,
            "max_message_chars",
        )
        self.max_summary_input_chars = self._validate_positive_int(
            max_summary_input_chars,
            "max_summary_input_chars",
        )
        self.max_summary_chars = self._validate_positive_int(
            max_summary_chars,
            "max_summary_chars",
        )
        self.summary_prompt = summary_prompt
        normalized_namespace_root = (
            str(namespace_root).strip().replace(".", "_") or "session_memory"
        )
        self.namespace = (normalized_namespace_root,)

        self._pending_batches: dict[str, deque[list[BaseMessage]]] = defaultdict(deque)
        self._session_summaries: dict[str, str] = {}
        self._loaded_sessions: set[str] = set()
        self._scheduled_sessions: set[str] = set()
        self._store: BaseStore | None = store

        self._start_lock = anyio.Lock()
        self._task_group: TaskGroup | None = None
        self._send_stream: MemoryObjectSendStream[str] | None = None
        self._receive_stream: MemoryObjectReceiveStream[str] | None = None

    async def abefore_agent(
        self, state: ManagerState, runtime: Runtime[ManagerContext]
    ) -> dict[str, Any] | None:
        _ = state
        if runtime.store is not None:
            self._store = self._store or runtime.store
        await self._ensure_session_summary_loaded(runtime.context["session_id"])

    async def aafter_agent(
        self, state: ManagerState, runtime: Runtime[ManagerContext]
    ) -> dict[str, Any] | None:
        if runtime.store is not None:
            self._store = self._store or runtime.store
        await self._handle_pruned_messages(
            session_id=runtime.context["session_id"],
            messages=state.get("summary_pruned_messages"),
        )

    async def awrap_model_call(
        self,
        request: ModelRequest[ManagerContext],
        handler,
    ) -> Any:
        if request.runtime.store is not None:
            self._store = self._store or request.runtime.store
        await self._ensure_session_summary_loaded(request.runtime.context["session_id"])
        summary = self._session_summaries.get(request.runtime.context["session_id"])
        if not summary:
            return await handler(request)

        summary_message = HumanMessage(
            content=SESSION_SUMMARY_INJECTION_PROMPT.format(summary=summary),
            additional_kwargs={"lc_source": SUMMARY_SOURCE},
        )
        return await handler(
            request.override(messages=[summary_message, *request.messages])
        )

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
        self._loaded_sessions.clear()
        self._scheduled_sessions.clear()

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
            task_group.start_soon(self._summary_worker, receive_stream)

            self._send_stream = send_stream
            self._receive_stream = receive_stream
            self._task_group = task_group

    async def _handle_pruned_messages(
        self,
        *,
        session_id: str,
        messages: list[BaseMessage] | None,
    ) -> None:
        if not messages:
            return
        await self._ensure_service()
        self._pending_batches[session_id].append(list(messages))
        await self._queue_summary_job(session_id)

    async def _queue_summary_job(self, session_id: str) -> None:
        send_stream = self._send_stream
        task_group = self._task_group
        if send_stream is None or task_group is None:
            return
        if session_id in self._scheduled_sessions:
            return

        self._scheduled_sessions.add(session_id)
        try:
            send_stream.send_nowait(session_id)
        except anyio.WouldBlock:
            pass
        except (anyio.BrokenResourceError, anyio.ClosedResourceError):
            self._scheduled_sessions.discard(session_id)
            logger.exception("Failed to queue session summary job")
            return
        else:
            return

        try:
            task_group.start_soon(self._send_summary_job, send_stream, session_id)
        except RuntimeError:
            self._scheduled_sessions.discard(session_id)
            logger.exception("Failed to schedule session summary job")

    async def _send_summary_job(
        self,
        send_stream: MemoryObjectSendStream[str],
        session_id: str,
    ) -> None:
        try:
            await send_stream.send(session_id)
        except (anyio.BrokenResourceError, anyio.ClosedResourceError):
            self._scheduled_sessions.discard(session_id)
            logger.exception("Failed to send session summary job")

    async def _summary_worker(
        self,
        receive_stream: MemoryObjectReceiveStream[str],
    ) -> None:
        async with receive_stream:
            async for session_id in receive_stream:
                try:
                    progressed = await self._summarize_session(session_id)
                except Exception:
                    logger.exception("Failed to update session summary")
                    progressed = False
                finally:
                    self._scheduled_sessions.discard(session_id)

                if progressed and self._pending_batches.get(session_id):
                    await self._queue_summary_job(session_id)

    async def _summarize_session(self, session_id: str) -> bool:
        progressed = False

        while pending_batches := self._pending_batches.get(session_id):
            batch_count, messages_text = self._collect_batches_to_summarize(
                pending_batches
            )
            if batch_count <= 0:
                break

            if not messages_text:
                self._consume_batches(session_id, batch_count)
                progressed = True
                continue

            summary = await self._acreate_summary(
                current_summary=self._session_summaries.get(
                    session_id,
                    DEFAULT_EMPTY_SUMMARY,
                ),
                messages_text=messages_text,
            )
            if summary is None:
                break

            if not await self._save_session_summary(session_id, summary):
                break
            self._consume_batches(session_id, batch_count)
            progressed = True

        return progressed

    async def _acreate_summary(
        self,
        *,
        current_summary: str,
        messages_text: str,
    ) -> str | None:
        try:
            response = await self.summary_model.ainvoke(
                self.summary_prompt.format(
                    current_summary=current_summary,
                    messages=messages_text,
                ).rstrip(),
                config={"metadata": {"lc_source": SUMMARY_SOURCE}},
            )
        except Exception:
            logger.exception("Failed to generate session summary")
            return None

        summary = content_to_text(response.content).strip()
        if not summary:
            logger.warning("Summary model returned an empty session summary")
            return None
        return summary

    def _collect_batches_to_summarize(
        self,
        pending_batches: deque[list[BaseMessage]],
    ) -> tuple[int, str]:
        selected_batches: list[list[BaseMessage]] = []
        selected_count = 0
        last_good_text = ""

        for batch in list(pending_batches)[: self.summary_context_batches]:
            selected_batches.append(batch)
            candidate_text = self._format_batches_for_summary(selected_batches)

            if not candidate_text:
                selected_count += 1
                continue

            if len(candidate_text) > self.max_summary_input_chars:
                if selected_count == 0:
                    logger.warning(
                        "A single pruned batch exceeded max_summary_input_chars; truncating summary input."
                    )
                    return 1, self._truncate_text(
                        candidate_text,
                        self.max_summary_input_chars,
                    )
                break

            last_good_text = candidate_text
            selected_count += 1

        if selected_count == 0 and pending_batches:
            first_batch_text = self._format_batches_for_summary([pending_batches[0]])
            return 1, self._truncate_text(
                first_batch_text,
                self.max_summary_input_chars,
            )

        return selected_count, last_good_text

    def _format_batches_for_summary(
        self,
        batches: Sequence[list[BaseMessage]],
    ) -> str:
        normalized_messages = self._normalize_messages_for_summary(
            message for batch in batches for message in batch
        )
        if not normalized_messages:
            return ""
        return get_buffer_string(normalized_messages).strip()

    def _normalize_messages_for_summary(
        self,
        messages: Iterable[BaseMessage],
    ) -> list[BaseMessage]:
        normalized_messages: list[BaseMessage] = []
        for message in messages:
            text = content_to_text(message.content).strip()
            if not text:
                continue

            normalized_messages.append(
                message.model_copy(
                    update={
                        "content": self._truncate_text(
                            text,
                            self.max_message_chars,
                        )
                    }
                )
            )
        return normalized_messages

    def _consume_batches(self, session_id: str, batch_count: int) -> None:
        pending_batches = self._pending_batches[session_id]
        for _ in range(min(batch_count, len(pending_batches))):
            pending_batches.popleft()

        if not pending_batches:
            self._pending_batches.pop(session_id, None)

    @staticmethod
    def _truncate_text(text: str, limit: int) -> str:
        if limit <= 0:
            return ""
        if len(text) <= limit:
            return text
        if limit <= 3:
            return text[:limit]
        return text[: limit - 3] + "..."

    async def _ensure_session_summary_loaded(self, session_id: str) -> None:
        if session_id in self._loaded_sessions:
            return
        if self._store is None:
            self._loaded_sessions.add(session_id)
            return
        try:
            item = await self._store.aget(self.namespace, session_id)
        except Exception:
            logger.exception("Failed to load session summary from store")
            return
        if item is None:
            self._loaded_sessions.add(session_id)
            return
        summary = item.value.get("summary", "").strip()
        if summary:
            self._session_summaries[session_id] = self._truncate_text(
                summary,
                self.max_summary_chars,
            )
        self._loaded_sessions.add(session_id)

    async def _save_session_summary(self, session_id: str, summary: str) -> bool:
        normalized_summary = self._truncate_text(
            summary.strip(), self.max_summary_chars
        )
        if not normalized_summary:
            return False

        store = self._store
        if store is None:
            self._session_summaries[session_id] = normalized_summary
            self._loaded_sessions.add(session_id)
            return True

        created_at = datetime.now(UTC).isoformat()
        try:
            existing_item = await store.aget(self.namespace, session_id)
            if existing_item is not None and isinstance(
                existing_item.value.get("created_at"), str
            ):
                created_at = existing_item.value["created_at"]

            now = datetime.now(UTC).isoformat()
            await store.aput(
                self.namespace,
                session_id,
                {
                    "id": session_id,
                    "session_id": session_id,
                    "summary": normalized_summary,
                    "created_at": created_at,
                    "updated_at": now,
                },
                index=False,
            )
        except Exception:
            logger.exception("Failed to persist session summary")
            return False

        self._session_summaries[session_id] = normalized_summary
        self._loaded_sessions.add(session_id)
        return True

    @staticmethod
    def _validate_positive_int(value: int, name: str) -> int:
        if value < 1:
            raise ValueError(f"{name} must be greater than 0, got {value}.")
        return value


__all__ = ["SummarizationMiddleware"]
