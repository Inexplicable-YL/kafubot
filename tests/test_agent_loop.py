from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from typing_extensions import override

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agent.base import UserMessage
from agent.message import QQMessage
from kafubot.agency.attention import AttentionScheduler
from kafubot.agency.compiler import ContextCompiler
from kafubot.agency.environment import InMemoryHistoryRepository, SocialEnvironment
from kafubot.agency.executive import MainExecutive
from kafubot.agency.gate import GlobalGate
from kafubot.agency.models import (
    ActiveThread,
    ConversationCandidate,
    ReplyResult,
    SelfState,
)
from kafubot.agency.providers import ConversationWorldProvider, WorldModelHub
from kafubot.agency.state import SelfStateStore
from kafubot.config import AttentionConfig, ExecutiveConfig, GateConfig


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def candidate(
    session_id: str,
    *,
    directed: int = 0,
    new_count: int = 1,
) -> ConversationCandidate:
    return ConversationCandidate(
        session_id=session_id,
        title=session_id,
        preview="hello",
        people=["user"],
        latest_at=datetime.now(UTC),
        new_count=new_count,
        directed_count=directed,
        question_count=0,
        latest_sequence=1,
    )


def user_message(*, directed: bool) -> UserMessage:
    return UserMessage(
        timestamp=datetime.now(UTC),
        user="user",
        message=QQMessage.from_str("hello"),
        user_id="1",
        message_id="2",
        is_tome=directed,
    )


def test_global_gate_wakes_loop_but_does_not_choose_a_reply() -> None:
    gate = GlobalGate(
        GateConfig(threshold=0.75),
        proactive_sessions={"active"},
        talk_value=0.2,
        keywords=set(),
    )
    state = SelfState()

    direct = gate.evaluate(
        "quiet",
        user_message(directed=True),
        candidate("quiet", directed=1),
        state,
    )
    background = gate.evaluate(
        "quiet",
        user_message(directed=False),
        candidate("quiet"),
        state,
    )

    assert direct.wake is True
    assert "directed" in direct.reasons
    assert background.wake is False
    assert not hasattr(direct, "should_reply")


def test_attention_ranks_directed_and_residual_context_globally() -> None:
    now = datetime.now(UTC)
    scheduler = AttentionScheduler(
        AttentionConfig(random_jitter=0),
        clock=lambda: now,
        rng=random.Random(0),  # noqa: S311 - deterministic test jitter
    )
    state = SelfState(
        current_focus=["residual"],
        active_threads={
            "residual": ActiveThread(
                session_id="residual",
                summary="unfinished topic",
                attention_residue=0.9,
                residue_updated_at=now - timedelta(seconds=10),
            )
        },
    )
    home = scheduler.rank(
        [candidate("background"), candidate("direct", directed=1), candidate("residual")],
        state,
        limit=3,
    )

    assert home.items[0].session_id == "direct"
    residual = next(item for item in home.items if item.session_id == "residual")
    background = next(item for item in home.items if item.session_id == "background")
    assert residual.features.continuity > 0
    assert residual.score > background.score


class ScriptedExecutiveModel(BaseChatModel):
    responses: list[AIMessage]

    @property
    @override
    def _llm_type(self) -> str:
        return "scripted-executive"

    @override
    def bind_tools(self, tools: Any, **kwargs: Any) -> BaseChatModel:
        _ = tools, kwargs
        return self

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        _ = messages, stop, run_manager, kwargs
        return ChatResult(
            generations=[ChatGeneration(message=self.responses.pop(0))]
        )


class FakeReplyer:
    def __init__(self) -> None:
        self.sessions: list[str] = []

    async def execute(self, context: Any, contract: Any, actions: Any) -> ReplyResult:
        _ = contract, actions
        self.sessions.append(context.session_id)
        return ReplyResult(full_text="sent", message_count=1)


@pytest.mark.anyio
async def test_one_executive_round_can_handle_multiple_sessions() -> None:
    environment = SocialEnvironment(
        history=InMemoryHistoryRepository(),
        max_entries=50,
    )
    for index, session_id in enumerate(("session_a", "session_b"), start=1):
        message = user_message(directed=True).model_copy(
            update={"message_id": str(index), "user_id": str(index)}
        )
        await environment.observe(session_id, message, cast("Any", object()))

    contract = {
        "target_session_id": "session_a",
        "target_user_ids": ["1"],
        "evidence_message_ids": ["1"],
        "stance": "friendly",
        "relationship_position": "casual peer",
        "response_need": "acknowledge the greeting",
        "behavior": "brief greeting",
        "expected_effect": "continue the conversation",
    }
    model = ScriptedExecutiveModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "open_chat",
                        "args": {"session_id": "session_a"},
                        "id": "open-a",
                        "type": "tool_call",
                    },
                    {
                        "name": "open_chat",
                        "args": {"session_id": "session_b"},
                        "id": "open-b",
                        "type": "tool_call",
                    },
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "reply",
                        "args": {"contract": contract},
                        "id": "reply-a",
                        "type": "tool_call",
                    },
                    {
                        "name": "skip",
                        "args": {
                            "session_id": "session_b",
                            "reason": "no response needed",
                        },
                        "id": "skip-b",
                        "type": "tool_call",
                    },
                    {
                        "name": "finish",
                        "args": {"reason": "handled both"},
                        "id": "finish",
                        "type": "tool_call",
                    },
                ],
            ),
        ]
    )
    config = ExecutiveConfig(max_steps=4, focus_budget=10)
    providers = WorldModelHub([ConversationWorldProvider(environment)])
    state_store = SelfStateStore(None)
    fake_replyer = FakeReplyer()
    executive = MainExecutive(
        environment,
        AttentionScheduler(
            AttentionConfig(random_jitter=0),
            rng=random.Random(0),  # noqa: S311 - deterministic test jitter
        ),
        ContextCompiler(environment, providers, config),
        cast("Any", fake_replyer),
        providers,
        state_store,
        config,
        model_factory=lambda: model,
    )

    result = await executive.run_round()

    assert result.replies == 1
    assert result.skipped == 1
    assert fake_replyer.sessions == ["session_a"]
    assert not await environment.has_unhandled()


@pytest.mark.anyio
async def test_reply_only_consumes_messages_visible_to_its_round() -> None:
    environment = SocialEnvironment(
        history=InMemoryHistoryRepository(),
        max_entries=20,
    )
    first = user_message(directed=True)
    first_candidate = await environment.observe(
        "session",
        first,
        cast("Any", object()),
    )
    second = first.model_copy(update={"message_id": "3"})
    await environment.observe("session", second, cast("Any", object()))

    await environment.commit_reply(
        "session",
        "reply to first",
        handled_through_sequence=first_candidate.latest_sequence,
    )

    remaining = await environment.candidates()
    assert len(remaining) == 1
    assert remaining[0].new_count == 1
    assert remaining[0].preview == "hello"
