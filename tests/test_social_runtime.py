from __future__ import annotations

from typing import Any

import anyio
import pytest

from kafubot.adapters.cqhttp.event import (
    GroupMessageEvent,
    PrivateMessageEvent,
    parse_event,
)
from kafubot.agency.environment import (
    ConversationState,
    InMemoryHistoryRepository,
    SocialEnvironment,
)
from kafubot.agency.models import RoundResult
from kafubot.agency.state import SelfStateStore
from kafubot.config import AgentConfig, GateConfig
from kafubot.social import SocialAgentRuntime


def message_payload(message_type: str, identity: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "time": 1_700_000_000,
        "self_id": 10001,
        "post_type": "message",
        "message_type": message_type,
        "sub_type": "normal" if message_type == "group" else "friend",
        "message_id": identity,
        "user_id": identity,
        "message": [{"type": "text", "data": {"text": "hello"}}],
        "raw_message": "hello",
        "font": 0,
        "sender": {"user_id": identity, "nickname": str(identity)},
        "to_me": True,
    }
    if message_type == "group":
        payload["group_id"] = 20002
    return payload


class FakeAnalyzer:
    async def ainvoke(self, value: dict[str, Any]) -> str:
        _ = value
        return "image"


class FakeExecutive:
    def __init__(self, environment: SocialEnvironment) -> None:
        self.environment = environment
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self.completed = anyio.Event()

    async def run_round(self) -> RoundResult:
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await anyio.sleep(0.02)
        for candidate in await self.environment.candidates():
            await self.environment.mark_handled(candidate.session_id)
        self.active -= 1
        self.completed.set()
        return RoundResult(home_size=2, steps=1)


def make_runtime() -> tuple[SocialAgentRuntime, FakeExecutive]:
    environment = SocialEnvironment(
        history=InMemoryHistoryRepository(),
        max_entries=50,
    )
    executive = FakeExecutive(environment)
    runtime = SocialAgentRuntime(
        AgentConfig(
            gate=GateConfig(debounce_seconds=0, idle_wake_seconds=10),
        ),
        environment=environment,
        state_store=SelfStateStore(None),
        executive=executive,
        image_analyzer_factory=FakeAnalyzer,
    )
    return runtime, executive


@pytest.mark.anyio
async def test_group_and_private_are_only_context_partitions() -> None:
    runtime, _ = make_runtime()
    group = parse_event(object(), message_payload("group", 30003))
    private = parse_event(object(), message_payload("private", 40004))
    assert isinstance(group, GroupMessageEvent)
    assert isinstance(private, PrivateMessageEvent)

    group_state = await runtime.session(group.conversation_id)
    private_state = await runtime.session(private.conversation_id)

    assert group.conversation_id == "group_20002"
    assert private.conversation_id == "private_40004"
    assert isinstance(group_state, ConversationState)
    assert type(group_state) is type(private_state)


@pytest.mark.anyio
async def test_event_workers_only_update_environment_and_one_loop_executes() -> None:
    runtime, executive = make_runtime()
    events = [
        parse_event(object(), message_payload("group", 30003)),
        parse_event(object(), message_payload("private", 40004)),
    ]

    for event in events:
        assert isinstance(event, (GroupMessageEvent, PrivateMessageEvent))
        await runtime.handle(event)

    assert executive.calls == 0
    assert len(await runtime.environment.candidates()) == 2

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(runtime.run)
        await executive.completed.wait()
        await runtime.aclose()
        task_group.cancel_scope.cancel()

    assert executive.calls == 1
    assert executive.max_active == 1
