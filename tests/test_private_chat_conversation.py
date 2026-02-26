from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from nodes import private_chat
from nodes.private_chat import PrivateReply, PrivateReplyState, SessionConversationStore


class FakeStreamResponse:
    def __init__(self, lines: list[str], status_code: int = 200) -> None:
        self._lines = lines
        self._status_code = status_code

    def raise_for_status(self) -> None:
        if self._status_code < 400:
            return
        request = httpx.Request("POST", "http://unit.test/chat-messages")
        response = httpx.Response(self._status_code, request=request)
        raise httpx.HTTPStatusError("request failed", request=request, response=response)

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class FakeDeleteResponse:
    def __init__(self, status_code: int) -> None:
        self._status_code = status_code

    def raise_for_status(self) -> None:
        if self._status_code < 400:
            return
        request = httpx.Request("DELETE", "http://unit.test/conversations/conv-id")
        response = httpx.Response(self._status_code, request=request)
        raise httpx.HTTPStatusError("request failed", request=request, response=response)


def build_node(state: PrivateReplyState) -> PrivateReply:
    node = PrivateReply()
    node._name = "PrivateReply"
    node.event = SimpleNamespace(
        adapter=SimpleNamespace(bot=SimpleNamespace(node_state={"PrivateReply": state}))
    )
    node.reply = AsyncMock()
    return node


@pytest.mark.anyio
async def test_session_conversation_store_crud(tmp_path) -> None:
    store = SessionConversationStore(db_path=tmp_path / "private_chat.db")

    assert await store.get_conversation_id("session-a") is None

    await store.set_conversation_id("session-a", "conversation-1")
    assert await store.get_conversation_id("session-a") == "conversation-1"

    await store.set_conversation_id("session-a", "conversation-2")
    assert await store.get_conversation_id("session-a") == "conversation-2"

    await store.delete_conversation_id("session-a")
    assert await store.get_conversation_id("session-a") is None


@pytest.mark.anyio
async def test_run_chat_first_call_sends_empty_conversation_and_persists_id(
    tmp_path, monkeypatch
) -> None:
    store = SessionConversationStore(db_path=tmp_path / "private_chat.db")
    state = PrivateReplyState(conversation_store=store)
    node = build_node(state)
    captured_payload: dict[str, str] = {}

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def _send_request(self, method, endpoint, json=None, stream=False):
            captured_payload["method"] = method
            captured_payload["endpoint"] = endpoint
            captured_payload["conversation_id"] = json["conversation_id"]
            return FakeStreamResponse(
                [
                    'data: {"event":"message","answer":"hello","conversation_id":"conv-1"}',
                    'data: {"event":"message_end","conversation_id":"conv-1"}',
                ]
            )

    monkeypatch.setattr(private_chat, "AsyncChatClient", lambda *args, **kwargs: FakeClient())

    await node._run_chat(query="hi", name="tester", session_id="session-a")

    assert captured_payload["method"] == "POST"
    assert captured_payload["endpoint"] == "/chat-messages"
    assert captured_payload["conversation_id"] == ""
    assert await store.get_conversation_id("session-a") == "conv-1"
    node.reply.assert_awaited_once_with("hello")


@pytest.mark.anyio
async def test_run_chat_reuses_existing_conversation_id(tmp_path, monkeypatch) -> None:
    store = SessionConversationStore(db_path=tmp_path / "private_chat.db")
    await store.set_conversation_id("session-a", "conv-old")
    state = PrivateReplyState(conversation_store=store)
    node = build_node(state)
    captured_payload: dict[str, str] = {}

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def _send_request(self, method, endpoint, json=None, stream=False):
            captured_payload["conversation_id"] = json["conversation_id"]
            return FakeStreamResponse(
                ['data: {"event":"message_end","conversation_id":"conv-old"}']
            )

    monkeypatch.setattr(private_chat, "AsyncChatClient", lambda *args, **kwargs: FakeClient())

    await node._run_chat(query="next", name="tester", session_id="session-a")

    assert captured_payload["conversation_id"] == "conv-old"
    assert await store.get_conversation_id("session-a") == "conv-old"


@pytest.mark.anyio
async def test_delete_conversation_success_removes_mapping(tmp_path, monkeypatch) -> None:
    store = SessionConversationStore(db_path=tmp_path / "private_chat.db")
    await store.set_conversation_id("session-a", "conv-1")
    state = PrivateReplyState(conversation_store=store)
    node = build_node(state)
    delete_args: dict[str, str] = {}

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def delete_conversation(self, conversation_id: str, user: str):
            delete_args["conversation_id"] = conversation_id
            delete_args["user"] = user
            return FakeDeleteResponse(200)

    monkeypatch.setattr(private_chat, "AsyncChatClient", lambda *args, **kwargs: FakeClient())

    await node._delete_conversation("session-a")

    assert delete_args == {"conversation_id": "conv-1", "user": "session-a"}
    assert await store.get_conversation_id("session-a") is None


@pytest.mark.anyio
async def test_delete_conversation_404_removes_mapping(tmp_path, monkeypatch) -> None:
    store = SessionConversationStore(db_path=tmp_path / "private_chat.db")
    await store.set_conversation_id("session-a", "conv-1")
    state = PrivateReplyState(conversation_store=store)
    node = build_node(state)

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def delete_conversation(self, conversation_id: str, user: str):
            return FakeDeleteResponse(404)

    monkeypatch.setattr(private_chat, "AsyncChatClient", lambda *args, **kwargs: FakeClient())

    await node._delete_conversation("session-a")

    assert await store.get_conversation_id("session-a") is None


@pytest.mark.anyio
async def test_delete_conversation_500_keeps_mapping(tmp_path, monkeypatch) -> None:
    store = SessionConversationStore(db_path=tmp_path / "private_chat.db")
    await store.set_conversation_id("session-a", "conv-1")
    state = PrivateReplyState(conversation_store=store)
    node = build_node(state)

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def delete_conversation(self, conversation_id: str, user: str):
            return FakeDeleteResponse(500)

    monkeypatch.setattr(private_chat, "AsyncChatClient", lambda *args, **kwargs: FakeClient())

    await node._delete_conversation("session-a")

    assert await store.get_conversation_id("session-a") == "conv-1"
