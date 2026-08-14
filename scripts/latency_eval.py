from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from time import perf_counter
from typing import Any
from typing_extensions import override

import anyio
from langchain_core.messages import AIMessage

from agent.base import UserMessage
from agent.message import QQMessage
from agent.social_signals import (
    SocialSignalAnalysis,
    SocialSignalAnalyzer,
    SocialSignalService,
)


@dataclass(slots=True)
class _ProbeAnalyzer(SocialSignalAnalyzer):
    delay: float
    calls: list[list[str]] = field(default_factory=list)

    def __init__(self, delay: float) -> None:
        SocialSignalAnalyzer.__init__(self, None)
        self.delay = delay
        self.calls = []

    @override
    async def analyze(
        self,
        *,
        target_messages: list[UserMessage],
        context_messages: list[UserMessage],
        bot_message: str | None = None,
    ) -> list[SocialSignalAnalysis]:
        del context_messages, bot_message
        self.calls.append([message.message_id for message in target_messages])
        await anyio.sleep(self.delay)
        return [
            SocialSignalAnalysis(
                message_id=message.message_id,
                valence="neutral",
                target="bot",
                confidence=0.8,
                evidence="latency probe",
            )
            for message in target_messages
        ]


def _message(index: int) -> UserMessage:
    return UserMessage(
        timestamp=datetime.now(UTC),
        user=f"u{index}",
        message=QQMessage(f"message {index}"),
        user_id=str(index),
        message_id=str(index),
        is_tome=True,
    )


async def run_probe(message_count: int, delay: float) -> dict[str, Any]:
    analyzer = _ProbeAnalyzer(delay)
    service = SocialSignalService(
        analyzer,
        coalesce_seconds=0.1,
        max_concurrency=2,
        analysis_timeout=delay + 1.0,
    )
    messages = [_message(index) for index in range(message_count)]
    started = perf_counter()
    for message in messages:
        await service.schedule(
            session_id="probe",
            target_messages=[message],
            context_messages=messages,
            bot_message=AIMessage(content="probe reply").text,
            force=True,
        )
    enqueue_seconds = perf_counter() - started
    await service.wait_ready(
        "probe",
        [message.message_id for message in messages],
        bot_message="probe reply",
        timeout=delay + 2.0,
    )
    total_seconds = perf_counter() - started
    await service.aclose()
    result = {
        "messages": message_count,
        "simulated_model_seconds": delay,
        "enqueue_seconds": round(enqueue_seconds, 6),
        "total_seconds": round(total_seconds, 6),
        "model_calls": len(analyzer.calls),
        "batch_sizes": [len(batch) for batch in analyzer.calls],
        "reply_path_blocked_by_model": enqueue_seconds >= delay * 0.5,
    }
    if result["model_calls"] != 1:
        raise AssertionError(f"expected one coalesced model call, got {result!r}")
    if result["batch_sizes"] != [message_count]:
        raise AssertionError(f"messages were not analyzed as one batch: {result!r}")
    if result["reply_path_blocked_by_model"]:
        raise AssertionError(f"model latency leaked into enqueue path: {result!r}")
    return result


async def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--messages", type=int, default=8)
    parser.add_argument("--model-delay", type=float, default=0.5)
    args = parser.parse_args()
    print(
        json.dumps(
            await run_probe(max(1, args.messages), max(0.01, args.model_delay)),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    anyio.run(_main)
