from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import aiosqlite
import anyio

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from kafubot.cognition.plugins.conversation import (  # noqa: E402
    MessageEvent,
    _SessionGraph,
)


async def load_events(db_path: Path, session_id: str) -> list[MessageEvent]:
    if not db_path.exists():
        return []
    async with aiosqlite.connect(db_path) as db:
        schema_cursor = await db.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            ("qq_message_events",),
        )
        has_table = await schema_cursor.fetchone()
        await schema_cursor.close()
        if not has_table:
            return []
        cursor = await db.execute(
            """
            SELECT message_id, chat_type, sender_id, sender_name, timestamp,
                   text, msgcode, reply_to_id, mention_user_ids_json,
                   directed_to_bot, has_image
            FROM qq_message_events
            WHERE session_id = ?
            ORDER BY timestamp
            """,
            (session_id,),
        )
        rows = await cursor.fetchall()
        await cursor.close()
    return [
        MessageEvent.model_validate(
            {
                "session_id": session_id,
                "message_id": row[0],
                "chat_type": row[1],
                "sender_id": row[2],
                "sender_name": row[3],
                "timestamp": row[4],
                "text": row[5],
                "msgcode": row[6],
                "reply_to_id": row[7],
                "mention_user_ids": json.loads(row[8] or "[]"),
                "directed_to_bot": bool(row[9]),
                "has_image": bool(row[10]),
            }
        )
        for row in rows
    ]


def evaluate(events: list[MessageEvent], session_id: str) -> dict[str, Any]:
    graph = _SessionGraph(session_id, max_events=max(160, len(events) + 1))
    for event in events:
        graph.observe_user(event)
    frame = graph.frame()
    explicit_addressing = sum(
        bool(event.reply_to_id or event.mention_user_ids or event.directed_to_bot)
        for event in events
    )
    inferred_hypotheses = sum(
        hypothesis.inferred for hypothesis in frame.addressee_hypotheses
    )
    return {
        "session_id": session_id,
        "event_count": len(events),
        "participant_count": len(frame.participants),
        "active_thread_count": len(frame.active_threads),
        "explicit_addressing_rate": (
            explicit_addressing / len(events) if events else 0.0
        ),
        "inferred_addressee_hypotheses": inferred_hypotheses,
        "group_state": frame.group_state.model_dump(mode="json"),
        "threads": [thread.model_dump(mode="json") for thread in frame.active_threads],
        "evidence_links": [
            link.model_dump(mode="json") for link in frame.evidence_links
        ],
    }


async def async_main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay the QQ-observable event ledger through conversation framing"
    )
    parser.add_argument("session_id", help="QQ session id, for example group_123")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(".database/qq_event_ledger.db"),
    )
    args = parser.parse_args()
    events = await load_events(args.db, args.session_id)
    print(json.dumps(evaluate(events, args.session_id), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    anyio.run(async_main)
