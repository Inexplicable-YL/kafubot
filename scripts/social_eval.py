from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            records.append(item)
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    events = Counter(str(item.get("event")) for item in records)
    actions = [
        item for item in records if item.get("event") == "social_action_selected"
    ]
    replies = [item for item in records if item.get("event") == "qq_reply_sent"]
    effects = [item for item in records if item.get("event") == "reply_effect_observed"]
    gates = [item for item in records if item.get("event") == "intervention_gate"]
    quoted = sum(
        bool(item.get("action", {}).get("quote_message_id")) for item in actions
    )
    group_directed = sum(
        not item.get("action", {}).get("target_user_ids") for item in actions
    )
    corrections = sum(
        int(item.get("metrics", {}).get("correction_count", 0)) for item in effects
    )
    negative = sum(
        int(item.get("metrics", {}).get("negative_signal_count", 0)) for item in effects
    )
    ignored = sum(
        bool(item.get("metrics", {}).get("ignored_in_window")) for item in effects
    )
    continued = sum(
        bool(item.get("metrics", {}).get("target_continued")) for item in effects
    )
    rewards = [
        float(item.get("metrics", {}).get("observable_reward", 0.0)) for item in effects
    ]
    confidences = [float(item.get("attribution_confidence", 0.0)) for item in effects]
    allowed_gates = sum(bool(item.get("allowed")) for item in gates)
    directed_gates = sum(bool(item.get("directed_to_bot")) for item in gates)
    goals = Counter(str(item.get("action", {}).get("social_goal")) for item in actions)
    missing_evidence = sum(
        not item.get("action", {}).get("evidence_message_ids") for item in actions
    )
    return {
        "records": len(records),
        "events": dict(events),
        "reply_count": len(replies),
        "quote_rate": quoted / len(actions) if actions else 0.0,
        "group_or_topic_directed_rate": group_directed / len(actions)
        if actions
        else 0.0,
        "observed_corrections": corrections,
        "observed_negative_signals": negative,
        "ignored_effect_rate": ignored / len(effects) if effects else 0.0,
        "target_continuation_rate": continued / len(effects) if effects else 0.0,
        "mean_observable_reward": mean(rewards) if rewards else 0.0,
        "mean_attribution_confidence": mean(confidences) if confidences else 0.0,
        "effect_coverage": len(effects) / len(replies) if replies else 0.0,
        "gate_admission_rate": allowed_gates / len(gates) if gates else 0.0,
        "undirected_gate_admission_rate": (
            (allowed_gates - directed_gates) / max(1, len(gates) - directed_gates)
        ),
        "action_goal_counts": dict(goals),
        "actions_missing_evidence": missing_evidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize KafuBot social-control telemetry"
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path(".logs/social_control.jsonl"),
    )
    args = parser.parse_args()
    print(json.dumps(summarize(load_records(args.path)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
