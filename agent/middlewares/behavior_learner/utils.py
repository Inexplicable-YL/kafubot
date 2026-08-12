"""Shared helpers for behavior learner modules."""

import json
from collections.abc import Sequence
from typing import Any

from .constants import MAX_BEHAVIOR_SCORE, MIN_BEHAVIOR_SCORE


def strip_json_code_fence(raw_response: str) -> str:
    normalized_response = raw_response.strip()
    if not normalized_response.startswith("```"):
        return normalized_response
    lines = normalized_response.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return normalized_response


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def load_json_list(raw_value: Any) -> list[Any]:
    if not raw_value:
        return []
    if isinstance(raw_value, list):
        return raw_value
    if not isinstance(raw_value, str):
        return []
    try:
        parsed = json.loads(raw_value)
    except (TypeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def dump_json_list(items: Sequence[Any]) -> str:
    return json.dumps(list(items), ensure_ascii=False)


def clamp_score(score: float) -> float:
    return min(MAX_BEHAVIOR_SCORE, max(MIN_BEHAVIOR_SCORE, score))


def normalize_source_ids(source_ids: Sequence[str]) -> list[str]:
    normalized_ids: list[str] = []
    for source_id in source_ids:
        normalized_id = str(source_id or "").strip()
        if not normalized_id or normalized_id in normalized_ids:
            continue
        normalized_ids.append(normalized_id)
    return normalized_ids


__all__ = [
    "clean_text",
    "clamp_score",
    "dump_json_list",
    "load_json_list",
    "normalize_source_ids",
    "strip_json_code_fence",
]
