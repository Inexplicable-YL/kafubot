from __future__ import annotations

import json
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from json_repair import repair_json


@dataclass(frozen=True, slots=True)
class ExpressionRuntimeConfig:
    use_expression: bool = True
    enable_learning: bool = True


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


def dump_json_list(items: list[Any]) -> str:
    return json.dumps(list(items), ensure_ascii=False)


def _compute_weights(population: list[dict[str, Any]]) -> list[float]:
    if not population:
        return []

    counts: list[float] = []
    for item in population:
        count = item.get("count", 1)
        try:
            count_value = float(count)
        except (TypeError, ValueError):
            count_value = 1.0
        counts.append(max(count_value, 0.0))

    min_count = min(counts)
    max_count = max(counts)
    if max_count == min_count:
        return [1.0 for _ in counts]

    weights: list[float] = []
    for count_value in counts:
        normalized = (count_value - min_count) / (max_count - min_count)
        weights.append(1.0 + normalized * 4.0)
    return weights


def weighted_sample(population: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    if not population or k <= 0:
        return []
    if len(population) <= k:
        return population.copy()

    selected: list[dict[str, Any]] = []
    population_copy = population.copy()
    for _ in range(min(k, len(population_copy))):
        weights = _compute_weights(population_copy)
        total_weight = sum(weights)
        if total_weight <= 0:
            index = random.randint(0, len(population_copy) - 1)  # noqa: S311
            selected.append(population_copy.pop(index))
            continue

        threshold = random.uniform(0, total_weight)  # noqa: S311
        cumulative = 0.0
        for index, weight in enumerate(weights):
            cumulative += weight
            if threshold <= cumulative:
                selected.append(population_copy.pop(index))
                break
    return selected


def _normalize_repair_json_result(repaired_result: Any) -> str:
    if isinstance(repaired_result, str):
        return repaired_result
    if isinstance(repaired_result, tuple) and repaired_result:
        first_item = repaired_result[0]
        if isinstance(first_item, str):
            return first_item
        return json.dumps(first_item, ensure_ascii=False)
    raise TypeError(
        f"repair_json 杩斿洖浜嗘棤娉曞鐞嗙殑缁撴灉绫诲瀷: {type(repaired_result)}"
    )


def _strip_markdown_code_fence(text: str) -> str:
    raw = text.strip()
    if match := re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL):
        return match[1].strip()
    raw = re.sub(r"^```\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)
    return raw.strip()


def _extract_json_object_candidate(text: str) -> str:
    start_index = text.find("{")
    end_index = text.rfind("}")
    if start_index != -1 and end_index != -1 and start_index < end_index:
        return text[start_index : end_index + 1].strip()
    return text.strip()


def _extract_reason_from_text(text: str) -> str | None:
    reason_key_match = re.search(
        r'["鈥溾€漖?reason["鈥溾€漖?\s*:\s*', text, re.IGNORECASE
    )
    if reason_key_match is None:
        return None

    value_text = text[reason_key_match.end() :].strip()
    if not value_text:
        return None

    if value_text.endswith("}"):
        value_text = value_text[:-1].rstrip()
    if value_text.endswith(","):
        value_text = value_text[:-1].rstrip()
    if not value_text:
        return None

    if value_text[0] in {'"', "'", "“", "”", "‘", "’"}:
        value_text = value_text[1:]
        while value_text and value_text[-1] in {'"', "'", "“", "”", "‘", "’"}:
            value_text = value_text[:-1].rstrip()

    return value_text.strip() or None


def _normalize_reason_text(reason: Any) -> str:
    normalized_reason = str(reason).strip()
    if (
        len(normalized_reason) >= 2
        and normalized_reason[0] == normalized_reason[-1]
        and normalized_reason[0] in {'"', "'", "“", "”", "‘", "’"}
    ):
        normalized_reason = normalized_reason[1:-1].strip()

    if normalized_reason.endswith('"') and normalized_reason.count('"') % 2 == 1:
        normalized_reason = normalized_reason[:-1].rstrip()
    if normalized_reason.endswith("'") and normalized_reason.count("'") % 2 == 1:
        normalized_reason = normalized_reason[:-1].rstrip()
    if normalized_reason.endswith('"') and not normalized_reason.startswith('"'):
        normalized_reason = normalized_reason[:-1].rstrip()
    if normalized_reason.endswith("'") and not normalized_reason.startswith("'"):
        normalized_reason = normalized_reason[:-1].rstrip()
    return normalized_reason


def fix_chinese_quotes_in_json(text: str) -> str:
    result: list[str] = []
    in_string = False
    escape_next = False

    for char in text:
        if escape_next:
            result.append(char)
            escape_next = False
            continue
        if char == "\\":
            result.append(char)
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            result.append(char)
            continue
        if in_string and char in ["“", "”"]:
            result.append('\\"')
            continue
        result.append(char)
    return "".join(result)


def parse_evaluation_response(response: str) -> dict[str, Any]:
    raw = _strip_markdown_code_fence(response)
    if not raw:
        msg = "LLM 鍝嶅簲涓虹┖"
        raise ValueError(msg)

    parse_candidates = [raw]
    json_candidate = _extract_json_object_candidate(raw)
    if json_candidate and json_candidate not in parse_candidates:
        parse_candidates.append(json_candidate)

    for candidate in parse_candidates:
        parsed = _try_parse(candidate)
        if isinstance(parsed, dict):
            if "reason" in parsed:
                parsed["reason"] = _normalize_reason_text(parsed["reason"])
            return parsed

        fixed_candidate = fix_chinese_quotes_in_json(candidate)
        if fixed_candidate != candidate:
            parsed = _try_parse(fixed_candidate)
            if isinstance(parsed, dict):
                if "reason" in parsed:
                    parsed["reason"] = _normalize_reason_text(parsed["reason"])
                return parsed

    suitable_match = re.search(
        r'["鈥溾€漖?suitable["鈥溾€漖?\s*:\s*(true|false)',
        raw,
        re.IGNORECASE,
    )
    reason = _extract_reason_from_text(json_candidate or raw)
    if suitable_match is None or reason is None:
        msg = f"鏃犳硶瑙ｆ瀽 LLM 鍝嶅簲涓鸿瘎浼扮粨鏋?JSON: {response}"
        raise ValueError(msg)

    return {
        "suitable": suitable_match.group(1).lower() == "true",
        "reason": _normalize_reason_text(reason),
    }


def normalize_expression_runtime_config(
    raw_value: Any,
    *,
    default: ExpressionRuntimeConfig | None = None,
) -> ExpressionRuntimeConfig:
    fallback = default or ExpressionRuntimeConfig()
    if raw_value is None:
        return fallback
    if isinstance(raw_value, ExpressionRuntimeConfig):
        return raw_value
    if isinstance(raw_value, bool):
        return ExpressionRuntimeConfig(
            use_expression=raw_value,
            enable_learning=raw_value,
        )
    if isinstance(raw_value, Mapping):
        use_expression = raw_value.get("use_expression", fallback.use_expression)
        enable_learning = raw_value.get("enable_learning", fallback.enable_learning)
        return ExpressionRuntimeConfig(
            use_expression=bool(use_expression),
            enable_learning=bool(enable_learning),
        )
    if isinstance(raw_value, Sequence) and not isinstance(raw_value, str):
        values = list(raw_value)
        if len(values) >= 2:
            return ExpressionRuntimeConfig(
                use_expression=bool(values[0]),
                enable_learning=bool(values[1]),
            )
        if len(values) == 1:
            return ExpressionRuntimeConfig(
                use_expression=bool(values[0]),
                enable_learning=bool(values[0]),
            )
    return fallback


def normalize_expression_scope(
    session_id: str,
    raw_value: Any,
) -> tuple[set[str], bool]:
    related_session_ids = {session_id} if session_id else set()
    has_global_share = False
    if raw_value is None:
        return related_session_ids, has_global_share

    raw_session_ids = raw_value
    if (
        isinstance(raw_value, tuple)
        and len(raw_value) == 2
        and isinstance(raw_value[1], bool)
    ):
        raw_session_ids, has_global_share = raw_value

    related_session_ids.update(_normalize_session_id_set(raw_session_ids))
    return related_session_ids, bool(has_global_share)


def parse_expression_response(
    response: str,
) -> list[tuple[str, str, str]]:
    if not response:
        return []

    raw = _strip_markdown_code_fence(response)

    parsed = _try_parse(raw)
    if parsed is None:
        fixed = fix_chinese_quotes_in_json(raw)
        parsed = _try_parse(fixed)
    if parsed is None:
        return []

    parsed_list = _extract_expression_items(parsed)
    if not parsed_list:
        return []

    expressions: list[tuple[str, str, str]] = []
    seen_keys: set[tuple[str, str, str]] = set()
    for item in parsed_list:
        if not isinstance(item, dict):
            continue

        situation = str(item.get("situation", "")).strip()
        style = str(item.get("style", "")).strip()
        source_id = str(item.get("source_id", "")).strip()
        if not situation or not style or not source_id:
            continue
        expression = (situation, style, source_id)
        if expression in seen_keys:
            continue
        seen_keys.add(expression)
        expressions.append(expression)

    return expressions


def _extract_expression_items(parsed: Any) -> list[Any]:
    if isinstance(parsed, list):
        return parsed
    if not isinstance(parsed, dict):
        return []

    for key in ("expressions", "items", "data"):
        candidate = parsed.get(key)
        if isinstance(candidate, list):
            return candidate
    return [parsed]


def _normalize_session_id_set(raw_value: Any) -> set[str]:
    if raw_value is None:
        return set()
    if isinstance(raw_value, str):
        normalized_value = raw_value.strip()
        return {normalized_value} if normalized_value else set()
    if isinstance(raw_value, Mapping):
        return set()
    if isinstance(raw_value, Sequence):
        normalized_session_ids: set[str] = set()
        for item in raw_value:
            normalized_value = str(item or "").strip()
            if normalized_value:
                normalized_session_ids.add(normalized_value)
        return normalized_session_ids
    try:
        iterator = iter(raw_value)
    except TypeError:
        normalized_value = str(raw_value or "").strip()
        return {normalized_value} if normalized_value else set()

    normalized_session_ids: set[str] = set()
    for item in iterator:
        normalized_value = str(item or "").strip()
        if normalized_value:
            normalized_session_ids.add(normalized_value)
    return normalized_session_ids


def _try_parse(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        try:
            repaired = _normalize_repair_json_result(repair_json(text))
            return json.loads(repaired)
        except Exception:
            return None


__all__ = [
    "clean_text",
    "dump_json_list",
    "ExpressionRuntimeConfig",
    "fix_chinese_quotes_in_json",
    "load_json_list",
    "normalize_expression_runtime_config",
    "normalize_expression_scope",
    "parse_evaluation_response",
    "parse_expression_response",
    "strip_json_code_fence",
    "weighted_sample",
]
