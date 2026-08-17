from __future__ import annotations

import os
from functools import cache
from typing import Literal

from dotenv import load_dotenv
from langchain_deepseek.chat_models import DEFAULT_API_BASE, ChatDeepSeek

load_dotenv()

MODEL_NAME = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-v4-flash")


@cache
def get_thinking_model(
    reasoning_effort: Literal["high", "max"] = "high",
) -> ChatDeepSeek:
    return ChatDeepSeek(
        model=MODEL_NAME,
        api_base=os.getenv("DEEPSEEK_BASE_URL", DEFAULT_API_BASE),
        max_retries=2,
        reasoning_effort=reasoning_effort,
        extra_body={"thinking": {"type": "enabled"}},
    )


@cache
def get_nonthinking_model(temperature: float = 0.8) -> ChatDeepSeek:
    return ChatDeepSeek(
        model=MODEL_NAME,
        api_base=os.getenv("DEEPSEEK_BASE_URL", DEFAULT_API_BASE),
        temperature=temperature,
        max_retries=2,
        extra_body={"thinking": {"type": "disabled"}},
    )
