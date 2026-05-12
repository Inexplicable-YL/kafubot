from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import os
from typing import Any

import aiofiles
import aiohttp
from _prompt import (
    IMAGE_BRIEF_SYSTEM_PROMPT,
    IMAGE_DETAIL_SYSTEM_PROMPT,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda
from langchain_openai import ChatOpenAI
from PIL import Image, ImageOps

MAX_JPG_SIZE_KB = 512
USER_AGENT = "Mozilla/5.0"


async def read_image_as_base64(path: str, url: str | None = None) -> str | None:
    raw: bytes | None = None
    if path:
        try:
            async with aiofiles.open(path, "rb") as f:
                raw = await f.read()
        except Exception:
            raw = None
    if raw is None and url:
        try:
            async with (
                aiohttp.ClientSession(
                    headers={"User-Agent": USER_AGENT},
                    timeout=aiohttp.ClientTimeout(total=20),
                ) as session,
                session.get(url) as resp,
            ):
                if resp.status < 400:  # noqa: PLR2004
                    raw = await resp.read()
        except Exception:
            raw = None
    if raw is None:
        return None

    def convert() -> str | None:
        try:
            max_bytes = MAX_JPG_SIZE_KB * 1024
            with Image.open(io.BytesIO(raw)) as img:
                img = ImageOps.exif_transpose(img)
                with contextlib.suppress(Exception):
                    img.seek(0)
                if img.mode in {"RGBA", "LA", "P"} or "transparency" in img.info:
                    rgba = img.convert("RGBA")
                    bg = Image.new("RGB", rgba.size, (255, 255, 255))
                    bg.paste(rgba, mask=rgba.getchannel("A"))
                    img = bg
                else:
                    img = img.convert("RGB")
                while True:
                    best: bytes | None = None

                    for quality in range(95, 14, -5):
                        buf = io.BytesIO()
                        img.save(
                            buf,
                            "JPEG",
                            quality=quality,
                            optimize=True,
                            progressive=True,
                        )
                        jpg = buf.getvalue()
                        if best is None or len(jpg) < len(best):
                            best = jpg
                        if len(jpg) <= max_bytes:
                            return base64.b64encode(jpg).decode("utf-8")
                    if best is None:
                        return None
                    w, h = img.size
                    if w <= 1 or h <= 1:
                        return None
                    scale = min(0.9, max(0.35, (max_bytes / len(best)) ** 0.5 * 0.9))
                    img = img.resize(
                        (max(1, int(w * scale)), max(1, int(h * scale))),
                        Image.Resampling.LANCZOS,
                    )
        except Exception:
            return None

    return await asyncio.to_thread(convert)


def get_image_analyzer() -> Runnable[dict[str, Any], str]:
    def _prepare_input(x: dict[str, Any]) -> dict[str, Any]:
        image = x.get("image")
        if not image or not isinstance(image, str):
            raise ValueError("Invalid image input")
        if image.startswith("data:image/"):
            image = image.split(",", 1)[1]
        return {"image": image, "detail": bool(x.get("detail", False))}

    def _compact_output(text: str) -> str:
        return " ".join(text.strip().split())

    brief_llm = ChatOpenAI(
        model="gpt-4.1-mini",
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=0.3,
        max_retries=2,
    )
    detail_llm = ChatOpenAI(
        model="gpt-5.4-mini",
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=0.3,
        max_retries=2,
    )

    brief_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                IMAGE_BRIEF_SYSTEM_PROMPT,
            ),
            (
                "human",
                [
                    {"type": "image", "base64": "{image}", "mime_type": "image/jpeg"},
                ],
            ),
        ]
    )

    detail_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                IMAGE_DETAIL_SYSTEM_PROMPT,
            ),
            (
                "human",
                [
                    {"type": "image", "base64": "{image}", "mime_type": "image/jpeg"},
                ],
            ),
        ]
    )

    prepare = RunnableLambda(_prepare_input)
    parser = StrOutputParser() | RunnableLambda(_compact_output)

    brief_chain = prepare | brief_prompt | brief_llm | parser
    detail_chain = prepare | detail_prompt | detail_llm | parser

    return RunnableBranch(
        (lambda x: bool(x["detail"]), detail_chain),
        brief_chain,
    )
