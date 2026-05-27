from __future__ import annotations

import base64
import contextlib
import io
import os
from datetime import UTC, datetime
from functools import cache
from typing import TYPE_CHECKING, Any, cast

import aiofiles
import aiohttp
import anyio
import imagehash
from anyio import to_thread
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda
from langchain_openai import ChatOpenAI
from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, SecretStr
from sqlalchemy import DateTime, Integer, Text, delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.pool import NullPool

from agent.prompts.image import (
    IMAGE_SYSTEM_PROMPT,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

load_dotenv()

MAX_JPG_SIZE_KB = 512
USER_AGENT = "Mozilla/5.0"
IMAGE_DB_URL = os.getenv(
    "IMAGE_DB_URL", "sqlite+aiosqlite:///./.database/image_cache.db"
)
MAX_RECORDS = 100
PHASH_DISTANCE = 5


class ImageReadResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    base64: str
    phash: imagehash.ImageHash


class CacheEntry(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    record_id: int
    base64: str
    phash: imagehash.ImageHash
    content: str | None
    created_at: datetime
    updated_at: datetime
    accessed_at: datetime


class Base(DeclarativeBase):
    pass


class Record(Base):
    __tablename__ = "image_cache"

    record_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    base64: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    phash: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    accessed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )


class ImageCache:
    def __init__(
        self, db_url: str = IMAGE_DB_URL, max_records: int = MAX_RECORDS
    ) -> None:
        if max_records < 1:
            raise ValueError("max_records must be positive")
        kw = {"poolclass": NullPool} if db_url.startswith("sqlite") else {}
        self.engine: AsyncEngine = create_async_engine(db_url, **kw)
        self.sessionmaker = async_sessionmaker(self.engine, expire_on_commit=False)
        self.max_records = max_records
        self._ready = False
        self._lock = anyio.Lock()

    async def _ensure(self) -> None:
        if self._ready:
            return
        async with self._lock:
            if self._ready:
                return
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            self._ready = True

    @staticmethod
    def _from_record(r: Record) -> CacheEntry:
        return CacheEntry(
            record_id=r.record_id,
            base64=r.base64,
            phash=imagehash.hex_to_hash(r.phash),
            content=r.content,
            created_at=r.created_at,
            updated_at=r.updated_at,
            accessed_at=r.accessed_at,
        )

    async def _touch(self, ids: list[int]) -> None:
        if not ids:
            return
        now = datetime.now(UTC)
        async with self.sessionmaker() as session:
            await session.execute(
                update(Record).where(Record.record_id.in_(ids)).values(accessed_at=now)
            )
            await session.commit()

    async def prune(self) -> int:
        await self._ensure()
        async with self.sessionmaker() as session:
            total = int(
                (
                    await session.execute(select(func.count()).select_from(Record))
                ).scalar_one()
            )
            overflow = total - self.max_records
            if overflow <= 0:
                return 0
            ids = list(
                (
                    await session.execute(
                        select(Record.record_id)
                        .order_by(Record.accessed_at.asc())
                        .limit(overflow)
                    )
                ).scalars()
            )
            if not ids:
                return 0
            await session.execute(delete(Record).where(Record.record_id.in_(ids)))
            await session.commit()
            return len(ids)

    async def add(
        self,
        *,
        base64: str,
        phash: imagehash.ImageHash | str,
        content: str | None = None,
    ) -> CacheEntry:
        await self._ensure()
        now = datetime.now(UTC)
        rec = Record(
            base64=base64,
            phash=str(phash),
            content=content,
            created_at=now,
            updated_at=now,
            accessed_at=now,
        )
        async with self.sessionmaker() as session:
            session.add(rec)
            await session.commit()
            await session.refresh(rec)
            entry = self._from_record(rec)
        await self.prune()
        return entry

    async def find_by_base64(self, base64: str) -> list[CacheEntry]:
        await self._ensure()
        async with self.sessionmaker() as session:
            rows = list(
                (
                    await session.execute(
                        select(Record)
                        .where(Record.base64 == base64)
                        .order_by(Record.accessed_at.desc())
                    )
                ).scalars()
            )
        await self._touch([r.record_id for r in rows])
        return [self._from_record(r) for r in rows]

    async def find_similar(
        self,
        phash: imagehash.ImageHash | str,
        limit: int | None = None,
    ) -> list[CacheEntry]:
        await self._ensure()
        target = imagehash.hex_to_hash(phash) if isinstance(phash, str) else phash
        async with self.sessionmaker() as session:
            rows = list((await session.execute(select(Record))).scalars())

        hits = [
            r for r in rows if imagehash.hex_to_hash(r.phash) - target <= PHASH_DISTANCE
        ]
        hits.sort(
            key=lambda r: (imagehash.hex_to_hash(r.phash) - target, r.accessed_at)
        )
        if limit:
            hits = hits[:limit]
        await self._touch([r.record_id for r in hits])
        return [self._from_record(r) for r in hits]

    async def update_by_base64(self, base64: str, *, content: str | None = None) -> int:
        await self._ensure()
        now = datetime.now(UTC)
        vals: dict[str, Any] = {"updated_at": now, "accessed_at": now}
        if content is not None:
            vals["content"] = content
        async with self.sessionmaker() as session:
            result = await session.execute(
                update(Record).where(Record.base64 == base64).values(**vals)
            )
            await session.commit()
            return int(getattr(result, "rowcount", 0) or 0)

    async def close(self) -> None:
        await self.engine.dispose()


async def read_image(path: str, url: str | None = None) -> ImageReadResult | None:
    raw: bytes | None = None
    if path:
        with contextlib.suppress(Exception):
            async with aiofiles.open(path, "rb") as f:
                raw = await f.read()
    if raw is None and url:
        with contextlib.suppress(Exception):
            async with (
                aiohttp.ClientSession(
                    headers={"User-Agent": USER_AGENT},
                    timeout=aiohttp.ClientTimeout(total=20),
                ) as session,
                session.get(url) as resp,
            ):
                if resp.status < 400:  # noqa: PLR2004
                    raw = await resp.read()
    if raw is None:
        return None

    def _convert() -> ImageReadResult | None:
        try:
            max_bytes = MAX_JPG_SIZE_KB * 1024
            with Image.open(io.BytesIO(raw)) as orig:
                img: Image.Image = ImageOps.exif_transpose(orig)
                with contextlib.suppress(Exception):
                    img.seek(0)
                if img.mode in {"RGBA", "LA", "P"} or "transparency" in img.info:
                    rgba = img.convert("RGBA")
                    bg = Image.new("RGB", rgba.size, (255, 255, 255))
                    bg.paste(rgba, mask=rgba.getchannel("A"))
                    img = bg
                else:
                    img = img.convert("RGB")
                phash = imagehash.phash(img)
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
                            return ImageReadResult(
                                base64=base64.b64encode(jpg).decode(),
                                phash=phash,
                            )
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

    return await to_thread.run_sync(_convert)


def _norm_phash(v: Any) -> imagehash.ImageHash | None:
    if isinstance(v, imagehash.ImageHash):
        return v
    if isinstance(v, str) and v:
        with contextlib.suppress(Exception):
            return imagehash.hex_to_hash(v)
    return None


@cache
def get_analyzer(
    use_cache: bool = True,
    *,
    add_memes_hook: Callable[
        [list[str], list[str], list[imagehash.ImageHash]], Awaitable[Any]
    ]
    | None = None,
) -> Runnable[dict[str, Any], str]:
    def _prepare(x: dict[str, Any]) -> dict[str, Any]:
        img = x.get("image")
        if not img or not isinstance(img, str):
            raise ValueError("Invalid image input")
        phash = x.get("phash")
        if use_cache and phash is None:
            raise ValueError("phash is required when use_cache=True")
        return {
            **x,
            "image": img,
            "phash": _norm_phash(phash),
            "as_meme": bool(x.get("as_meme", False)),
        }

    def _compact(text: str) -> str:
        return " ".join(text.strip().split())

    llm = ChatOpenAI(
        model="kimi-k2.6",
        api_key=SecretStr(os.getenv("KIMI_API_KEY", "")),
        base_url=os.getenv("KIMI_BASE_URL"),
        temperature=0.6,
        max_retries=2,
        extra_body={
            "thinking": {
                "type": "disabled",
            }
        },
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                IMAGE_SYSTEM_PROMPT,
            ),
            (
                "human",
                [
                    {"type": "image", "base64": "{image}", "mime_type": "image/jpeg"},
                    {"type": "text", "text": "请按要求分析这张图片并给出描述。"},
                ],
            ),
        ]
    )

    chain = (
        RunnableLambda(_prepare)
        | prompt
        | llm
        | StrOutputParser()
        | RunnableLambda(_compact)
    )

    if not use_cache:
        return chain

    cache = ImageCache(max_records=MAX_RECORDS)

    async def _lookup(x: dict[str, Any]) -> dict[str, Any]:
        b64 = str(x["image"])
        content: str | None = None
        exact = await cache.find_by_base64(b64)
        for e in exact:
            if e.content:
                content = e.content
                break

        phash = x.get("phash")
        if content is None and isinstance(phash, imagehash.ImageHash):
            for e in await cache.find_similar(phash, limit=10):
                if e.content:
                    content = e.content
                    break

        if (
            x.get("as_meme")
            and content
            and isinstance(phash, imagehash.ImageHash)
            and add_memes_hook
        ):
            await add_memes_hook([b64], [content], [phash])

        return {**x, "cache_hit": content, "has_exact": bool(exact)}

    async def _analyze(x: dict[str, Any]) -> str:
        text = await chain.ainvoke(x)
        phash = x.get("phash")
        if isinstance(phash, imagehash.ImageHash):
            if x.get("has_exact"):
                await cache.update_by_base64(str(x["image"]), content=text)
            else:
                await cache.add(base64=str(x["image"]), phash=phash, content=text)
        if (
            x.get("as_meme")
            and isinstance(phash, imagehash.ImageHash)
            and add_memes_hook
        ):
            await add_memes_hook([str(x["image"])], [text], [phash])
        return text

    return (
        RunnableLambda(_prepare)
        | RunnableLambda(_lookup)
        | RunnableBranch(
            (
                lambda x: isinstance(cast("dict", x).get("cache_hit"), str),
                RunnableLambda(lambda x: str(cast("dict", x)["cache_hit"])),
            ),
            RunnableLambda(_analyze),
        )
    )
