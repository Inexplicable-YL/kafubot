from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import aiofiles  # type: ignore[import-untyped]
import aiohttp
import anyio
import chromadb
import imagehash
import numpy as np
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from PIL import Image, ImageOps
from pydantic import SecretStr
from sqlalchemy import DateTime, Integer, Text, delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.pool import NullPool

from chat.prompt import (
    IMAGE_BRIEF_SYSTEM_PROMPT,
    IMAGE_DETAIL_SYSTEM_PROMPT,
)

load_dotenv()

MAX_JPG_SIZE_KB = 512
USER_AGENT = "Mozilla/5.0"
IMAGE_ANALYSIS_CACHE_DB_URL = os.getenv(
    "IMAGE_ANALYSIS_CACHE_DB_URL",
    "sqlite+aiosqlite:///./image_analysis_cache.db",
)
IMAGE_ANALYSIS_CACHE_TABLE = os.getenv(
    "IMAGE_ANALYSIS_CACHE_TABLE",
    "image_analysis_cache",
)
IMAGE_ANALYSIS_CACHE_MAX_RECORDS = int(
    os.getenv("IMAGE_ANALYSIS_CACHE_MAX_RECORDS", "1000")
)
IMAGE_ANALYSIS_CACHE_PHASH_DISTANCE = int(
    os.getenv("IMAGE_ANALYSIS_CACHE_PHASH_DISTANCE", "5")
)

CHROMA_PATH = os.getenv("CHROMA_PATH", "./.meme_vectordb")

persistent_client = chromadb.PersistentClient(path=CHROMA_PATH)
_vectorstore: Chroma | None = None


def get_vectorstore() -> Chroma:
    global _vectorstore  # noqa: PLW0603
    if _vectorstore is None:
        _vectorstore = Chroma(
            client=persistent_client,
            collection_name="meme_analysis",
            embedding_function=OpenAIEmbeddings(
                model="text-embedding-3-large", base_url=os.getenv("OPENAI_BASE_URL")
            ),
        )
    return _vectorstore


@dataclass(frozen=True)
class ImageReadResult:
    base64: str
    phash: imagehash.ImageHash


@dataclass(frozen=True)
class ImageAnalysisCacheEntry:
    record_id: int
    base64: str
    phash: imagehash.ImageHash
    brief: str | None
    detail: str | None
    created_at: datetime
    updated_at: datetime
    accessed_at: datetime


class ImageAnalysisBase(DeclarativeBase):
    pass


class ImageAnalysisRecord(ImageAnalysisBase):
    __tablename__ = IMAGE_ANALYSIS_CACHE_TABLE

    record_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    base64: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    phash: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    brief: Mapped[str | None] = mapped_column(Text, nullable=True)
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    accessed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )


class ImageAnalysisCache:
    def __init__(
        self,
        *,
        db_url: str = IMAGE_ANALYSIS_CACHE_DB_URL,
        max_records: int = 1000,
    ) -> None:
        if max_records < 1:
            raise ValueError("max_records must be positive")
        engine_kwargs = {"poolclass": NullPool} if db_url.startswith("sqlite") else {}
        self.engine: AsyncEngine = create_async_engine(db_url, **engine_kwargs)
        self.sessionmaker = async_sessionmaker(self.engine, expire_on_commit=False)
        self.max_records = max_records
        self._schema_ready = False
        self._schema_lock = asyncio.Lock()

    @staticmethod
    def _phash_to_text(phash: imagehash.ImageHash | str) -> str:
        return str(phash)

    @staticmethod
    def _entry_from_record(record: ImageAnalysisRecord) -> ImageAnalysisCacheEntry:
        return ImageAnalysisCacheEntry(
            record_id=record.record_id,
            base64=record.base64,
            phash=imagehash.hex_to_hash(record.phash),
            brief=record.brief,
            detail=record.detail,
            created_at=record.created_at,
            updated_at=record.updated_at,
            accessed_at=record.accessed_at,
        )

    async def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        async with self._schema_lock:
            if self._schema_ready:
                return
            async with self.engine.begin() as conn:
                await conn.run_sync(ImageAnalysisBase.metadata.create_all)
            self._schema_ready = True

    async def _touch_records(self, record_ids: list[int]) -> None:
        if not record_ids:
            return
        now = datetime.now(UTC)
        async with self.sessionmaker() as session:
            await session.execute(
                update(ImageAnalysisRecord)
                .where(ImageAnalysisRecord.record_id.in_(record_ids))
                .values(accessed_at=now)
            )
            await session.commit()

    async def prune_lru(self) -> int:
        await self._ensure_schema()
        async with self.sessionmaker() as session:
            count_result = await session.execute(
                select(func.count()).select_from(ImageAnalysisRecord)
            )
            record_count = int(count_result.scalar_one())
            overflow = record_count - self.max_records
            if overflow <= 0:
                return 0

            ids_result = await session.execute(
                select(ImageAnalysisRecord.record_id)
                .order_by(ImageAnalysisRecord.accessed_at.asc())
                .limit(overflow)
            )
            record_ids = list(ids_result.scalars())
            if not record_ids:
                return 0

            await session.execute(
                delete(ImageAnalysisRecord).where(
                    ImageAnalysisRecord.record_id.in_(record_ids)
                )
            )
            await session.commit()
            return len(record_ids)

    async def add(
        self,
        *,
        base64: str,
        phash: imagehash.ImageHash | str,
        brief: str | None = None,
        detail: str | None = None,
    ) -> ImageAnalysisCacheEntry:
        await self._ensure_schema()
        now = datetime.now(UTC)
        record = ImageAnalysisRecord(
            base64=base64,
            phash=self._phash_to_text(phash),
            brief=brief,
            detail=detail,
            created_at=now,
            updated_at=now,
            accessed_at=now,
        )
        async with self.sessionmaker() as session:
            session.add(record)
            await session.commit()
            await session.refresh(record)
            entry = self._entry_from_record(record)
        await self.prune_lru()
        return entry

    async def find_by_base64(self, base64: str) -> list[ImageAnalysisCacheEntry]:
        await self._ensure_schema()
        async with self.sessionmaker() as session:
            result = await session.execute(
                select(ImageAnalysisRecord)
                .where(ImageAnalysisRecord.base64 == base64)
                .order_by(ImageAnalysisRecord.accessed_at.desc())
            )
            records = list(result.scalars())
        await self._touch_records([record.record_id for record in records])
        return [self._entry_from_record(record) for record in records]

    async def find_similar_by_phash(
        self,
        phash: imagehash.ImageHash | str,
        *,
        max_distance: int = 5,
        limit: int | None = None,
    ) -> list[ImageAnalysisCacheEntry]:
        await self._ensure_schema()
        target = imagehash.hex_to_hash(phash) if isinstance(phash, str) else phash
        async with self.sessionmaker() as session:
            result = await session.execute(select(ImageAnalysisRecord))
            records = list(result.scalars())

        similar_records = [
            record
            for record in records
            if imagehash.hex_to_hash(record.phash) - target <= max_distance
        ]
        similar_records.sort(
            key=lambda record: (
                imagehash.hex_to_hash(record.phash) - target,
                record.accessed_at,
            )
        )
        if limit is not None:
            similar_records = similar_records[:limit]
        await self._touch_records([record.record_id for record in similar_records])
        return [self._entry_from_record(record) for record in similar_records]

    async def update_by_base64(
        self,
        base64: str,
        *,
        brief: str | None = None,
        detail: str | None = None,
    ) -> int:
        await self._ensure_schema()
        now = datetime.now(UTC)
        values: dict[str, str | datetime | None] = {
            "updated_at": now,
            "accessed_at": now,
        }
        if brief is not None:
            values["brief"] = brief
        if detail is not None:
            values["detail"] = detail
        async with self.sessionmaker() as session:
            result = await session.execute(
                update(ImageAnalysisRecord)
                .where(ImageAnalysisRecord.base64 == base64)
                .values(**values)
            )
            await session.commit()
            return int(getattr(result, "rowcount", 0) or 0)

    async def close(self) -> None:
        await self.engine.dispose()


async def read_image(path: str, url: str | None = None) -> ImageReadResult | None:
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

    def convert() -> ImageReadResult | None:
        try:
            max_bytes = MAX_JPG_SIZE_KB * 1024
            with Image.open(io.BytesIO(raw)) as original_img:
                img: Image.Image = ImageOps.exif_transpose(original_img)
                with contextlib.suppress(Exception):
                    img.seek(0)
                if img.mode in {"RGBA", "LA", "P"} or "transparency" in img.info:
                    rgba = img.convert("RGBA")
                    bg = Image.new("RGB", rgba.size, (255, 255, 255))
                    bg.paste(rgba, mask=rgba.getchannel("A"))
                    img = bg
                else:
                    img = img.convert("RGB")
                perceptual_hash = imagehash.dhash(img)
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
                                base64=base64.b64encode(jpg).decode("utf-8"),
                                phash=perceptual_hash,
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

    return await asyncio.to_thread(convert)


def _normalize_phash(value: Any) -> imagehash.ImageHash | None:
    if isinstance(value, imagehash.ImageHash):
        return value
    if isinstance(value, str) and value:
        with contextlib.suppress(Exception):
            return imagehash.hex_to_hash(value)
    return None


def get_image_analyzer(
    use_cache: bool = True,
) -> Runnable[dict[str, Any], str]:
    def _prepare_input(x: dict[str, Any]) -> dict[str, Any]:
        image = x.get("image")
        if not image or not isinstance(image, str):
            raise ValueError("Invalid image input")
        if image.startswith("data:image/"):
            image = image.split(",", 1)[1]
        phash = x.get("phash")
        if use_cache and phash is None:
            raise ValueError("phash is required when use_cache=True")
        return {
            "image": image,
            "phash": _normalize_phash(phash),
            "detail": bool(x.get("detail", False)),
        }

    def _compact_output(text: str) -> str:
        return " ".join(text.strip().split())

    def _is_cache_hit(x: dict[str, Any]) -> bool:
        return isinstance(x.get("cache_hit"), str)

    def _return_cache_hit(x: dict[str, Any]) -> str:
        return str(x["cache_hit"])

    brief_llm = ChatOpenAI(
        model="kimi-k2.6",
        api_key=SecretStr(os.getenv("KIMI_API_KEY", "")),
        base_url=os.getenv("KIMI_BASE_URL"),
        temperature=1,
        max_retries=2,
    )
    detail_llm = ChatOpenAI(
        model="kimi-k2.6",
        api_key=SecretStr(os.getenv("KIMI_API_KEY", "")),
        base_url=os.getenv("KIMI_BASE_URL"),
        temperature=1,
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
    analyzer_chain = RunnableBranch[dict[str, Any], str](
        (lambda x: bool(x.get("detail", False)), detail_chain),
        brief_chain,
    )

    if not use_cache:
        return analyzer_chain

    cache = ImageAnalysisCache(max_records=IMAGE_ANALYSIS_CACHE_MAX_RECORDS)

    async def _lookup_cache(x: dict[str, Any]) -> dict[str, Any]:
        target_field = "detail" if bool(x.get("detail", False)) else "brief"
        cached_text: str | None = None

        base64_text = str(x["image"])
        exact_entries = await cache.find_by_base64(base64_text)
        for entry in exact_entries:
            value = getattr(entry, target_field)
            if value:
                cached_text = value
                break

        phash = x.get("phash")
        if cached_text is None and isinstance(phash, imagehash.ImageHash):
            similar_entries = await cache.find_similar_by_phash(
                phash,
                max_distance=IMAGE_ANALYSIS_CACHE_PHASH_DISTANCE,
                limit=10,
            )
            for entry in similar_entries:
                value = getattr(entry, target_field)
                if value:
                    cached_text = value
                    break

        return {
            **x,
            "cache_hit": cached_text,
            "cache_has_exact_base64": bool(exact_entries),
        }

    async def _analyze_and_cache(x: dict[str, Any]) -> str:
        abstract = await analyzer_chain.ainvoke(x)
        phash = x.get("phash")
        if isinstance(phash, imagehash.ImageHash):
            target_field = "detail" if bool(x.get("detail", False)) else "brief"
            values = {target_field: abstract}
            if bool(x.get("cache_has_exact_base64")):
                await cache.update_by_base64(str(x["image"]), **values)
            else:
                await cache.add(
                    base64=str(x["image"]),
                    phash=phash,
                    **values,
                )
        await add_memes([str(x["image"])], [abstract])
        return abstract

    return (
        RunnableLambda(_prepare_input)
        | RunnableLambda(_lookup_cache)
        | RunnableBranch(
            (_is_cache_hit, RunnableLambda(_return_cache_hit)),
            RunnableLambda(_analyze_and_cache),
        )
    )


async def add_memes(
    base64s: list[str],
    analyses: list[str],
) -> None:
    if len(base64s) != len(analyses):
        raise ValueError("base64s and analyses must have the same length")
    base64s = [b if b.startswith("base64://") else "base64://" + b for b in base64s]

    await get_vectorstore().aadd_texts(
        texts=analyses,
        metadatas=[{"base64": base64} for base64 in base64s],
    )


@dataclass
class SearchResult:
    base64: str
    analysis: str


async def meme_analysis(files: list[str]) -> list[tuple[str, SearchResult]]:
    image_analyzer = get_image_analyzer(use_cache=False)
    analysis_results: list[tuple[str, SearchResult]] = []
    failed_files: list[str] = []
    semaphore = anyio.Semaphore(5)

    async def _handle_url(file: str) -> None:
        async with semaphore:
            try:
                if file.startswith(("http://", "https://")):
                    result = await read_image(
                        path="",
                        url=file,
                    )
                else:
                    result = await read_image(
                        path=file,
                        url=None,
                    )
                if result is None:
                    failed_files.append(file)
                    return
                analysis = await image_analyzer.ainvoke(
                    {
                        "image": result.base64,
                        "phash": result.phash,
                        "detail": True,
                    }
                )
                analysis_results.append(
                    (
                        file,
                        SearchResult(
                            base64="base64://" + result.base64,
                            analysis=analysis,
                        ),
                    )
                )
            except Exception:
                failed_files.append(file)

    async with anyio.create_task_group() as tg:
        for file in files:
            tg.start_soon(_handle_url, file)

    if analysis_results:
        texts = [r[1].analysis for r in analysis_results]
        metadatas = [{"base64": r[1].base64} for r in analysis_results]

        await get_vectorstore().aadd_texts(
            texts=texts,
            metadatas=metadatas,
        )
    if failed_files:
        print(f"以下 {len(failed_files)} 个 FILE 分析失败：{failed_files}")
    return analysis_results


async def search_meme(
    query: str,
    temperature: float = 0.0,
    min_score: float | None = None,
) -> SearchResult | None:
    results = await get_vectorstore().asimilarity_search_with_relevance_scores(
        query, k=20, score_threshold=min_score or 0.0
    )
    print(
        "\n".join(
            f"Score: {score:.4f}, Analysis: {doc.page_content}"
            for doc, score in results
        )
    )
    if not results:
        return None

    if temperature == 0.0:
        best_doc, _ = max(results, key=lambda x: x[1])
    else:
        scores = np.array([s for _, s in results])
        scores = scores - np.max(scores)
        exp_scores = np.exp(scores / temperature)
        probs = exp_scores / np.sum(exp_scores)
        rng = np.random.default_rng()
        chosen_idx = rng.choice(len(results), p=probs)
        best_doc = results[chosen_idx][0]

    return SearchResult(
        base64=best_doc.metadata.get("base64", ""),
        analysis=best_doc.page_content,
    )


if __name__ == "__main__":

    async def main() -> None:
        result = await search_meme("花谱呆萌脸", temperature=0.5)
        print(result.analysis if result else "No result")

    anyio.run(main)
