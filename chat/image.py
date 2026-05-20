from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import os
from datetime import UTC, datetime
from functools import cache
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
from pydantic import BaseModel, ConfigDict, SecretStr
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

IMAGE_DB_URL = os.getenv(
    "IMAGE_DB_URL", "sqlite+aiosqlite:///./.database/image_analysis_cache.db"
)

IMAGE_MAX_RECORDS = int(os.getenv("IMAGE_MAX_RECORDS", "100"))

IMAGE_PHASH_DISTANCE = int(os.getenv("IMAGE_PHASH_DISTANCE", "5"))

MEME_CHROMA_PATH = os.getenv("MEME_CHROMA_PATH", "./.database/meme_vectordb")

MEME_PHASH_DISTANCE = int(os.getenv("MEME_PHASH_DISTANCE", "5"))

persistent_client = chromadb.PersistentClient(path=MEME_CHROMA_PATH)


@cache
def get_vectorstore() -> Chroma:
    return Chroma(
        client=persistent_client,
        collection_name="meme_analysis",
        embedding_function=OpenAIEmbeddings(
            model="text-embedding-3-large", base_url=os.getenv("OPENAI_BASE_URL")
        ),
    )


class ImageReadResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    base64: str
    phash: imagehash.ImageHash


class ImageAnalysisCacheEntry(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    record_id: int
    base64: str
    phash: imagehash.ImageHash
    brief: str | None
    detail: str | None
    created_at: datetime
    updated_at: datetime
    accessed_at: datetime


class MemeResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    base64: str
    analysis: str
    phash: imagehash.ImageHash | None = None


class ImageAnalysisBase(DeclarativeBase):
    pass


class ImageAnalysisRecord(ImageAnalysisBase):
    __tablename__ = "image_analysis_cache"

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
        db_url: str = IMAGE_DB_URL,
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
            if imagehash.hex_to_hash(record.phash) - target <= IMAGE_PHASH_DISTANCE
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
                perceptual_hash = imagehash.phash(img)
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


@cache
def get_image_analyzer(  # noqa: PLR0915
    use_cache: bool = True,
) -> Runnable[dict[str, Any], str]:
    def _prepare_input(x: dict[str, Any]) -> dict[str, Any]:
        image = x.get("image")
        if not image or not isinstance(image, str):
            raise ValueError("Invalid image input")
        phash = x.get("phash")
        if use_cache and phash is None:
            raise ValueError("phash is required when use_cache=True")
        x.update(
            {
                "image": image,
                "phash": _normalize_phash(phash),
                "detail": bool(x.get("detail", False)),
                "as_meme": bool(x.get("as_meme", False)),
            }
        )
        return x

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
        temperature=0.6,
        max_retries=2,
    ).bind(
        extra_body={
            "thinking": {
                "type": "disabled",
            }
        },
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
                    {"type": "text", "text": "请按要求描述图片。"},
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
                    {"type": "text", "text": "请按要求理解并描述图片。"},
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

    cache = ImageAnalysisCache(max_records=IMAGE_MAX_RECORDS)

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
                limit=10,
            )
            for entry in similar_entries:
                value = getattr(entry, target_field)
                if value:
                    cached_text = value
                    break
        if (
            x.get("as_meme", False)
            and cached_text
            and isinstance(phash, imagehash.ImageHash)
        ):
            await add_memes([str(x["image"])], [cached_text], [phash])
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
        if x.get("as_meme", False) and isinstance(phash, imagehash.ImageHash):
            await add_memes([str(x["image"])], [abstract], [phash])
        return abstract

    return (
        RunnableLambda(_prepare_input)
        | RunnableLambda(_lookup_cache)
        | RunnableBranch(
            (_is_cache_hit, RunnableLambda(_return_cache_hit)),
            RunnableLambda(_analyze_and_cache),
        )
    )


def has_similar_meme_phash(
    phash: imagehash.ImageHash,
    *,
    max_distance: int = MEME_PHASH_DISTANCE,
) -> bool:
    collection = get_vectorstore()._collection
    total = collection.count()
    batch_size = 500
    for offset in range(0, total, batch_size):
        result = collection.get(
            limit=batch_size,
            offset=offset,
            include=["metadatas"],
        )
        metadatas = result.get("metadatas") or []
        for metadata_value in metadatas:
            metadata = dict(metadata_value or {})
            existing_phash = metadata.get("phash")
            if not isinstance(existing_phash, str) or not existing_phash:
                continue
            with contextlib.suppress(Exception):
                if imagehash.hex_to_hash(existing_phash) - phash <= max_distance:
                    return True
    return False


async def add_memes(
    base64s: list[str],
    analyses: list[str],
    phashs: list[imagehash.ImageHash],
) -> list[str | None]:
    if len(base64s) != len(analyses) or len(base64s) != len(phashs):
        raise ValueError("base64s, analyses and phashs must have the same length")
    base64s = [b if b.startswith("base64://") else "base64://" + b for b in base64s]

    accepted_indices: list[int] = []
    accepted_phashs: list[imagehash.ImageHash] = []
    ids_by_index: list[str | None] = [None] * len(base64s)
    for index, phash in enumerate(phashs):
        if has_similar_meme_phash(phash):
            continue
        if any(
            existing_phash - phash < MEME_PHASH_DISTANCE
            for existing_phash in accepted_phashs
        ):
            continue
        accepted_indices.append(index)
        accepted_phashs.append(phash)

    if not accepted_indices:
        return ids_by_index

    added_ids = await get_vectorstore().aadd_texts(
        texts=[analyses[index] for index in accepted_indices],
        metadatas=[
            {
                "base64": base64s[index],
                "phash": str(phashs[index]),
                "manually_annotated": False,
            }
            for index in accepted_indices
        ],
    )
    for index, id_ in zip(accepted_indices, added_ids, strict=False):
        ids_by_index[index] = id_
    return ids_by_index


async def meme_analysis(files: list[str]) -> list[tuple[str, MemeResult]]:
    image_analyzer = get_image_analyzer(use_cache=False)
    analysis_results: list[tuple[str, MemeResult]] = []
    failed_files: list[str] = []
    duplicate_files: list[str] = []
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
                if has_similar_meme_phash(result.phash):
                    duplicate_files.append(file)
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
                        MemeResult(
                            base64="base64://" + result.base64,
                            analysis=analysis,
                            phash=result.phash,
                        ),
                    )
                )
            except Exception:
                failed_files.append(file)

    async with anyio.create_task_group() as tg:
        for file in files:
            tg.start_soon(_handle_url, file)

    if analysis_results:
        ready_results: list[tuple[str, MemeResult]] = []
        ready_phashs: list[imagehash.ImageHash] = []
        for file, result in analysis_results:
            if result.phash is None:
                failed_files.append(file)
                continue
            ready_results.append((file, result))
            ready_phashs.append(result.phash)
        added_ids = await add_memes(
            [result.base64 for _, result in ready_results],
            [result.analysis for _, result in ready_results],
            ready_phashs,
        )
        added_results: list[tuple[str, MemeResult]] = []
        for (file, result), id_ in zip(ready_results, added_ids, strict=False):
            if id_ is None:
                duplicate_files.append(file)
                continue
            added_results.append((file, result))
        analysis_results = added_results
    if failed_files:
        print(f"以下 {len(failed_files)} 个 FILE 分析失败：{failed_files}")
    if duplicate_files:
        print(
            f"以下 {len(duplicate_files)} 个 FILE 已存在相似 meme，跳过入库：{duplicate_files}"
        )
    return analysis_results


EXTEND_PAIRS: list[tuple[str, str]] = [
    ("生气", "处于生气、心情不好、恼怒、不开心的状态。"),
    ("生闷气", "角色正在生闷气。"),
]


async def search_meme(
    query: str,
    temperature: float = 0.0,
    min_score: float | None = None,
    *,
    log: bool = False,
) -> MemeResult | None:
    for k, v in EXTEND_PAIRS:
        if k in query:
            query = f"{query}。{v}"
    results = await get_vectorstore()._asimilarity_search_with_relevance_scores(
        query, k=20, filter={"manually_annotated": True}
    )
    candidates = [
        (doc, score)
        for doc, score in results
        if min_score is None or score >= min_score
    ]
    if not candidates:
        return None

    if temperature == 0.0:
        best_doc, _ = max(candidates, key=lambda x: x[1])
    else:
        scores = np.array([s for _, s in candidates])
        low, min_val, max_val = np.float64(1e-5), np.min(scores), np.max(scores)
        if max_val == min_val:
            scores = np.full_like(scores, 0.5)
        else:
            scores = low + (scores - min_val) * (1 - 2 * low) / (max_val - min_val)
        scores = scores - np.max(scores)
        exp_scores = np.exp(scores / temperature)
        probs = exp_scores / np.sum(exp_scores)
        if log:
            print(
                "\n".join(
                    f"S: {doc[1]:.6f}, P: {prob:.6f}, A: {doc[0].page_content}"
                    for doc, prob in zip(candidates, probs, strict=False)
                )
            )
        rng = np.random.default_rng()
        chosen_idx = rng.choice(len(candidates), p=probs)
        best_doc = candidates[chosen_idx][0]

    return MemeResult(
        base64=best_doc.metadata.get("base64", ""),
        analysis=best_doc.page_content,
    )


if __name__ == "__main__":

    async def main() -> None:
        result = await search_meme("害羞", temperature=0.5, min_score=0.05, log=True)
        print(result.analysis if result else "No result")

    anyio.run(main)
