from __future__ import annotations

import contextlib
import logging
import os
from functools import cache, partial
from typing import TYPE_CHECKING, Any, cast

import anyio
import chromadb
import imagehash
import numpy as np
from anyio.to_thread import run_sync
from chromadb import Where
from chromadb.errors import InternalError
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, ConfigDict

from agent.multimodal.image import get_analyzer, read_image

if TYPE_CHECKING:
    from collections.abc import Callable

load_dotenv()

logger = logging.getLogger(__name__)


MEME_CHROMA_PATH = "./.database/meme_vectordb"

MEME_PHASH_DISTANCE = 5
MEME_SEARCH_CANDIDATE_LIMIT = 20
MEME_EXACT_SEARCH_BATCH_SIZE = 200


@cache
def get_vectorstore() -> Chroma:
    return Chroma(
        client=chromadb.PersistentClient(path=MEME_CHROMA_PATH),
        collection_name="meme_analysis",
        embedding_function=OpenAIEmbeddings(
            model="text-embedding-3-large", base_url=os.getenv("OPENAI_BASE_URL")
        ),
    )


class MemeResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    base64: str
    analysis: str
    phash: imagehash.ImageHash | None = None


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
    image_analyzer = get_analyzer(use_cache=False)
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


def _is_missing_vector_id_error(exc: InternalError) -> bool:
    return "Error finding id" in str(exc)


def _read_embeddings_by_id(
    collection: Any,
    ids: list[str],
) -> dict[str, np.ndarray[Any, np.dtype[np.float32]]]:
    """读取一批向量；索引损坏时递归缩小范围并跳过悬空记录。"""
    if not ids:
        return {}
    try:
        result = collection.get(ids=ids, include=["embeddings"])
    except InternalError as exc:
        if not _is_missing_vector_id_error(exc):
            raise
        if len(ids) == 1:
            return {}
        middle = len(ids) // 2
        return {
            **_read_embeddings_by_id(collection, ids[:middle]),
            **_read_embeddings_by_id(collection, ids[middle:]),
        }

    result_ids = result.get("ids") or []
    embeddings = result.get("embeddings")
    if embeddings is None:
        return {}
    return {
        str(id_): np.asarray(embedding, dtype=np.float32)
        for id_, embedding in zip(result_ids, embeddings, strict=False)
        if embedding is not None
    }


def _exact_similarity_search_with_relevance_scores(
    query_embedding: list[float],
    *,
    k: int,
    metadata_filter: Where,
    relevance_score_fn: Callable[[float], float],
) -> list[tuple[Document, float]]:
    """绕开 HNSW，使用持久层中的可用向量做精确 L2 检索。"""
    collection = get_vectorstore()._collection
    query_vector = np.asarray(query_embedding, dtype=np.float32)
    total = collection.count()
    ranked: list[tuple[float, Document]] = []
    missing_embedding_count = 0

    for offset in range(0, total, MEME_EXACT_SEARCH_BATCH_SIZE):
        result = collection.get(
            limit=MEME_EXACT_SEARCH_BATCH_SIZE,
            offset=offset,
            where=metadata_filter,
            include=["documents", "metadatas"],
        )
        ids = [str(id_) for id_ in result.get("ids") or []]
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        embeddings_by_id = _read_embeddings_by_id(collection, ids)
        missing_embedding_count += len(ids) - len(embeddings_by_id)

        for id_, page_content, metadata in zip(
            ids, documents, metadatas, strict=False
        ):
            embedding = embeddings_by_id.get(id_)
            if embedding is None or page_content is None:
                continue
            if embedding.shape != query_vector.shape:
                missing_embedding_count += 1
                continue
            distance = float(np.sum(np.square(embedding - query_vector)))
            ranked.append(
                (
                    distance,
                    Document(
                        id=id_,
                        page_content=str(page_content),
                        metadata=dict(metadata or {}),
                    ),
                )
            )

    if missing_embedding_count:
        logger.warning(
            "Meme vector index contains %d unreadable embeddings; skipped during "
            "exact-search fallback",
            missing_embedding_count,
        )
    ranked.sort(key=lambda item: item[0])
    return [
        (document, relevance_score_fn(distance))
        for distance, document in ranked[:k]
    ]


async def _similarity_search_with_relevance_scores(
    query: str,
    *,
    k: int = MEME_SEARCH_CANDIDATE_LIMIT,
) -> list[tuple[Document, float]]:
    """单次生成查询向量；HNSW 损坏时自动退回只读精确检索。"""
    vectorstore = get_vectorstore()
    embedding_function = vectorstore.embeddings
    if embedding_function is None:
        raise RuntimeError("Meme vector store has no embedding function")

    query_embedding = await embedding_function.aembed_query(query)
    relevance_score_fn = vectorstore._select_relevance_score_fn()
    try:
        results = await run_sync(
            partial(
                vectorstore.similarity_search_by_vector_with_relevance_scores,
                query_embedding,
                k=k,
                filter=cast("dict[str, str]", {"manually_annotated": True}),
            )
        )
    except InternalError as exc:
        if not _is_missing_vector_id_error(exc):
            raise
        logger.warning(
            "Meme HNSW index has a dangling vector id; using exact-search fallback"
        )
        return await run_sync(
            partial(
                _exact_similarity_search_with_relevance_scores,
                query_embedding,
                k=k,
                metadata_filter={"manually_annotated": True},
                relevance_score_fn=relevance_score_fn,
            )
        )

    return [
        (document, relevance_score_fn(distance))
        for document, distance in results
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
    results = await _similarity_search_with_relevance_scores(query)
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


async def search_memes(
    query: str,
    limit: int = 10,
    min_score: float | None = None,
) -> list[MemeResult] | None:
    for k, v in EXTEND_PAIRS:
        if k in query:
            query = f"{query}。{v}"
    results = await _similarity_search_with_relevance_scores(query)
    candidates = [
        (doc, score)
        for doc, score in results
        if min_score is None or score >= min_score
    ]
    if not candidates:
        return None

    return [
        MemeResult(
            base64=doc.metadata.get("base64", ""),
            analysis=doc.page_content,
        )
        for doc, _ in candidates[:limit]
    ]


if __name__ == "__main__":

    async def main() -> None:
        result = await search_meme("不可以捏", temperature=0.5, min_score=0.0, log=True)
        print(result.analysis if result else "No result")

    anyio.run(main)
