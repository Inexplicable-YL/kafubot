from __future__ import annotations

import argparse
import base64
import contextlib
import io
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

import imagehash
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _iter_records(
    collection: Any, *, batch_size: int
) -> list[tuple[str, dict[str, Any]]]:
    total = collection.count()
    records: list[tuple[str, dict[str, Any]]] = []
    for offset in range(0, total, batch_size):
        result = collection.get(
            limit=batch_size,
            offset=offset,
            include=["metadatas"],
        )
        ids = result.get("ids") or []
        metadatas = result.get("metadatas") or []
        for i, id_ in enumerate(ids):
            metadata = dict(metadatas[i] or {}) if i < len(metadatas) else {}
            records.append((id_, metadata))
    return records


def _phash_from_base64(value: str) -> imagehash.ImageHash | None:
    if value.startswith("base64://"):
        value = value.removeprefix("base64://")
    try:
        raw = base64.b64decode(value)
        with Image.open(io.BytesIO(raw)) as original_img:
            img = ImageOps.exif_transpose(original_img)
            with contextlib.suppress(Exception):
                img.seek(0)
            if img.mode in {"RGBA", "LA", "P"} or "transparency" in img.info:
                rgba = img.convert("RGBA")
                bg = Image.new("RGB", rgba.size, (255, 255, 255))
                bg.paste(rgba, mask=rgba.getchannel("A"))
                img = bg
            else:
                img = img.convert("RGB")
            return imagehash.phash(img)
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill meme phash data for Chroma and image analysis cache."
    )
    parser.add_argument(
        "--chroma-path",
        default=None,
        help="Override CHROMA_PATH for this run.",
    )
    parser.add_argument(
        "--cache-db-url",
        default=None,
        help="Override IMAGE_ANALYSIS_CACHE_DB_URL for this run.",
    )
    parser.add_argument(
        "--cache-table",
        default=None,
        help="Override IMAGE_ANAL for this run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Number of Chroma records to read per batch.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and report changes without writing metadata.",
    )
    return parser.parse_args()


def _sqlite_path(db_url: str) -> Path:
    for prefix in ("sqlite+aiosqlite:///", "sqlite:///"):
        if db_url.startswith(prefix):
            return Path(db_url.removeprefix(prefix))
    raise ValueError(f"unsupported cache db url: {db_url}")


def _backfill_chroma(args: argparse.Namespace) -> None:
    from chat.image import get_vectorstore  # noqa: PLC0415

    collection = get_vectorstore()._collection
    records = _iter_records(collection, batch_size=max(1, args.batch_size))
    updated = 0
    skipped_existing = 0
    missing_base64 = 0
    failed = 0

    for id_, metadata in records:
        if metadata.get("phash"):
            skipped_existing += 1
            continue
        base64_value = metadata.get("base64")
        if not isinstance(base64_value, str) or not base64_value:
            missing_base64 += 1
            continue

        phash = _phash_from_base64(base64_value)
        if phash is None:
            failed += 1
            continue

        metadata["phash"] = str(phash)
        if not args.dry_run:
            collection.update(ids=[id_], metadatas=[metadata])
        updated += 1

    mode = "dry-run" if args.dry_run else "write"
    print(
        f"chroma {mode}: total={len(records)} updated={updated} "
        f"skipped_existing={skipped_existing} missing_base64={missing_base64} "
        f"failed={failed}"
    )


def _backfill_cache(args: argparse.Namespace) -> None:
    from chat.image import (  # noqa: PLC0415
        IMAGE_DB_URL,
    )

    db_url = args.cache_db_url or IMAGE_DB_URL
    table = args.cache_table or "image_analysis_cache"
    db_path = _sqlite_path(db_url)
    if not db_path.exists():
        print(f"cache skipped: db does not exist: {db_path}")
        return

    updated = 0
    unchanged = 0
    missing_base64 = 0
    failed = 0
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            f'SELECT record_id, base64, phash FROM "{table}"'  # noqa: S608
        ).fetchall()
        for record_id, base64_value, old_phash in rows:
            if not isinstance(base64_value, str) or not base64_value:
                missing_base64 += 1
                continue
            phash = _phash_from_base64(base64_value)
            if phash is None:
                failed += 1
                continue

            new_phash = str(phash)
            if new_phash == old_phash:
                unchanged += 1
                continue
            if not args.dry_run:
                conn.execute(
                    f'UPDATE "{table}" SET phash = ? WHERE record_id = ?',  # noqa: S608
                    (new_phash, record_id),
                )
            updated += 1
        if not args.dry_run:
            conn.commit()

    mode = "dry-run" if args.dry_run else "write"
    print(
        f"cache {mode}: total={len(rows)} updated={updated} unchanged={unchanged} "
        f"missing_base64={missing_base64} failed={failed}"
    )


def main() -> None:
    args = parse_args()
    if args.chroma_path:
        os.environ["CHROMA_PATH"] = args.chroma_path
    if args.cache_db_url:
        os.environ["IMAGE_ANALYSIS_CACHE_DB_URL"] = args.cache_db_url
    if args.cache_table:
        os.environ["IMAGE_ANAL"] = args.cache_table

    _backfill_chroma(args)
    _backfill_cache(args)


if __name__ == "__main__":
    main()
