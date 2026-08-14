from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from contextlib import suppress

logger = logging.getLogger(__name__)

SessionClearer = Callable[[str], Awaitable[None]]
_session_clearers: list[SessionClearer] = []


def register_session_clearer(clearer: SessionClearer) -> None:
    """Register a derived-state clearer owned by a live agent service."""
    if clearer not in _session_clearers:
        _session_clearers.append(clearer)


def unregister_session_clearer(clearer: SessionClearer) -> None:
    with suppress(ValueError):
        _session_clearers.remove(clearer)


async def clear_runtime_session(session_id: str) -> None:
    """Clear every registered cache/store view derived from one QQ session."""
    for clearer in tuple(_session_clearers):
        try:
            await clearer(session_id)
        except Exception:
            logger.exception("Failed to clear derived state for %s", session_id)
