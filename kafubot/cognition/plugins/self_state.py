from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import aiofiles
import anyio

from kafubot.agency.models import ActiveThread, RecentAction, SelfState, StateDelta
from kafubot.cognition.plugins.base import PluginContext, PluginDefinition

if TYPE_CHECKING:
    from collections.abc import Callable

    from kafubot.config import AgentConfig


class SelfStateStore:
    """Atomic persistence for self state, active threads, actions and deltas."""

    def __init__(self, path: str | Path | None, *, recent_action_limit: int = 30):
        self.path = Path(path) if path else None
        self.recent_action_limit = recent_action_limit
        self._state: SelfState | None = None
        self._lock = anyio.Lock()

    async def load(self) -> SelfState:
        async with self._lock:
            if self._state is not None:
                return self._state.model_copy(deep=True)
            if self.path is None or not self.path.exists():
                self._state = SelfState()
            else:
                async with aiofiles.open(self.path, encoding="utf-8") as state_file:
                    self._state = SelfState.model_validate_json(await state_file.read())
            return self._state.model_copy(deep=True)

    async def update(self, operation: Callable[[SelfState], None]) -> SelfState:
        async with self._lock:
            if self._state is None:
                if self.path is not None and self.path.exists():
                    async with aiofiles.open(self.path, encoding="utf-8") as state_file:
                        self._state = SelfState.model_validate_json(
                            await state_file.read()
                        )
                else:
                    self._state = SelfState()
            operation(self._state)
            self._state.recent_actions = self._state.recent_actions[
                -self.recent_action_limit :
            ]
            await self._save_unlocked(self._state)
            return self._state.model_copy(deep=True)

    async def record_reply(self, action: RecentAction) -> SelfState:
        now = datetime.now(UTC)

        def mutate(state: SelfState) -> None:
            state.current_focus = [
                action.session_id,
                *(item for item in state.current_focus if item != action.session_id),
            ][:3]
            previous = state.active_threads.get(action.session_id)
            residue = min(1.0, (previous.attention_residue if previous else 0) + 0.65)
            state.active_threads[action.session_id] = ActiveThread(
                session_id=action.session_id,
                summary=action.behavior,
                attention_residue=residue,
                residue_updated_at=now,
                last_action_at=now,
            )
            state.recent_actions.append(action)
            state.fatigue_by_session[action.session_id] = min(
                1.0,
                state.fatigue_by_session.get(action.session_id, 0) + 0.2,
            )
            state.last_delta = StateDelta(
                summary=f"acted in {action.session_id}: {action.behavior}",
                focus_added=[action.session_id],
                action=action,
            )

        return await self.update(mutate)

    async def record_skip(self, session_id: str, reason: str) -> SelfState:
        def mutate(state: SelfState) -> None:
            state.fatigue_by_session[session_id] = max(
                0,
                state.fatigue_by_session.get(session_id, 0) - 0.05,
            )
            action = RecentAction(
                session_id=session_id,
                behavior=f"read and stayed silent: {reason}",
                expected_effect="avoid unnecessary interruption",
            )
            state.recent_actions.append(action)
            state.last_delta = StateDelta(
                summary=f"skipped {session_id}: {reason}",
                action=action,
            )

        return await self.update(mutate)

    async def clear_session(self, session_id: str) -> SelfState:
        def mutate(state: SelfState) -> None:
            state.current_focus = [
                item for item in state.current_focus if item != session_id
            ]
            state.active_threads.pop(session_id, None)
            state.fatigue_by_session.pop(session_id, None)
            state.last_delta = StateDelta(
                summary=f"cleared {session_id}",
                focus_removed=[session_id],
            )

        return await self.update(mutate)

    async def cool_fatigue(self, factor: float = 0.92) -> SelfState:
        def mutate(state: SelfState) -> None:
            state.fatigue_by_session = {
                key: value * factor
                for key, value in state.fatigue_by_session.items()
                if value * factor >= 0.01
            }

        return await self.update(mutate)

    async def _save_unlocked(self, state: SelfState) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        async with aiofiles.open(temporary_path, "w", encoding="utf-8") as state_file:
            await state_file.write(state.model_dump_json(indent=2))
        temporary_path.replace(self.path)


def apply(context: PluginContext, _config: Any) -> None:
    if context.optional_service("state_store") is not None:
        return
    config = cast("AgentConfig", context.service("config"))
    context.provide(
        "state_store",
        SelfStateStore(
            config.executive.state_file,
            recent_action_limit=config.executive.recent_action_limit,
        ),
    )


plugin = PluginDefinition(name="self_state", apply=apply)
