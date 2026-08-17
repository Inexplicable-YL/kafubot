from __future__ import annotations

from typing import TYPE_CHECKING

import anyio

from .models import ActionContract, CompiledContext
from .providers import ProviderQuery, WorldModelHub

if TYPE_CHECKING:
    from kafubot.config import ExecutiveConfig

    from .environment import SocialEnvironment


class ContextCompilationError(ValueError):
    pass


class ContextCompiler:
    """Compiles only the local evidence required by an Action Contract."""

    def __init__(
        self,
        environment: SocialEnvironment,
        providers: WorldModelHub,
        config: ExecutiveConfig,
    ) -> None:
        self.environment = environment
        self.providers = providers
        self.config = config

    async def compile(self, contract: ActionContract) -> CompiledContext:
        messages = await self.environment.read(
            contract.target_session_id,
            limit=self.config.context_message_limit,
        )
        user_messages = [item for item in messages if item.role == "user"]
        by_id = {item.message_id: item for item in user_messages if item.message_id}
        evidence_ids = list(dict.fromkeys(contract.evidence_message_ids))
        if not evidence_ids:
            raise ContextCompilationError("at least one evidence_message_id is required")
        for message_id in evidence_ids:
            if message_id in by_id:
                continue
            message = await self.environment.find_message(
                contract.target_session_id,
                message_id,
            )
            if message is not None and message.role == "user":
                by_id[message_id] = message
                messages.append(message)
        messages.sort(key=lambda item: item.sequence)
        missing = [message_id for message_id in evidence_ids if message_id not in by_id]
        if missing:
            raise ContextCompilationError(
                f"evidence messages are not visible in this session: {missing}"
            )
        known_user_ids = {item.user_id for item in by_id.values() if item.user_id}
        unknown_users = [
            user_id
            for user_id in contract.target_user_ids
            if user_id not in known_user_ids
        ]
        if unknown_users:
            raise ContextCompilationError(
                f"target users are not visible in this session: {unknown_users}"
            )
        if contract.quote_message_id:
            if contract.quote_message_id not in by_id:
                raise ContextCompilationError("quote_message_id is not visible")
            if not contract.quote_message_id.isdigit():
                raise ContextCompilationError("quote_message_id must be a QQ numeric id")

        provider_context: list[str] = []

        async def inspect_person(user_id: str) -> None:
            provider_context.extend(
                await self.providers.query(
                    ProviderQuery(
                        operation="person",
                        session_id=contract.target_session_id,
                        user_id=user_id,
                    )
                )
            )

        async def inspect_media(message_id: str) -> None:
            message = by_id[message_id]
            if message.has_media:
                provider_context.extend(
                    await self.providers.query(
                        ProviderQuery(
                            operation="media",
                            session_id=contract.target_session_id,
                            message_id=message_id,
                        )
                    )
                )

        async with anyio.create_task_group() as task_group:
            for user_id in contract.target_user_ids:
                task_group.start_soon(inspect_person, user_id)
            for message_id in evidence_ids:
                task_group.start_soon(inspect_media, message_id)

        return CompiledContext(
            session_id=contract.target_session_id,
            messages=messages,
            evidence=[by_id[message_id] for message_id in evidence_ids],
            provider_context=list(dict.fromkeys(provider_context)),
        )
