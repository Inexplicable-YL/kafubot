from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, ClassVar, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class LogConfig(ConfigModel):
    level: str | int = "INFO"
    verbose_exception: bool = True


class BotConfig(ConfigModel):
    event_queue_size: int = Field(default=100, ge=0)
    event_workers: int = Field(default=100, ge=1)
    adapter: str = "cqhttp"
    adapter_max_retries: int = Field(default=30, ge=0)
    log: LogConfig = LogConfig()


class CQHTTPConfig(ConfigModel):
    __config_name__: ClassVar[str] = "cqhttp"
    adapter_type: Literal["ws", "reverse-ws", "ws-reverse"] = "reverse-ws"
    host: str = "127.0.0.1"
    port: int = Field(default=8080, ge=1, le=65535)
    url: str = "/cqhttp/ws"
    reconnect_interval: float = Field(default=3, ge=0)
    api_timeout: float = Field(default=10, gt=0)
    access_token: str = ""


class GateConfig(ConfigModel):
    """Cheap wake-up policy applied before the executive model is called."""

    threshold: float = Field(default=0.75, ge=0)
    debounce_seconds: float = Field(default=1.5, ge=0)
    idle_wake_seconds: float = Field(default=30, gt=0)
    novelty_per_message: float = Field(default=0.2, ge=0)
    max_novelty: float = Field(default=0.8, ge=0)
    residue_half_life_seconds: float = Field(default=180, gt=0)


class AttentionConfig(ConfigModel):
    """Weights for the global, cross-conversation attention scheduler."""

    directedness: float = 1.4
    social_obligation: float = 0.8
    relationship: float = 0.35
    urgency: float = 0.45
    continuity: float = 0.9
    novelty: float = 0.55
    fatigue: float = 0.55
    interruption: float = 0.35
    random_jitter: float = Field(default=0.03, ge=0)
    residue_half_life_seconds: float = Field(default=180, gt=0)


class ExecutiveConfig(ConfigModel):
    """Budgets and progressive-disclosure limits for one executive round."""

    home_session_limit: int = Field(default=12, ge=1)
    initial_chat_messages: int = Field(default=12, ge=1)
    read_page_size: int = Field(default=20, ge=1)
    focus_budget: int = Field(default=12, ge=1)
    max_steps: int = Field(default=10, ge=1)
    max_replies_per_round: int = Field(default=3, ge=1)
    context_message_limit: int = Field(default=36, ge=1)
    recent_action_limit: int = Field(default=30, ge=1)
    state_file: str = ".database/agent_loop_state.json"
    display_timezone: str = "Auto"

    @field_validator("display_timezone")
    @classmethod
    def validate_display_timezone(cls, value: str) -> str:
        timezone_name = value.strip()
        if timezone_name.casefold() == "auto":
            return "Auto"
        try:
            ZoneInfo(timezone_name)
        except (ValueError, ZoneInfoNotFoundError) as exc:
            raise ValueError(f"unknown IANA timezone: {value!r}") from exc
        return timezone_name


class PluginSettings(ConfigModel):
    """Enable one cognitive component and pass its component-owned options."""

    enabled: bool = True
    options: dict[str, Any] = Field(default_factory=dict)


class AgentConfig(ConfigModel):
    """Configuration for the single social/cognitive runtime."""

    proactive_sessions: set[str] = Field(default_factory=set)
    ignore_user_ids: set[int] = Field(default_factory=set)
    clear_keywords: set[str] = {"/clear", "/清除"}
    environment_message_limit: int = Field(default=100, ge=1)
    image_analyzer_workers: int = Field(default=10, ge=1)
    talk_value: float = Field(default=0.8, ge=0)
    reply_keywords: set[tuple[str, float]] = Field(default_factory=set)
    gate: GateConfig = GateConfig()
    attention: AttentionConfig = AttentionConfig()
    executive: ExecutiveConfig = ExecutiveConfig()
    plugins: dict[str, PluginSettings] = Field(default_factory=dict)

    def plugin(self, name: str, *, default_enabled: bool = True) -> PluginSettings:
        return self.plugins.get(name, PluginSettings(enabled=default_enabled))


class AppConfig(ConfigModel):
    bot: BotConfig = BotConfig()
    adapter: dict[str, Any] = Field(default_factory=dict)
    agent: AgentConfig = AgentConfig()

    @property
    def cqhttp(self) -> CQHTTPConfig:
        return CQHTTPConfig.model_validate(self.adapter.get("cqhttp", {}))


def load_config(path: str | Path) -> AppConfig:
    with Path(path).open("rb") as config_file:
        return AppConfig.model_validate(tomllib.load(config_file))
