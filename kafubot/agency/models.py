from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from kafubot.cognition.types import UserMessage


class ToneVector(BaseModel):
    warmth: float = Field(default=0.5, ge=0, le=1)
    playfulness: float = Field(default=0.4, ge=0, le=1)
    intimacy: float = Field(default=0.3, ge=0, le=1)
    assertiveness: float = Field(default=0.4, ge=0, le=1)
    formality: float = Field(default=0.1, ge=0, le=1)
    energy: float = Field(default=0.5, ge=0, le=1)


class ActionContract(BaseModel):
    """Semantic social action selected by the executive, before wording."""

    target_session_id: str = Field(description="Session selected from Social Home.")
    target_user_ids: list[str] = Field(default_factory=list, max_length=4)
    evidence_message_ids: list[str] = Field(default_factory=list, max_length=8)
    quote_message_id: str = ""
    stance: str = Field(description="What position the agent takes in this exchange.")
    relationship_position: str = Field(
        description="How close, distant, playful, or careful the agent should be."
    )
    response_need: str = Field(description="The concrete thing this response must do.")
    prohibited_topics: list[str] = Field(default_factory=list, max_length=8)
    behavior: str = Field(description="Speech act and visible behavior to realize.")
    expected_effect: str = Field(description="Expected observable social effect.")
    facts_to_preserve: list[str] = Field(default_factory=list, max_length=8)
    meme_intent: str = Field(
        default="",
        description="Optional visual reaction intent; empty means no meme should be sent.",
        max_length=160,
    )
    tone: ToneVector = Field(default_factory=ToneVector)


class TimelineEntry(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    sequence: int = 0
    role: Literal["user", "assistant"]
    timestamp: datetime
    content: str
    user: str = ""
    user_id: str = ""
    message_id: str = ""
    directed_to_bot: bool = False
    has_media: bool = False
    raw: UserMessage | None = Field(default=None, exclude=True)


class ConversationCandidate(BaseModel):
    session_id: str
    title: str
    preview: str
    people: list[str] = Field(default_factory=list)
    latest_at: datetime
    new_count: int
    directed_count: int
    question_count: int
    latest_sequence: int


class AttentionFeatures(BaseModel):
    directedness: float = 0
    social_obligation: float = 0
    relationship: float = 0
    urgency: float = 0
    continuity: float = 0
    novelty: float = 0
    fatigue: float = 0
    interruption: float = 0


class HomeItem(ConversationCandidate):
    score: float
    features: AttentionFeatures


class SocialHome(BaseModel):
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    items: list[HomeItem] = Field(default_factory=list)

    def as_prompt(self) -> str:
        if not self.items:
            return "(no conversations currently require attention)"
        lines: list[str] = []
        for item in self.items:
            people = ", ".join(item.people) or "unknown"
            lines.append(
                f"- {item.session_id} | score={item.score:.3f} | "
                f"new={item.new_count} directed={item.directed_count} | "
                f"people={people} | preview={item.preview}"
            )
        return "\n".join(lines)


class ActiveThread(BaseModel):
    session_id: str
    summary: str
    attention_residue: float = Field(default=0, ge=0)
    residue_updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_action_at: datetime | None = None


class RecentAction(BaseModel):
    session_id: str
    behavior: str
    expected_effect: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class StateDelta(BaseModel):
    """Persistable state change; never contains model reasoning or scratchpad text."""

    summary: str
    focus_added: list[str] = Field(default_factory=list)
    focus_removed: list[str] = Field(default_factory=list)
    action: RecentAction | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SelfState(BaseModel):
    version: int = 1
    current_focus: list[str] = Field(default_factory=list)
    active_threads: dict[str, ActiveThread] = Field(default_factory=dict)
    recent_actions: list[RecentAction] = Field(default_factory=list)
    fatigue_by_session: dict[str, float] = Field(default_factory=dict)
    last_delta: StateDelta | None = None


class CompiledContext(BaseModel):
    session_id: str
    messages: list[TimelineEntry]
    evidence: list[TimelineEntry]
    provider_context: list[str] = Field(default_factory=list)

    @property
    def known_user_ids(self) -> set[str]:
        return {message.user_id for message in self.messages if message.user_id}


class ReplyResult(BaseModel):
    full_text: str
    message_count: int


class RoundResult(BaseModel):
    home_size: int = 0
    steps: int = 0
    replies: int = 0
    skipped: int = 0
    exhausted_budget: bool = False
