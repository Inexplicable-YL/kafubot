import math
from bisect import bisect_left
from collections import defaultdict, deque
from collections.abc import Sequence
from datetime import datetime
from enum import Enum, auto
from hashlib import blake2s
from typing import Any, ClassVar, NotRequired, TypedDict, cast
from typing_extensions import override

import numpy as np
from langchain.agents.middleware import (
    AgentMiddleware,
    hook_config,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.runtime import Runtime
from opencc import OpenCC
from pydantic import BaseModel, ConfigDict, Field

from agent.base import (
    ManagerContext,
    ManagerState,
    OutputMessage,
    UserMessage,
)
from agent.session import register_session_clearer
from agent.telemetry import log_social_event


class TimeGateConfig(TypedDict):
    talk_value: NotRequired[float]
    velocity_alpha: NotRequired[float]
    relevance_decay: NotRequired[float]
    keywords: NotRequired[set[tuple[str, float]]]


class _State(Enum):
    IDLE = auto()
    ACCUMULATING = auto()
    PRIMED = auto()


class IgnoreMessage(BaseMessage):
    pass


class _TimeGate(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    talk_value: float = 0.2
    velocity_alpha: float = 0.2
    relevance_decay: float = 0.4
    keywords: set[tuple[str, float]] = Field(default_factory=set)

    state: _State = Field(default=_State.IDLE, repr=False)
    pressure: float = Field(default=0.0, repr=False)
    ignore_count: int = Field(default=0, repr=False)

    human_intervals: deque[float] = Field(
        default_factory=lambda: deque(maxlen=100), repr=False
    )
    human_velocity: float = Field(default=0.5, repr=False)
    last_timestamp: float | None = Field(default=None, repr=False)
    last_ai_timestamp: float | None = Field(default=None, repr=False)
    seen_message_ids: deque[str] = Field(
        default_factory=lambda: deque(maxlen=512), repr=False
    )

    celi_relevance: float = Field(default=0.0, repr=False)
    relevance_maps: dict[str, float] = Field(default_factory=dict, repr=False)
    last_transition_probs: tuple[float, float, float] = Field(
        default=(1.0, 0.0, 0.0), repr=False
    )
    last_transition_draw: float = Field(default=0.0, repr=False)
    last_decision_key: str = Field(default="", repr=False)
    seed_scope: str = Field(default="", repr=False)
    last_utility: float = Field(default=0.0, repr=False)
    last_threshold: float = Field(default=0.0, repr=False)
    last_allowed: bool = Field(default=False, repr=False)
    last_costs: dict[str, float] = Field(default_factory=dict, repr=False)

    _trans_t2s: ClassVar[OpenCC] = OpenCC("t2s")
    _trans_s2t: ClassVar[OpenCC] = OpenCC("s2t")

    def observe(
        self,
        messages: Sequence[BaseMessage],
    ) -> None:
        if self.last_timestamp is not None:
            messages = [
                msg
                for msg in messages
                if isinstance(msg, AIMessage | IgnoreMessage)
                or (
                    isinstance(msg, HumanMessage)
                    and (user_msg := msg.additional_kwargs.get("raw"))
                    and isinstance(user_msg, UserMessage)
                    and user_msg.message_id not in self.seen_message_ids
                )
            ]
        last_ai_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], AIMessage):
                last_ai_idx = i
                break

        if last_ai_idx >= 0:
            for msg in messages[:last_ai_idx]:
                if (
                    isinstance(msg, HumanMessage)
                    and (user_msg := msg.additional_kwargs.get("raw"))
                    and isinstance(user_msg, UserMessage)
                ):
                    relevance = self._update_relevance(user_msg)
                    self.celi_relevance = max(
                        relevance,
                        (
                            (1 - self.relevance_decay) * self.celi_relevance
                            + self.relevance_decay * relevance
                        ),
                    )
                    self._observe_human_fast(
                        user_msg.timestamp.timestamp(), user_msg.message_id
                    )
            last_ai = messages[last_ai_idx]
            created_at: Any = last_ai.additional_kwargs.get("created_at")
            self._observe_ai(
                cast("datetime", created_at).timestamp()
                if isinstance(created_at, datetime)
                else None
            )
            for msg in messages[last_ai_idx + 1 :]:
                self._process_message(msg)
        else:
            for msg in messages:
                self._process_message(msg)

    def evaluate(self) -> bool:
        pressure_value = math.tanh(
            self.pressure * max(0.02, self.talk_value) * self._handle_velocity()
        )
        interruption_cost = 0.24 * self.human_velocity * (1 - self.celi_relevance)
        refractory_cost = 0.0
        if self.last_ai_timestamp is not None and self.last_timestamp is not None:
            seconds_since_ai = max(0.0, self.last_timestamp - self.last_ai_timestamp)
            if seconds_since_ai < 45:
                refractory_cost = 0.3 * (1 - seconds_since_ai / 45)
                refractory_cost *= 1 - self.celi_relevance
        utility = (
            0.5 * self.celi_relevance
            + 0.36 * pressure_value
            + 0.38 * min(1.0, self.ignore_count / 4)
            - interruption_cost
            - refractory_cost
        )
        threshold = 0.48 - 0.28 * max(0.0, min(self.talk_value, 1.0))
        # Small hysteresis avoids alternating speak/silence around the boundary.
        if self.last_allowed:
            threshold -= 0.04
        self.last_utility = utility
        self.last_threshold = threshold
        self.last_costs = {
            "interruption": interruption_cost,
            "refractory": refractory_cost,
            "pressure_value": pressure_value,
        }
        self.last_allowed = utility >= threshold
        return self.last_allowed

    def clear(self) -> None:
        self.state = _State.IDLE
        self.pressure = 0.0
        self.human_velocity = 0.5
        self.human_intervals.clear()
        self.last_timestamp = None
        self.last_ai_timestamp = None
        self.seen_message_ids.clear()
        self.relevance_maps.clear()
        self.celi_relevance = 0.0
        self.ignore_count = 0
        self.last_transition_probs = (1.0, 0.0, 0.0)
        self.last_transition_draw = 0.0
        self.last_decision_key = ""
        self.last_utility = 0.0
        self.last_threshold = 0.0
        self.last_allowed = False
        self.last_costs.clear()

    def _process_message(self, msg: BaseMessage) -> None:
        if isinstance(msg, AIMessage):
            created_at: Any = msg.additional_kwargs.get("created_at")
            timestamp = (
                cast("datetime", created_at).timestamp()
                if isinstance(created_at, datetime)
                else self.last_timestamp
            )
            self._observe_ai(timestamp)
            return
        if isinstance(msg, IgnoreMessage):
            self._observe_ignore()
            return

        if (
            isinstance(msg, HumanMessage)
            and (user_msg := msg.additional_kwargs.get("raw"))
            and isinstance(user_msg, UserMessage)
        ):
            relevance = self._update_relevance(user_msg)
            self.celi_relevance = max(
                relevance,
                (1 - self.relevance_decay) * self.celi_relevance
                + self.relevance_decay * relevance,
            )
            self._observe_human(
                user_msg.timestamp.timestamp(),
                self.celi_relevance,
                decision_key=user_msg.message_id,
            )

    def _update_relevance(self, user_msg: UserMessage) -> float:
        self._set_relevance()
        relevance = self.relevance_maps.get(user_msg.user_id, 0.0)
        if user_msg.is_tome:
            relevance = 1.0
        elif self.keywords and (content := user_msg.message.get_msgcode().casefold()):
            content = self._trans_t2s.convert(content) + self._trans_s2t.convert(
                content
            )
            _relevance = 0.0
            for keyword, weight in self.keywords:
                if keyword in content:
                    _relevance += weight
            relevance = max(relevance, math.tanh(_relevance))
        relevance = max(0.0, min(relevance, 1.0))
        self.relevance_maps[user_msg.user_id] = relevance
        return relevance

    def _set_relevance(self) -> None:
        expired = []
        for user_id, relevance in self.relevance_maps.items():
            _relevance = relevance * self.relevance_decay
            if _relevance < 1e-2:  # noqa: PLR2004
                expired.append(user_id)
            else:
                self.relevance_maps[user_id] = _relevance
        for user_id in expired:
            del self.relevance_maps[user_id]

    def _observe_ai(self, timestamp: float | None = None) -> None:
        self.state = _State.IDLE
        self.pressure = 0.0
        self.ignore_count = 0
        self.last_ai_timestamp = (
            timestamp if timestamp is not None else self.last_timestamp
        )
        self.last_allowed = False

    def _observe_ignore(self) -> None:
        self.state = _State.ACCUMULATING
        self.pressure *= 0.2 + 0.8 * self.talk_value
        self.ignore_count += 1

    def _observe_human(
        self, timestamp: float, relevance: float, *, decision_key: str
    ) -> None:
        if self.last_timestamp is not None:
            self.human_intervals.append(timestamp - self.last_timestamp)
        if len(self.human_intervals) > 1:
            self.human_velocity = (
                self.velocity_alpha
                * (
                    1
                    - bisect_left(
                        sorted(self.human_intervals), self.human_intervals[-1]
                    )
                    / len(self.human_intervals)
                )
                + (1 - self.velocity_alpha) * self.human_velocity
            )
        self.last_timestamp = timestamp
        if decision_key:
            self.seen_message_ids.append(decision_key)

        self.pressure += (0.5 * 0.8**self.ignore_count + 0.5) + relevance

        if self.state is _State.IDLE:
            probs = self._idle_probs(self.pressure, relevance)
            self._transition(probs, decision_key=decision_key)
        elif self.state is _State.ACCUMULATING:
            probs = self._accumulating_probs(self.pressure, relevance)
            self._transition(probs, decision_key=decision_key)

    def _observe_human_fast(self, timestamp: float, message_id: str) -> None:
        if self.last_timestamp is not None:
            self.human_intervals.append(timestamp - self.last_timestamp)
        if len(self.human_intervals) > 1:
            self.human_velocity = (
                self.velocity_alpha
                * (
                    1
                    - bisect_left(
                        sorted(self.human_intervals), self.human_intervals[-1]
                    )
                    / len(self.human_intervals)
                )
                + (1 - self.velocity_alpha) * self.human_velocity
            )
        self.last_timestamp = timestamp
        if message_id:
            self.seen_message_ids.append(message_id)

    def _idle_probs(
        self, pressure: float, relevance: float
    ) -> tuple[float, float, float]:
        cali_pressure = pressure * self.talk_value * self._handle_velocity()
        base = math.tanh(cali_pressure)
        mobility = 0.3 + relevance * 0.7

        p_primed = base * mobility
        remaining = 1.0 - p_primed
        p_acc = remaining * mobility
        p_idle = remaining * (1.0 - mobility)
        return p_idle, p_acc, p_primed

    def _accumulating_probs(
        self, pressure: float, relevance: float
    ) -> tuple[float, float, float]:
        cali_pressure = pressure * self.talk_value * self._handle_velocity()
        base = math.tanh(cali_pressure)
        mobility1 = 0.3 + relevance * 0.7
        mobility2 = 0.3 + (1.0 - relevance) * 0.7

        p_primed = base * mobility1
        remaining = 1.0 - p_primed
        p_idle = remaining * mobility2
        p_acc = remaining * (1.0 - mobility2)
        return p_idle, p_acc, p_primed

    def _transition(
        self,
        probs: tuple[float, float, float],
        *,
        decision_key: str,
    ) -> None:
        normalized = tuple(max(0.0, float(item)) for item in probs)
        total = sum(normalized)
        if total <= 0:
            normalized = (1.0, 0.0, 0.0)
            total = 1.0
        normalized = (
            normalized[0] / total,
            normalized[1] / total,
            normalized[2] / total,
        )
        seed_material = (
            f"{self.seed_scope}|{decision_key}|{self.state.name}|{self.pressure:.8f}|"
            f"{self.celi_relevance:.8f}|{self.ignore_count}"
        )
        seed = int.from_bytes(blake2s(seed_material.encode(), digest_size=8).digest())
        draw = seed / 2**64
        cumulative = 0.0
        index = len(normalized) - 1
        for candidate_index, probability in enumerate(normalized):
            cumulative += probability
            if draw <= cumulative:
                index = candidate_index
                break
        self.last_transition_probs = normalized
        self.last_transition_draw = draw
        self.last_decision_key = decision_key
        self.state = (_State.IDLE, _State.ACCUMULATING, _State.PRIMED)[index]

    def _handle_velocity(self) -> float:
        value = 0.3 + self.talk_value * 0.7
        return 1 + value * self.human_velocity / (
            1 + np.exp(10 * (self.human_velocity - 0.85))
        )


class TimeGateMiddleware(AgentMiddleware[ManagerState, ManagerContext]):
    time_gates: dict[str, _TimeGate]
    initialized: dict[str, bool]

    def __init__(
        self,
        talk_value: float = 0.2,
        velocity_alpha: float = 0.2,
        relevance_decay: float = 0.4,
        keywords: set[tuple[str, float]] | None = None,
    ) -> None:
        self.talk_value = talk_value
        self.time_gates = defaultdict(
            lambda: _TimeGate(
                talk_value=talk_value,
                velocity_alpha=velocity_alpha,
                relevance_decay=relevance_decay,
                keywords=keywords or set(),
            )
        )
        self.initialized = defaultdict(lambda: False)
        register_session_clearer(self.clear_session)

    async def clear_session(self, session_id: str) -> None:
        self.time_gates.pop(session_id, None)
        self.initialized.pop(session_id, None)

    @override
    @hook_config(can_jump_to=["end"])
    async def abefore_agent(
        self, state: ManagerState, runtime: Runtime[ManagerContext]
    ) -> dict[str, Any] | None:
        session_id = runtime.context["session_id"]
        time_gate = self.time_gates[session_id]
        time_gate.seed_scope = session_id
        if not self.initialized[session_id]:
            time_gate.clear()
            time_gate.observe(state["histories"] + state["currents"])
            self.initialized[session_id] = True
        else:
            time_gate.observe(state["currents"])
        policy_allowed = time_gate.evaluate()
        allowed = runtime.context["is_tome"] or policy_allowed
        decision = {
            "allowed": allowed,
            "directed_to_bot": runtime.context["is_tome"],
            "policy_allowed": policy_allowed,
            "state": time_gate.state.name.lower(),
            "pressure": time_gate.pressure,
            "relevance": time_gate.celi_relevance,
            "human_velocity": time_gate.human_velocity,
            "transition_probs": time_gate.last_transition_probs,
            "transition_draw": time_gate.last_transition_draw,
            "decision_message_id": time_gate.last_decision_key,
            "utility": time_gate.last_utility,
            "threshold": time_gate.last_threshold,
            "costs": time_gate.last_costs,
        }
        await log_social_event(
            "intervention_gate",
            session_id=session_id,
            **decision,
        )
        if not allowed:
            return {
                "jump_to": "end",
                "outputs": [
                    OutputMessage(
                        type="finish", data={"reason": "restricted by talk_value."}
                    )
                ],
                "intervention_decision": decision,
            }
        return {"intervention_decision": decision}

    @override
    async def aafter_agent(
        self, state: ManagerState, runtime: Runtime[ManagerContext]
    ) -> dict[str, Any] | None:
        _ = runtime
        observe = []
        for output in state["outputs"]:
            if output["type"] == "reply":
                observe.append(AIMessage(output["data"]["full_text"]))
            elif output["type"] == "meme":
                observe.append(AIMessage(output["data"]["content"]))
        if observe:
            self.time_gates[runtime.context["session_id"]].observe(observe)
        else:
            self.time_gates[runtime.context["session_id"]].observe(
                [IgnoreMessage(type="non_standard", content="ignore")]
            )


__all__ = ["TimeGateConfig", "TimeGateMiddleware"]
