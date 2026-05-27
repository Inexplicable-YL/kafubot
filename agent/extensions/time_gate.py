import math
from bisect import bisect_left
from collections import defaultdict, deque
from enum import Enum, auto
from typing import Any, ClassVar, NotRequired, TypedDict

import numpy as np
from langchain.agents.middleware import (
    AgentMiddleware,
    hook_config,
)
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.runtime import Runtime
from opencc import OpenCC
from pydantic import BaseModel, ConfigDict, Field

from agent.base import (
    ManagerContext,
    ManagerState,
    OutputMessage,
    UserMessage,
)


class State(Enum):
    IDLE = auto()
    ACCUMULATING = auto()
    PRIMED = auto()


class TimeGateConfig(TypedDict):
    talk_value: NotRequired[float]
    velocity_alpha: NotRequired[float]
    temperature: NotRequired[float]
    relevance_decay: NotRequired[float]
    keywords: NotRequired[set[tuple[str, float]]]


class TimeGateSnapshot(TypedDict):
    state: State
    pressure: float
    human_velocity: float
    human_intervals: NotRequired[list[float]]
    last_timestamp: NotRequired[float | None]
    relevance_maps: NotRequired[dict[str, float]]


class TimeGate(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    talk_value: float = 0.2
    velocity_alpha: float = 0.2
    temperature: float = 1.0
    relevance_decay: float = 0.4
    keywords: set[tuple[str, float]] = Field(default_factory=set)

    state: State = Field(default=State.IDLE, repr=False)
    pressure: float = Field(default=0.0, repr=False)

    human_intervals: deque[float] = Field(
        default_factory=lambda: deque(maxlen=100), repr=False
    )
    human_velocity: float = Field(default=0.5, repr=False)
    last_timestamp: float | None = Field(default=None, repr=False)

    celi_relevance: float = Field(default=0.0, repr=False)
    relevance_maps: dict[str, float] = Field(default_factory=dict, repr=False)

    _trans_t2s: ClassVar[OpenCC] = OpenCC("t2s")
    _trans_s2t: ClassVar[OpenCC] = OpenCC("s2t")

    def observe(
        self,
        messages: list[AnyMessage],
        from_snapshot: TimeGateSnapshot | None = None,
    ) -> TimeGateSnapshot:
        if from_snapshot is not None:
            self.state = from_snapshot["state"]
            self.pressure = from_snapshot["pressure"]
            self.human_velocity = from_snapshot["human_velocity"]
            self.human_intervals = deque(from_snapshot.get("human_intervals", []))
            self.last_timestamp = from_snapshot.get("last_timestamp")
            self.relevance_maps = from_snapshot.get("relevance_maps", {})

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
                    self._observe_human_fast(user_msg.timestamp.timestamp())
            self._observe_ai()
            for msg in messages[last_ai_idx + 1 :]:
                self._process_message(msg)
        else:
            for msg in messages:
                self._process_message(msg)

        return TimeGateSnapshot(
            state=self.state,
            pressure=self.pressure,
            human_velocity=self.human_velocity,
            human_intervals=list(self.human_intervals),
            last_timestamp=self.last_timestamp,
            relevance_maps=self.relevance_maps,
        )

    def evaluate(self) -> bool:
        print(f"pressure: {self.pressure}, human_velocity: {self.human_velocity}")
        return self.state is State.PRIMED

    def clear(self) -> None:
        self.state = State.IDLE
        self.pressure = 0.0
        self.human_velocity = 0.5
        self.human_intervals.clear()
        self.last_timestamp = None
        self.relevance_maps.clear()

    def _process_message(self, msg: AnyMessage) -> None:
        if isinstance(msg, AIMessage):
            self._observe_ai()
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
            self._observe_human(user_msg.timestamp.timestamp(), self.celi_relevance)

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

    def _observe_ai(self) -> None:
        self.state = State.IDLE
        self.pressure = 0.0

    def _observe_human(self, timestamp: float, relevance: float) -> None:
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

        self.pressure += 0.5 * (1 + relevance)

        if self.state is State.IDLE:
            probs = self._idle_probs(self.pressure, relevance)
            self._transition(probs)
        elif self.state is State.ACCUMULATING:
            probs = self._accumulating_probs(self.pressure, relevance)
            self._transition(probs)

    def _observe_human_fast(self, timestamp: float) -> None:
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
    ) -> None:
        rng = np.random.default_rng()
        index = rng.choice(len(probs), p=probs)
        self.state = (State.IDLE, State.ACCUMULATING, State.PRIMED)[index]

    def _handle_velocity(self) -> float:
        value = 0.3 + self.talk_value * 0.7
        return 1 + value * self.human_velocity / (
            1 + np.exp(10 * (self.human_velocity - 0.85))
        )


class TimeGateMiddleware(AgentMiddleware[ManagerState, ManagerContext]):
    time_gates: dict[str, TimeGate]
    snapshots: dict[str, TimeGateSnapshot]

    def __init__(
        self,
        gate_config: TimeGateConfig,
    ) -> None:
        self.time_gates = defaultdict(lambda: TimeGate(**gate_config))
        self.snapshots = {}

    @hook_config(can_jump_to=["end"])
    async def abefore_agent(
        self, state: ManagerState, runtime: Runtime[ManagerContext]
    ) -> dict[str, Any] | None:
        session_id = runtime.context["session_id"]
        time_gate = self.time_gates[session_id]
        if session_id not in self.snapshots:
            time_gate.clear()
            self.snapshots[session_id] = time_gate.observe(state["history_messages"])
        time_gate.observe(
            state["current_messages"], from_snapshot=self.snapshots[session_id]
        )
        if not runtime.context["is_tome"] and (not time_gate.evaluate()):
            return {
                "jump_to": "end",
                "outputs": [
                    OutputMessage(
                        type="finish", data={"reason": "restricted by talk_value."}
                    )
                ],
            }
        return None

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
            self.snapshots[runtime.context["session_id"]] = self.time_gates[
                runtime.context["session_id"]
            ].observe(observe)
