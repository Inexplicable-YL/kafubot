import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from html import escape
from typing import Any, Literal, NotRequired, TypedDict, cast
from typing_extensions import override

import pandas as pd
from langchain.agents.middleware import (
    AgentMiddleware,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
    hook_config,
)
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.load.load import load
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    Runnable,
    RunnableGenerator,
    RunnableLambda,
)
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tracers.schemas import Run
from langgraph.runtime import Runtime
from langgraph.types import Command
from numpy import random
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from sekaibot.adapter.cqhttp.message import CQHTTPMessageSegment

from agent.base import (
    MODEL_VISIBLE_TZ,
    ManagerContext,
    ManagerState,
    OutputMessage,
    UserMessage,
)
from agent.message import QQMessage
from agent.prompts.manager import (
    BOT_NAME,
    IDENTITY,
)
from agent.prompts.replyer import (
    GROUP_SYSTEM_PROMPT,
    LESS_REPLY,
    MORE_REPLY,
    REPLY_USER_PROMPT,
)
from agent.session import register_session_clearer
from agent.telemetry import log_social_event
from agent.utils import content_to_text, to_reply

logger = logging.getLogger(__name__)


class InteractionConfig(TypedDict):
    reply_model: NotRequired[BaseChatModel]
    get_session_history: NotRequired[Callable[..., BaseChatMessageHistory]]
    stop_when_reply: NotRequired[bool]
    average_reply_count: NotRequired[float]
    min_history_window: NotRequired[int]
    max_history_window: NotRequired[int]
    max_reply_per_turn: NotRequired[int]
    max_turns: NotRequired[int]
    max_tool_calls: NotRequired[int]
    mas_retries: NotRequired[int]


@tool(
    description="结束本轮思考，等待后续新的外部消息再继续。或者本轮不进行任何动作，等待其他用户的新消息；也用于用户可能还没说完、需要先把发言权交还给用户的场景。"
)
def finish(runtime: ToolRuntime) -> Command:
    """调用finish工具实现结束对话。"""
    return Command(
        update={
            "outputs": [OutputMessage(type="finish", data={})],
            "messages": [
                ToolMessage(
                    content="当前 Manager 已结束本轮思考，等待新的群聊消息。",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


class ToneVector(BaseModel):
    warmth: float = Field(default=0.5, ge=0.0, le=1.0)
    playfulness: float = Field(default=0.5, ge=0.0, le=1.0)
    intimacy: float = Field(default=0.3, ge=0.0, le=1.0)
    assertiveness: float = Field(default=0.4, ge=0.0, le=1.0)
    formality: float = Field(default=0.1, ge=0.0, le=1.0)
    energy: float = Field(default=0.5, ge=0.0, le=1.0)
    sarcasm: float = Field(default=0.0, ge=0.0, le=1.0)
    face_threat: float = Field(default=0.0, ge=0.0, le=1.0)


class SocialActionInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    focus_thread_id: str = Field(
        default="",
        description="内部会话结构中的话题ID；不确定或私聊时可为空。",
    )
    target_user_ids: list[str] = Field(
        default_factory=list,
        max_length=4,
        description="语义上主要回应的QQ用户ID列表；面向整个话题时为空。不是自动at列表。",
    )
    evidence_message_ids: list[str] = Field(
        default_factory=list,
        min_length=1,
        max_length=8,
        description="支撑本次行动的QQ消息ID，可引用多人的消息作为理解证据。",
    )
    quote_message_id: str = Field(
        default="",
        description="确有消歧需要时引用的一条QQ消息ID；多数群聊回复应为空。QQ一次只引用一条。",
    )
    social_goal: Literal[
        "answer",
        "align",
        "joke",
        "comfort",
        "clarify",
        "correct",
        "continue",
        "redirect",
        "repair",
    ] = "continue"
    speech_act: str = Field(
        default="回应",
        description="简短描述本次话语行为，如回答、接梗、安慰、澄清。",
    )
    tone: ToneVector = Field(default_factory=ToneVector)
    facts_to_preserve: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="回复中不可歪曲的事实或记忆；没有则为空。",
    )
    prohibited_implications: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="回复必须避免暗示的内容，如泄露私聊信息或越过关系边界。",
    )
    runtime: ToolRuntime = Field(exclude=True)


# Compatibility import for extensions that referenced the old schema name.
ReplyInput = SocialActionInput


class GetEarlyMessagesInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    limit: int = Field(default=20, description="要显示的消息数量限制。", gt=1, le=20)
    runtime: ToolRuntime = Field(exclude=True)


class InteractionMiddleware(AgentMiddleware[ManagerState, ManagerContext, Any]):
    reply_count_alpha = 0.1

    get_session_history: Callable[..., BaseChatMessageHistory]
    history_caches: dict[str, list[BaseMessage]]
    summary_pruned_messages: dict[str, list[BaseMessage]]
    reply_turn: dict[str, int]
    real_average_count: dict[str, float]
    display_messages: dict[str, list[BaseMessage]]

    def __init__(
        self,
        reply_model: BaseChatModel,
        get_session_history: Callable[..., BaseChatMessageHistory],
        *,
        stop_when_reply: bool = False,
        average_reply_count: float = 1.25,
        min_history_window: int = 20,
        max_history_window: int = 40,
        max_reply_per_turn: int | None = None,
        max_turns: int = 10,
        max_tool_calls: int = 10,
        mas_retries: int = 3,
        **_: Any,
    ) -> None:
        self.chat_app = self._get_chat_app(reply_model)
        self.get_session_history = get_session_history

        self.min_history_window = min_history_window
        self.max_history_window = max_history_window

        self.stop_when_reply = stop_when_reply
        self.max_reply_per_turn = max_reply_per_turn
        self.max_turns = max_turns
        self.max_tool_calls = max_tool_calls
        self.mas_retries = mas_retries
        self.average_reply_count = average_reply_count

        self.history_caches = defaultdict(list)
        self.summary_pruned_messages = defaultdict(list)
        self.reply_turn = defaultdict(lambda: 0)
        self.real_average_count = defaultdict(lambda: 0)
        self.display_messages = defaultdict(list)
        register_session_clearer(self.clear_session)

        self.tools = [
            finish,
            tool(
                "reply",
                args_schema=SocialActionInput,
                description=(
                    "执行一次可见社交行动。行动可以回应一个话题、一个人或多人；"
                    "必须给出证据消息ID。QQ引用最多一条且仅在消歧必要时使用。"
                ),
            )(self.reply),
            tool(
                "get_early_messages",
                args_schema=GetEarlyMessagesInput,
                description="获取比当前可见的消息更早的消息，并自动添加到上下文中。需要使用`limit`限制显示的消息数量。",
            )(self.get_early_messages),
        ]

    async def clear_session(self, session_id: str) -> None:
        self.history_caches.pop(session_id, None)
        self.summary_pruned_messages.pop(session_id, None)
        self.reply_turn.pop(session_id, None)
        self.real_average_count.pop(session_id, None)
        self.display_messages.pop(session_id, None)

    async def reply(  # noqa: PLR0915
        self,
        focus_thread_id: str,
        target_user_ids: list[str],
        evidence_message_ids: list[str],
        quote_message_id: str,
        social_goal: str,
        speech_act: str,
        tone: ToneVector,
        facts_to_preserve: list[str],
        prohibited_implications: list[str],
        runtime: ToolRuntime[ManagerContext, ManagerState],
    ) -> Any:
        """Compile a structured social action into QQ-supported message segments."""
        if (
            self.max_reply_per_turn is not None
            and self.reply_turn[runtime.context["session_id"]]
            >= self.max_reply_per_turn
        ):
            return "本轮系统允许的回复次数已用完，请等待下一轮。"
        evidence_output: list[str] = []
        target_messages: list[str] = []
        meme_msgs: list[str] = [
            str(content).replace("已发送表情包", "当前系统已自动发送表情包")
            for output in runtime.state["outputs"]
            if output["type"] == "meme" and (content := output["data"].get("content"))
        ]
        input_by_id = {msg.message_id: msg for msg in runtime.state["inputs"]}
        frame = runtime.state.get("conversation_frame")
        frame_user_ids = {
            str(getattr(participant, "user_id", ""))
            for participant in getattr(frame, "participants", [])
            if getattr(participant, "user_id", "")
        }
        allowed_target_ids = {
            *(msg.user_id for msg in runtime.state["inputs"]),
            *frame_user_ids,
        }
        invalid_target_ids = [
            user_id
            for user_id in dict.fromkeys(item.strip() for item in target_user_ids)
            if user_id and user_id not in allowed_target_ids
        ]
        if invalid_target_ids:
            return "`target_user_ids` 只能使用当前QQ会话中可观察到的真实用户ID。"
        normalized_target_ids = [
            user_id
            for user_id in dict.fromkeys(item.strip() for item in target_user_ids)
            if user_id
        ]
        focus_thread_id = focus_thread_id.strip()
        frame_thread_ids = {
            str(getattr(thread, "thread_id", ""))
            for thread in getattr(frame, "active_threads", [])
            if getattr(thread, "thread_id", "")
        }
        if focus_thread_id and focus_thread_id not in frame_thread_ids:
            return "`focus_thread_id` 必须为空或来自当前 conversation-structure。"
        valid_evidence_ids = list(
            dict.fromkeys(
                message_id.strip()
                for message_id in evidence_message_ids
                if message_id.strip() in input_by_id
            )
        )
        for message_id in valid_evidence_ids:
            msg = input_by_id[message_id]
            target_messages.append(msg.message.get_msgcode())
            evidence_output.append(
                f"- 时间：{msg.timestamp.astimezone(MODEL_VISIBLE_TZ).strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"- message_id：{msg.message_id}\n"
                f"- 发送人：{escape(msg.user, quote=True)}（QQ {msg.user_id}）\n"
                f"- 消息内容：{msg.message.get_msgcode()}\n"
            )
        if not evidence_output:
            return "请检查 `evidence_message_ids`；至少一个ID必须来自当前QQ消息。"
        quote_message_id = quote_message_id.strip()
        if quote_message_id and quote_message_id not in input_by_id:
            return "`quote_message_id` 必须为空或来自当前QQ消息。"
        if quote_message_id and not quote_message_id.isdigit():
            return "`quote_message_id` 必须是QQ提供的数字消息ID。"
        target_frames = [
            participant
            for participant in getattr(frame, "participants", [])
            if getattr(participant, "user_id", "") in normalized_target_ids
        ]
        has_rapport_evidence = bool(
            target_frames and getattr(frame, "bot_recently_spoke", False)
        ) and all(
            getattr(participant, "directed_to_bot_count", 0) >= 2
            and getattr(participant, "observable_affect", "neutral") == "positive"
            for participant in target_frames
        )
        max_face_threat = 0.4 if has_rapport_evidence else 0.15
        guarded_tone = tone.model_copy(
            update={
                "face_threat": min(tone.face_threat, max_face_threat),
                "sarcasm": min(tone.sarcasm, 0.55 if has_rapport_evidence else 0.2),
            }
        )
        action_plan = {
            "focus_thread_id": focus_thread_id,
            "target_user_ids": normalized_target_ids,
            "evidence_message_ids": valid_evidence_ids,
            "quote_message_id": quote_message_id or None,
            "social_goal": social_goal,
            "speech_act": speech_act.strip(),
            "tone": guarded_tone.model_dump(),
            "tone_guardrail_applied": guarded_tone != tone,
            "rapport_evidence": has_rapport_evidence,
            "facts_to_preserve": facts_to_preserve,
            "prohibited_implications": prohibited_implications,
        }
        await log_social_event(
            "social_action_selected",
            session_id=runtime.context["session_id"],
            action=action_plan,
            selected_behavior_ids=runtime.state.get("selected_behavior_ids", []),
            selected_expression_ids=runtime.state.get("selected_expression_ids", []),
        )
        focus_output = "\n".join(evidence_output)
        if quote_message_id:
            focus_output += "\n- QQ发送层将引用其中一条消息；不要再次at同一人。"
        if meme_msgs:
            focus_output += "\n- 系统已发送表情包：\n" + "\n".join(meme_msgs)
        reference_info = "\n".join(facts_to_preserve) or "无额外事实"
        language_style = (
            f"话语行为={speech_act}；社交目标={social_goal}；"
            f"语气向量={guarded_tone.model_dump_json()}；"
            f"禁止暗示={prohibited_implications or ['无']}"
        )
        top_messages = list(runtime.state.get("reply_top_messages", []) or [])
        bottom_messages = list(runtime.state.get("reply_bottom_messages", []) or [])
        visible_messages = [
            *self.history_caches[runtime.context["session_id"]],
            *runtime.state["currents"],
        ]
        reply_bottom_message_factories = cast(
            "list[Callable[[dict[str, Any]], Awaitable[list[BaseMessage]]]]",
            runtime.state.get("reply_bottom_message_factories", []) or [],
        )
        if reply_bottom_message_factories:
            factory_payload = {
                "session_id": runtime.context["session_id"],
                "messages": visible_messages,
                "target_message": "\n".join(target_messages),
                "reply_reason": reference_info,
                "reasoning_content": "",
                "language_style": language_style,
                "focus_message": focus_output,
                "social_action": action_plan,
            }
            for factory in reply_bottom_message_factories:
                try:
                    extra_messages = await factory(factory_payload)
                except Exception:
                    logger.exception("Failed to build reply bottom messages")
                    continue
                bottom_messages.extend(extra_messages)
        full_text = ""
        async for reply in self.chat_app.astream(
            {
                "messages": runtime.state["currents"],
                "history": self.history_caches[runtime.context["session_id"]],
                "top_messages": top_messages,
                "bottom_messages": bottom_messages,
                "focus_message": focus_output,
                "reference_info": reference_info,
                "language_style": language_style,
                "reasoning_content": "",
                "social_action": action_plan,
                "time": datetime.now(tz=MODEL_VISIBLE_TZ).strftime("%Y-%m-%d %H:%M:%S"),
            },
            config={"configurable": {"session_id": runtime.context["session_id"]}},
        ):
            if reply is not None and (reply_msg := reply.strip()):
                if reply_msg.endswith("（"):
                    reply_msg = reply_msg[:-1].rstrip()
                    rng = random.default_rng()
                    pools = ("（", "（）", "。。", "")
                    probs = (0.4, 0.2, 0.1, 0.3)
                    reply_msg += pools[rng.choice(len(pools), p=probs)]
                raw_msg = QQMessage.from_str(reply_msg)
                known_user_ids = {item.user_id for item in runtime.state["inputs"]}
                known_names = {item.user for item in runtime.state["inputs"]}
                msg = QQMessage(
                    segment
                    for segment in raw_msg
                    if segment.type == "text"
                    or (
                        segment.type == "at"
                        and (
                            str(segment.data.get("user_id", "")) in known_user_ids
                            or str(segment.data.get("name", "")) in known_names
                        )
                    )
                )
                if not msg.get_plain_text().strip():
                    continue
                raw_cq_msg = await msg.get_cqhttp_message(runtime.state["inputs"])
                full_text += msg.get_msgcode() + "\n"
                for seg in raw_cq_msg:
                    if seg.type == "image":
                        await runtime.context["node"].reply(seg)
                        break
                else:
                    if quote_message_id:
                        await runtime.context["node"].reply(
                            CQHTTPMessageSegment.reply(int(quote_message_id))
                            + raw_cq_msg
                        )
                        quote_message_id = ""
                    else:
                        await runtime.context["node"].reply(raw_cq_msg)
        full_text = full_text.strip()
        if not full_text:
            return "回复失败：无法生成回复。"
        await log_social_event(
            "qq_reply_sent",
            session_id=runtime.context["session_id"],
            text=full_text,
            action=action_plan,
            sent_at=datetime.now(UTC).isoformat(),
        )
        print(
            f"Agent-Invoking: 已省略{max(0, len(runtime.state['inputs']) - 5)}个消息，{[m.message.get_msgcode() for m in runtime.state['inputs']][-5:]}\n",
            f"Agent-Reply: {full_text.replace(chr(10), chr(92) + 'n ').strip()}",
        )
        self.reply_turn[runtime.context["session_id"]] += 1
        return Command(
            update={
                "outputs": [
                    OutputMessage(
                        type="reply",
                        data={
                            "full_text": full_text,
                            "social_action": action_plan,
                            "sent_at": datetime.now(UTC).isoformat(),
                            "selected_behavior_ids": runtime.state.get(
                                "selected_behavior_ids", []
                            ),
                            "selected_expression_ids": runtime.state.get(
                                "selected_expression_ids", []
                            ),
                        },
                    )
                ],
                "messages": [
                    ToolMessage(
                        content="已回复消息：\n" + full_text,
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
                "summary_pruned_messages": self.summary_pruned_messages.pop(
                    runtime.context["session_id"], None
                )
                or None,
            }
        )

    async def get_early_messages(
        self,
        limit: int,
        runtime: ToolRuntime[ManagerContext, ManagerState],
    ) -> str:
        session_id = runtime.context["session_id"]
        all_history = await self.get_session_history(session_id).aget_messages()
        early_messages = all_history[
            : -(len(self.history_caches[session_id]) + len(runtime.state["currents"]))
        ]
        if not early_messages:
            return "没有比当前可见的消息更早的消息。"

        message_count = len(self.display_messages[session_id])
        self.display_messages[session_id] = early_messages[-(limit + message_count) :]
        get_message_count = len(self.display_messages[session_id]) - message_count
        return f"获取到 {get_message_count} 条比当前可见的消息更早的消息。消息已添加到上下文中。"

    @override
    async def abefore_agent(
        self, state: ManagerState, runtime: Runtime[ManagerContext]
    ) -> dict[str, Any] | None:
        session_id = runtime.context.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id is required")

        self.reply_turn[session_id] = 0
        self.display_messages[session_id] = []

        inputs = TypeAdapter(list[UserMessage]).validate_python(
            state["inputs"] or state["messages"]
        )[-self.max_history_window :]
        currents = [
            HumanMessage(
                content=item.as_content(),
                additional_kwargs={"raw": item},
            )
            for item in inputs
        ]
        history_messages = await self.get_session_history(session_id).aget_messages()

        if not self.real_average_count[session_id]:
            reply_counts: list[int] = [
                (content_to_text(message.content).strip().count("\n") + 1)
                for message in history_messages
                if isinstance(message, AIMessage)
            ]
            self.real_average_count[session_id] = (
                float(
                    pd.Series(reply_counts)
                    .ewm(alpha=self.reply_count_alpha)
                    .mean()
                    .iloc[-1]
                )
                if len(reply_counts) > 0
                else self.average_reply_count
            )
        if not self.history_caches[session_id]:
            window = min(
                self.max_history_window - len(currents), self.min_history_window
            )
            self.history_caches[session_id] = history_messages[-window:]
        messages = [
            HumanMessage(
                content=f"<bot-message user=可不>\n{content_to_text(message.content)}\n</bot-message>"
            )
            if isinstance(message, AIMessage)
            else message
            for message in self.history_caches[session_id] + currents
        ]
        return {
            **{
                key: value
                for key, value in state.items()
                if key not in {"inputs", "outputs", "histories"}
            },
            "messages": messages,
            "currents": currents,
            "histories": history_messages,
            "inputs": inputs,
            "outputs": [],
            "summary_pruned_messages": self.summary_pruned_messages.pop(
                session_id, None
            )
            or None,
        }

    @override
    async def aafter_agent(
        self, state: ManagerState, runtime: Runtime[ManagerContext]
    ) -> dict[str, Any] | None:
        _ = state
        self.display_messages[runtime.context["session_id"]] = []
        self.reply_turn[runtime.context["session_id"]] = 0

    @override
    @hook_config(can_jump_to=["end"])
    async def abefore_model(
        self, state: ManagerState, runtime: Runtime[ManagerContext]
    ) -> dict[str, Any] | None:
        _ = runtime
        stop_tags = (
            {"finish", "stop", "reply"} if self.stop_when_reply else {"finish", "stop"}
        )
        if any(o["type"] in stop_tags for o in state["outputs"]):
            return {"jump_to": "end"}
        messgaes = state["messages"]
        if sum(isinstance(x, AIMessage) for x in messgaes) >= self.max_turns:
            reason = "model has reached the maximum number of rounds limit."
        elif sum(isinstance(x, ToolMessage) for x in messgaes) >= self.max_tool_calls:
            reason = "model has reached the maximum number of tool calls limit."
        elif any(isinstance(x, AIMessage) and not x.tool_calls for x in messgaes):
            reason = "model returned content but did not invoke the tool."
        else:
            return None
        return {
            "outputs": [OutputMessage(type="stop", data={"reason": reason})],
            "jump_to": "end",
        }

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest[ManagerContext],
        handler: Callable[
            [ModelRequest[ManagerContext]], Awaitable[ModelResponse[Any]]
        ],
    ) -> ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]:
        request = request.override(
            messages=cast(
                "list[AnyMessage]",
                self.display_messages[request.runtime.context["session_id"]],
            )
            + request.messages
        )
        resp = await handler(request)
        for _ in range(self.mas_retries - 1):
            last = resp.result[-1]
            if not (isinstance(last, AIMessage) and last.content):
                break
            resp = await handler(request)
        return resp

    def _get_output_messages(
        self, output_val: str | BaseMessage | Sequence[BaseMessage] | dict
    ) -> list[BaseMessage]:
        if isinstance(output_val, dict):
            key = next(iter(output_val.keys())) if len(output_val) == 1 else "output"
            if key not in output_val and "generations" in output_val:
                output_val = output_val["generations"][0][0]["message"]
            else:
                output_val = output_val[key]
        if isinstance(output_val, str):
            return [AIMessage(content=output_val)]
        if isinstance(output_val, BaseMessage):
            return [output_val]
        if isinstance(output_val, (list, tuple)):
            return list(output_val)
        msg = (
            f"Expected str, BaseMessage, list[BaseMessage], or tuple[BaseMessage]. "
            f"Got {output_val}."
        )
        raise ValueError(msg)

    def _get_chat_app(
        self,
        reply_model: BaseChatModel,
    ) -> Runnable[dict[str, Any], str]:  # noqa: PLR0915
        def _handle_prompt(
            payload: dict[str, Any], config: RunnableConfig
        ) -> dict[str, Any]:
            assert "configurable" in config
            social_goal = str(
                (payload.get("social_action") or {}).get("social_goal") or ""
            )
            needs_complete_answer = social_goal in {
                "answer",
                "comfort",
                "clarify",
                "correct",
                "repair",
            }
            if (
                not needs_complete_answer
                and
                self.real_average_count[config["configurable"]["session_id"]]
                >= self.average_reply_count
            ):
                payload["reply_style"] = LESS_REPLY
            else:
                payload["reply_style"] = MORE_REPLY
            payload.setdefault("top_messages", [])
            payload.setdefault("bottom_messages", [])
            return payload

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    GROUP_SYSTEM_PROMPT.format(bot_name=BOT_NAME, identity=IDENTITY),
                ),
                MessagesPlaceholder("top_messages"),
                MessagesPlaceholder("history"),
                MessagesPlaceholder("messages"),
                MessagesPlaceholder("bottom_messages"),
                ("human", REPLY_USER_PROMPT),
            ]
        )
        return (
            RunnableLambda(_handle_prompt)
            | (prompt | reply_model).with_alisteners(on_end=self._aexit_history)
            | RunnableGenerator(to_reply)
        )

    @staticmethod
    def _stamp_history_messages(
        messages: Sequence[BaseMessage],
        *,
        timestamp: datetime | None = None,
    ) -> list[BaseMessage]:
        history_timestamp = (timestamp or datetime.now(tz=MODEL_VISIBLE_TZ)).isoformat()
        stamped_messages: list[BaseMessage] = []
        for message in messages:
            if isinstance(message, AIMessage):
                stamped_messages.append(
                    message.model_copy(
                        update={
                            "additional_kwargs": {
                                **message.additional_kwargs,
                                "history_timestamp": (
                                    message.additional_kwargs.get("history_timestamp")
                                    or history_timestamp
                                ),
                            }
                        }
                    )
                )
                continue
            stamped_messages.append(message)
        return stamped_messages

    async def _aexit_history(self, run: Run, config: RunnableConfig) -> None:
        new_messages: list[BaseMessage] = []
        input_essages = run.inputs.get("messages", [])
        output_val = load(run.outputs, allowed_objects="messages")
        output_messages = self._get_output_messages(output_val)
        output_messages = self._stamp_history_messages(output_messages)

        new_messages = input_essages + output_messages
        configurable = config.get("configurable", {})
        if "session_id" not in configurable:
            raise ValueError("session_id is required in configurable")
        session_id = configurable["session_id"]

        await self.get_session_history(session_id).aadd_messages(new_messages)
        if (
            len(self.history_caches[session_id]) + len(new_messages)
            > self.max_history_window
        ):
            combined_messages = self.history_caches[session_id] + new_messages
            kept_messages = combined_messages[-self.min_history_window :]
            pruned_messages = combined_messages[: -self.min_history_window]
            self.history_caches[session_id] = kept_messages
            if pruned_messages:
                self.summary_pruned_messages[session_id].extend(pruned_messages)
        else:
            self.history_caches[session_id] += new_messages
        for output in output_messages:
            if isinstance(output, AIMessage):
                self.real_average_count[session_id] = (
                    self.reply_count_alpha
                    * (content_to_text(output.content).strip().count("\n") + 1)
                    + (1 - self.reply_count_alpha) * self.real_average_count[session_id]
                )
