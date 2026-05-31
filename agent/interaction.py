from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime
from html import escape
from typing import Any, NotRequired, TypedDict, cast

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
from agent.utils import content_to_text, to_reply


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


class ReplyInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    focus: str = Field(description="要回复的目标用户消息的 message_id。")
    use_reply: bool = Field(
        description="是否使用引用回复模式。在消息较多时，可以使用引用回复模式。其会自动引用回复focus指向的消息。只在需要时设置为True。"
    )
    reference_info: str = Field(
        description="有助于回复的信息，之前搜集得到的事实性信息，记忆等，使用平文本格式。需要较为详细地陈述。你需要针对focus指向的消息进行回复。"
    )
    language_style: str = Field(
        description="你需要指导回复的语言风格，例如是反骨、拌嘴、可爱地攻击（如虾头、变态等）、傲娇属性，或者是温柔的回复或安慰等，使用平文本格式。"
    )
    runtime: ToolRuntime = Field(exclude=True)


class GetEarlyMessagesInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    limit: int = Field(default=20, description="要显示的消息数量限制。", gt=1, le=20)
    runtime: ToolRuntime = Field(exclude=True)


class InteractionMiddleware(AgentMiddleware[ManagerState, ManagerContext, Any]):
    reply_count_alpha = 0.1

    get_session_history: Callable[..., BaseChatMessageHistory]
    history_caches: dict[str, list[BaseMessage]]
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
        self.reply_turn = defaultdict(lambda: 0)
        self.real_average_count = defaultdict(lambda: 0)
        self.display_messages = defaultdict(list)

        self.tools = [
            finish,
            tool(
                "reply",
                args_schema=ReplyInput,
                description="根据当前思考生成并发送一条可见回复。每一次只能回复一条消息。你需要针对focus指向的消息进行回复。",
            )(self.reply),
            tool(
                "get_early_messages",
                args_schema=GetEarlyMessagesInput,
                description="获取比当前可见的消息更早的消息，并自动添加到上下文中。需要使用`limit`限制显示的消息数量。",
            )(self.get_early_messages),
        ]

    async def reply(
        self,
        focus: str,
        use_reply: bool,
        reference_info: str,
        language_style: str,
        runtime: ToolRuntime[ManagerContext, ManagerState],
    ) -> Any:
        """调用reply工具实现对用户进行回复。"""
        if (
            self.max_reply_per_turn is not None
            and self.reply_turn[runtime.context["session_id"]]
            >= self.max_reply_per_turn
        ):
            return "本轮系统允许的回复次数已用完，请等待下一轮。"
        focus_output: str | None = None
        meme_msgs: list[str] = [
            str(content).replace("已发送表情包", "当前系统已自动发送表情包")
            for output in runtime.state["outputs"]
            if output["type"] == "meme" and (content := output["data"].get("content"))
        ]
        for msg in runtime.state["inputs"]:
            if msg.message_id == focus.strip():
                focus_output = (
                    f"- 时间：{msg.timestamp.astimezone(MODEL_VISIBLE_TZ).strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"- 发送人：{escape(msg.user, quote=True)}>\n"
                    f"- 消息内容：{msg.message.get_msgcode()}\n"
                )
                if use_reply:
                    focus_output += (
                        "- 是否将被引用：是\n"
                        "这条消息已经是引用回复的消息。请不要at发送人。以避免重复。"
                    )
                if meme_msgs:
                    focus_output += "-系统是否自动发送表情包：是\n" + "\n".join(
                        meme_msgs
                    )
                break

        if not focus_output:
            return "请检查 `focus` 的 message_id 是否正确。"
        if not reference_info:
            return "`reply` 工具需要填充 `reference_info` 参数。"
        if not language_style:
            return "`reply` 工具需要填充 `language_style` 参数。"
        reasoning_content: str = next(
            (
                str(m.additional_kwargs.get("reasoning_content"))
                for m in reversed(runtime.state["messages"])
                if isinstance(m, AIMessage)
                and m.additional_kwargs.get("reasoning_content")
            ),
            "",
        )

        full_text = ""
        async for reply in self.chat_app.astream(
            {
                "messages": runtime.state["currents"],
                "history": self.history_caches[runtime.context["session_id"]],
                "focus_message": focus_output,
                "reference_info": reference_info,
                "language_style": language_style,
                "reasoning_content": reasoning_content,
                "time": datetime.now(tz=MODEL_VISIBLE_TZ).strftime("%Y-%m-%d %H:%M:%S"),
            },
            config={"configurable": {"session_id": runtime.context["session_id"]}},
        ):
            if reply is not None and (reply_msg := reply.strip()):
                raw_msg = QQMessage.from_str(reply_msg)
                msg = QQMessage(filter(lambda x: x.type in {"at", "text"}, raw_msg))
                if not msg.get_plain_text().strip():
                    continue
                raw_cq_msg = await msg.get_cqhttp_message(runtime.state["inputs"])
                full_text += msg.get_msgcode() + "\n"
                for seg in raw_cq_msg:
                    if seg.type == "image":
                        await runtime.context["node"].reply(seg)
                        break
                else:
                    if use_reply:
                        await runtime.context["node"].reply(
                            CQHTTPMessageSegment.reply(int(focus.strip())) + raw_cq_msg
                        )
                        use_reply = False
                    else:
                        await runtime.context["node"].reply(raw_cq_msg)
        full_text = full_text.strip()
        if not full_text:
            return "回复失败：无法生成回复。"
        self.reply_turn[runtime.context["session_id"]] += 1
        return Command(
            update={
                "outputs": [
                    OutputMessage(
                        type="reply",
                        data={"full_text": full_text},
                    )
                ],
                "messages": [
                    ToolMessage(
                        content="已回复消息：\n" + full_text,
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
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
        }

    async def aafter_agent(
        self, state: ManagerState, runtime: Runtime[ManagerContext]
    ) -> dict[str, Any] | None:
        _ = state
        self.display_messages[runtime.context["session_id"]] = []
        self.reply_turn[runtime.context["session_id"]] = 0

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

    def _get_input_messages(
        self, input_val: str | BaseMessage | Sequence[BaseMessage] | dict
    ) -> list[BaseMessage]:
        if isinstance(input_val, dict):
            input_val = input_val["messages"]
        if isinstance(input_val, str):
            return [HumanMessage(content=input_val)]
        if isinstance(input_val, BaseMessage):
            return [input_val]
        if isinstance(input_val, (list, tuple)):
            if len(input_val) == 0:
                return list(input_val)
            if isinstance(input_val[0], list):
                if len(input_val) != 1:
                    msg = f"Expected a single list of messages. Got {input_val}."
                    raise ValueError(msg)
                return input_val[0]
            return list(input_val)
        msg = (
            f"Expected str, BaseMessage, list[BaseMessage], or tuple[BaseMessage]. "
            f"Got {input_val}."
        )
        raise ValueError(msg)

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
            if (
                self.real_average_count[config["configurable"]["session_id"]]
                >= self.average_reply_count
            ):
                payload["reply_style"] = LESS_REPLY
            else:
                payload["reply_style"] = MORE_REPLY
            return payload

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    GROUP_SYSTEM_PROMPT.format(bot_name=BOT_NAME, identity=IDENTITY),
                ),
                MessagesPlaceholder("history"),
                MessagesPlaceholder("messages"),
                ("human", REPLY_USER_PROMPT),
            ]
        )
        return (
            RunnableLambda(_handle_prompt)
            | prompt
            | reply_model.with_alisteners(on_end=self._aexit_history)
            | RunnableGenerator(to_reply)
        )

    async def _aexit_history(self, run: Run, config: RunnableConfig) -> None:
        new_messages: list[BaseMessage] = []
        input_val = load(run.inputs, allowed_objects="messages")
        input_essages = self._get_input_messages(input_val)
        output_val = load(run.outputs, allowed_objects="messages")
        output_messages = self._get_output_messages(output_val)

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
            self.history_caches[session_id] = (
                self.history_caches[session_id] + new_messages
            )[-self.min_history_window :]
        else:
            self.history_caches[session_id] += new_messages
        for output in output_messages:
            if isinstance(output, AIMessage):
                self.real_average_count[session_id] = (
                    self.reply_count_alpha
                    * (content_to_text(output.content).strip().count("\n") + 1)
                    + (1 - self.reply_count_alpha) * self.real_average_count[session_id]
                )
