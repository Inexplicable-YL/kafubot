import os
from datetime import UTC, datetime
from functools import cache
from typing import Any, cast
from zoneinfo import ZoneInfo

from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    Runnable,
    RunnableGenerator,
    RunnableLambda,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_deepseek import ChatDeepSeek
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from chat.agent.base import (
    MODEL_VISIBLE_TZ,
    ManagerContext,
    ManagerState,
    OutputMessage,
    UserMessage,
)
from chat.agent.history import get_session_history
from chat.agent.prompt import GROUP_SYSTEM_PROMPT, IDENTITY, REPLY_USER_PROMPT
from chat.message import QQMessage
from chat.utils import to_reply

CHAT_DEEPSEEK_MODEL = "deepseek-v4-flash"
HISTORY_WINDOW = 20


@cache
def get_chat_app() -> Runnable[dict[str, Any], str]:  # noqa: PLR0915
    def _normalize_input(payload: dict[str, Any]) -> dict[str, Any]:
        reasoning_effort = payload.get("reasoning_effort", "high")
        prompt_variables = {
            key: value
            for key, value in payload.items()
            if key not in {"messages", "thinking", "reasoning_effort"}
        }

        if reasoning_effort not in {"high", "max"}:
            raise ValueError("reasoning_effort must be 'high' or 'max'")

        return {
            **prompt_variables,
            "current_messages": [
                HumanMessage(
                    content=item.as_plain_content(timezone=MODEL_VISIBLE_TZ),
                    additional_kwargs={"raw": item},
                )
                for item in TypeAdapter(list[UserMessage]).validate_python(
                    payload["messages"]
                )
            ],
            "thinking": bool(payload.get("thinking", False)),
            "reasoning_effort": reasoning_effort,
        }

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", GROUP_SYSTEM_PROMPT.format(bot_name="可不", identity=IDENTITY)),
            MessagesPlaceholder("final_messages"),
            ("human", REPLY_USER_PROMPT),
        ]
    )

    def _chat_chain_for_payload(
        payload: dict[str, Any],
    ) -> Runnable[dict[str, Any], AIMessage]:
        if bool(payload["thinking"]):
            runtime_kwargs = {
                "reasoning_effort": payload["reasoning_effort"]
                if payload["reasoning_effort"] == "max"
                else "high",
                "extra_body": {
                    "thinking": {
                        "type": "enabled",
                    }
                },
            }
        else:
            runtime_kwargs = {
                "extra_body": {
                    "thinking": {
                        "type": "disabled",
                    }
                },
            }

        def _cut_history(payload: dict[str, Any]) -> dict[str, Any]:
            return {
                **payload,
                "final_messages": (payload["history"] + payload["current_messages"])[
                    -HISTORY_WINDOW:
                ],
            }

        model = ChatDeepSeek(
            model=CHAT_DEEPSEEK_MODEL,
            base_url=os.getenv("DEEPSEEK_BASE_URL"),
            temperature=1.2,
            max_retries=2,
        ).bind(**runtime_kwargs)
        return cast(
            "Runnable[dict[str, Any], AIMessage]",
            RunnableLambda(_cut_history) | prompt | model,
        )

    chain_with_history = RunnableWithMessageHistory(
        RunnableLambda(_chat_chain_for_payload),
        get_session_history,
        input_messages_key="current_messages",
        history_messages_key="history",
    )

    return (
        RunnableLambda(_normalize_input)
        | chain_with_history
        | RunnableGenerator(to_reply)
    )


MIN_FOCUS_LENGTH = 6


class ReplyInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    focus: list[str] = Field(
        description="要回复1或多条目标用户消息的具体发送时间，每一条的格式为“YY-mm-dd HH:MM:SS”。"
    )
    reference_info: str = Field(
        description="有助于回复的信息，之前搜集得到的事实性信息，记忆等，使用平文本格式。"
    )
    language_style: str = Field(
        description="你需要指导回复的语言风格，例如是反骨、拌嘴、可爱地攻击（如虾头、变态等）、傲娇属性，或者是温柔的回复或安慰等，使用平文本格式。"
    )
    runtime: ToolRuntime = Field(exclude=True)


@tool(args_schema=ReplyInput, description="根据当前思考生成并发送一条可见回复。")
async def reply(
    focus: list[str],
    reference_info: str,
    language_style: str,
    runtime: ToolRuntime,
) -> Any:
    """调用reply工具实现对用户进行回复。"""
    _runtime = cast("ToolRuntime[ManagerContext, ManagerState]", runtime)
    focus_output: list[str] = []
    for time in focus:
        for msg in _runtime.state["full_messages"]:
            if (
                isinstance(msg, HumanMessage)
                and (group_msg := msg.additional_kwargs.get("raw"))
                and isinstance(group_msg, UserMessage)
            ):
                time_text = (
                    group_msg.timestamp.replace(tzinfo=UTC)
                    .astimezone(ZoneInfo("Asia/Shanghai"))
                    .strftime("%H:%M:%S")
                )
                if len(time.strip()) > MIN_FOCUS_LENGTH and time_text in time.strip():
                    focus_output.append(group_msg.as_content())
    if len(focus_output) != len(focus):
        return "请检查 `focus` 的时间格式是否正确。"
    if not reference_info:
        return "`reply` 工具需要填充 `reference_info` 参数。"
    if not language_style:
        return "`reply` 工具需要填充 `language_style` 参数。"
    focus_messages = "\n".join(focus_output)
    full_text = ""
    inputs = _runtime.state["inputs"]
    async for reply in get_chat_app().astream(
        {
            "messages": inputs,
            "focus_messages": focus_messages,
            "reference_info": reference_info,
            "language_style": language_style,
            "time": datetime.now(tz=MODEL_VISIBLE_TZ).strftime("%Y-%m-%d %H:%M:%S"),
            "thinking": True,
            "reasoning_effort": "max",
        },
        config={"configurable": {"session_id": _runtime.context["session_id"]}},
    ):
        if reply is not None and (reply_msg := reply.strip()):
            raw_cq_msg = await QQMessage.from_str(reply_msg).get_cqhttp_message(inputs)
            full_text += reply_msg + "\n"
            for seg in raw_cq_msg:
                if seg.type == "image":
                    await _runtime.context["node"].reply(seg)
                    break
            else:
                await _runtime.context["node"].reply(raw_cq_msg)
    return Command(
        update={
            "outputs": _runtime.state["outputs"]
            + [
                OutputMessage(
                    type="reply",
                    data={"full_text": full_text},
                )
            ],
            "messages": [
                ToolMessage(
                    content="Bot 回复：" + full_text,
                    tool_call_id=_runtime.tool_call_id,
                )
            ],
        }
    )
