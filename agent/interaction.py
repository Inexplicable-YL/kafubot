import itertools
import os
from collections.abc import Awaitable, Callable
from datetime import datetime
from functools import cache
from typing import Any, Literal

from langchain.agents.middleware import (
    AgentMiddleware,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
    hook_config,
)
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
from agent.commons.meme import MemeResult, search_memes
from agent.history import get_session_history
from agent.message import QQMessage, QQMessageSegment
from agent.prompts.prompt import (
    BOT_NAME,
    GROUP_SYSTEM_PROMPT,
    IDENTITY,
    LESS_REPLY,
    MORE_REPLY,
    REPLY_USER_PROMPT,
)
from agent.utils import to_reply


@cache
def get_chat_app(
    model: str = "deepseek-v4-flash",
    *,
    history_window: int = 20,
    thinking: bool = True,
    reasoning_effort: Literal["high", "max"] = "max",
) -> Runnable[dict[str, Any], str]:  # noqa: PLR0915
    def _normalize_input(payload: dict[str, Any]) -> dict[str, Any]:
        prompt_variables = {
            key: value for key, value in payload.items() if key not in {"messages"}
        }
        current_messages = [
            HumanMessage(
                content=item.as_plain_content(),
                additional_kwargs={"raw": item},
            )
            for item in TypeAdapter(list[UserMessage]).validate_python(
                payload["messages"]
            )
        ]

        return {
            **prompt_variables,
            "current_messages": current_messages,
        }

    def _handle_history(payload: dict[str, Any]) -> dict[str, Any]:
        final_messages = (
            payload.get("history", []) + payload.get("current_messages", [])
        )[-history_window:]
        if payload.get("real_average_count", 1.5) > payload.get(
            "average_reply_count", 1.5
        ):
            payload["reply_style"] = LESS_REPLY
        else:
            payload["reply_style"] = MORE_REPLY
        return payload | {"final_messages": final_messages}

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                GROUP_SYSTEM_PROMPT.format(bot_name=BOT_NAME, identity=IDENTITY),
            ),
            MessagesPlaceholder("final_messages"),
            ("human", REPLY_USER_PROMPT),
        ]
    )
    llm = ChatDeepSeek(
        model=model,
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=1.2,
        max_retries=2,
        reasoning_effort=reasoning_effort,
        extra_body={
            "thinking": {
                "type": "enabled" if thinking else "disabled",
            }
        },
    )
    chain_with_history = RunnableWithMessageHistory(
        RunnableLambda(_handle_history) | prompt | llm,
        get_session_history,
        input_messages_key="current_messages",
        history_messages_key="history",
    )

    return (
        RunnableLambda(_normalize_input)
        | chain_with_history
        | RunnableGenerator(to_reply)
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


@tool(
    args_schema=ReplyInput,
    description="根据当前思考生成并发送一条可见回复。每一次只能回复一条消息。你需要针对focus指向的消息进行回复。",
)
async def reply(
    focus: str,
    use_reply: bool,
    reference_info: str,
    language_style: str,
    runtime: ToolRuntime[ManagerContext, ManagerState],
) -> Any:
    """调用reply工具实现对用户进行回复。"""
    focus_output: str | None = None
    for msg in runtime.state["full_messages"]:
        if (
            isinstance(msg, HumanMessage)
            and (group_msg := msg.additional_kwargs.get("raw"))
            and isinstance(group_msg, UserMessage)
            and group_msg.message_id == focus.strip()
        ):
            focus_output = (
                f"- 时间：{group_msg.timestamp.astimezone(MODEL_VISIBLE_TZ).strftime('%Y-%m-%d %H:%M:%S')}\n"
                "- 发送人：{escape(group_msg.user, quote=True)}>\n"
                "- 消息内容：{group_msg.message.get_msgcode()}\n"
            )
            if use_reply:
                focus_output += (
                    "- 是否将被引用：是\n"
                    "这条消息已经是引用回复的消息。请不要at发送人。以避免重复。"
                )
            break

    if not focus_output:
        return "请检查 `focus` 的 message_id 是否正确。"
    if not reference_info:
        return "`reply` 工具需要填充 `reference_info` 参数。"
    if not language_style:
        return "`reply` 工具需要填充 `language_style` 参数。"
    full_text = ""
    inputs = runtime.state["inputs"]
    reasoning_content: str = next(
        (
            str(m.additional_kwargs.get("reasoning_content"))
            for m in reversed(runtime.state["messages"])
            if isinstance(m, AIMessage) and m.additional_kwargs.get("reasoning_content")
        ),
        "",
    )
    async for reply in get_chat_app().astream(
        {
            "messages": inputs,
            "focus_message": focus_output,
            "outputs": runtime.state["outputs"],
            "reference_info": reference_info,
            "language_style": language_style,
            "reasoning_content": reasoning_content,
            "real_average_count": runtime.state.get("real_average_count", 1.5),
            "average_reply_count": runtime.context["average_reply_count"],
            "time": datetime.now(tz=MODEL_VISIBLE_TZ).strftime("%Y-%m-%d %H:%M:%S"),
        },
        config={"configurable": {"session_id": runtime.context["session_id"]}},
    ):
        if reply is not None and (reply_msg := reply.strip()):
            raw_cq_msg = await QQMessage.from_str(reply_msg).get_cqhttp_message(inputs)
            full_text += reply_msg + "\n"
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
                    content="Bot 回复：" + full_text,
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


class SearchMemeInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    content: str = Field(description="你希望发送的表情包应该呈现出的内容。")


@tool(
    args_schema=SearchMemeInput,
    description="从表情包库搜索想要的表情包。",
)
async def search_meme(
    content: str,
    runtime: ToolRuntime[ManagerContext, ManagerState],
) -> Any:
    """从表情包库搜索想要的表情包。"""
    results = await search_memes(content)
    if not results:
        return "未找到匹配的表情包，请尝试更完整详细的搜索语句。"

    def _gen_id() -> str:
        runtime.state.setdefault("meme_id", itertools.count(0))
        assert "meme_id" in runtime.state
        return str(next(runtime.state["meme_id"]))

    found_meme_dict: dict[str, MemeResult] = {}
    for msg in runtime.state["messages"]:
        if (
            isinstance(msg, ToolMessage)
            and (meme_dict := msg.additional_kwargs.get("search_meme_results", {}))
            and isinstance(meme_dict, dict)
        ):
            found_meme_dict.update(meme_dict)
    found_meme_analysis = {r.analysis: k for k, r in found_meme_dict.items()}

    newly_result_map, old_result_map = {}, {}
    for result in results:
        if result.analysis in found_meme_analysis:
            old_result_map[found_meme_analysis[result.analysis]] = result
        else:
            newly_result_map[_gen_id()] = result

    result_map: dict[str, MemeResult] = {**old_result_map, **newly_result_map}

    content_lines: list[str] = [
        f"已找到 {len(results)} 个匹配的表情包，请从下列中选择你希望发送的表情包：",
        *[
            f"- [{meme_id}]{'（本次新发现）' if meme_id in newly_result_map else '（此前已发现）'} {meme_result.analysis}"
            for meme_id, meme_result in result_map.items()
        ],
    ]
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content="\n".join(content_lines),
                    tool_call_id=runtime.tool_call_id,
                    additional_kwargs={"search_meme_results": newly_result_map},
                )
            ],
        }
    )


class SendMemeInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    meme_id: str = Field(
        description="你希望发送的表情包的索引。索引与 `search_meme` 的输出对应，请参考 `search_meme` 的输出。"
    )


@tool(
    args_schema=SendMemeInput,
    description="向聊天中发送一条表情包。在发送表情包前，必须先使用 `search_meme` 搜索表情包。",
)
async def send_meme(
    meme_id: str,
    runtime: ToolRuntime[ManagerContext, ManagerState],
) -> Any:
    """向聊天中发送一条表情包。"""
    normalized_meme_id = meme_id.replace("[", "").replace("]", "").casefold().strip()
    found_meme_dict: dict[str, MemeResult] = {}
    for msg in runtime.state["messages"]:
        if (
            isinstance(msg, ToolMessage)
            and (meme_dict := msg.additional_kwargs.get("search_meme_results", {}))
            and isinstance(meme_dict, dict)
        ):
            found_meme_dict.update(meme_dict)
    result = found_meme_dict.get(normalized_meme_id)
    if not result:
        return "无法通过索引找到对应的表情包，请检查索引是否正确。"
    await runtime.context["node"].reply(
        CQHTTPMessageSegment.image(result.base64, sub_type=1)
    )
    await get_session_history(runtime.context["session_id"]).aadd_message(
        AIMessage(
            content=QQMessageSegment.meme(
                content=result.analysis,
            ).get_msgcode(),
        )
    )
    return Command(
        update={
            "outputs": [OutputMessage(type="meme", data={"content": result.analysis})],
            "messages": [
                ToolMessage(
                    content=f"已发送表情包：[{meme_id}] " + result.analysis,
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


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


class InteractionMiddleware(AgentMiddleware[ManagerState, ManagerContext, Any]):
    tools = [reply, finish, search_meme, send_meme]

    def __init__(
        self,
        max_turns: int = 10,
        max_tool_calls: int = 10,
        mas_retries: int = 3,
        **_: Any,
    ) -> None:
        self.max_turns = max_turns
        self.max_tool_calls = max_tool_calls
        self.mas_retries = mas_retries

    @hook_config(can_jump_to=["end"])
    async def abefore_model(
        self, state: ManagerState, runtime: Runtime[ManagerContext]
    ) -> dict[str, Any] | None:
        _ = runtime
        if any(o["type"] in {"finish", "stop"} for o in state["outputs"]):
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
        resp = await handler(request)
        for _ in range(self.mas_retries - 1):
            last = resp.result[-1]
            if not (isinstance(last, AIMessage) and last.content):
                break
            resp = await handler(request)
        return resp
