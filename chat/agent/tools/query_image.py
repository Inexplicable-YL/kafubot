import os
from functools import cache
from typing import Any, cast

from dotenv import load_dotenv
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from chat.agent.base import ManagerContext, ManagerState, UserMessage

load_dotenv()


IMAGE_ASK_SYSTEM_PROMPT = """
你是一个图像分析助手。请根据提供的图片和用户的问题，用简洁准确的平文本语言进行回答。
不要书写markdown格式，仅用一段凝练但能完美回答用户问题的文本回复。
"""


@cache
def get_image_analyzer() -> Runnable[dict[str, Any], str]:
    def _prepare_messages(x: dict[str, Any]) -> list:
        images = x.get("images")
        if not images or not isinstance(images, list):
            raise ValueError("Invalid image input: expected a non-empty list")
        query = x.get("query")
        if not query or not isinstance(query, str):
            raise ValueError("Invalid query input: expected a non-empty string")

        content = [
            *(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}",
                    },
                }
                for img_b64 in images
            ),
            {"type": "text", "text": query},
        ]

        return [
            SystemMessage(content=IMAGE_ASK_SYSTEM_PROMPT),
            HumanMessage(content=content),
        ]

    def _compact_output(text: str) -> str:
        return " ".join(text.strip().split())

    model = ChatOpenAI(
        model="kimi-k2.6",
        api_key=SecretStr(os.getenv("KIMI_API_KEY", "")),
        base_url=os.getenv("KIMI_BASE_URL"),
        temperature=1,
        max_retries=2,
    )

    prepare = RunnableLambda(_prepare_messages)
    parser = StrOutputParser() | RunnableLambda(_compact_output)

    return prepare | model | parser


class QueryImageInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    message_id: str = Field(description="要问询的包含图片的目标用户消息的 message_id。")
    query: str = Field(
        description="对于指定的消息中的图片进行询问的问题，使用平文本格式。"
    )
    runtime: ToolRuntime = Field(exclude=True)


@tool(
    args_schema=QueryImageInput,
    description="对于某条消息中的图片进行详细询问。",
)
def query_image(message_id: str, query: str, runtime: ToolRuntime) -> str:
    _runtime = cast("ToolRuntime[ManagerContext, ManagerState]", runtime)
    message: UserMessage | None = None
    for msg in _runtime.state["full_messages"]:
        if (
            isinstance(msg, HumanMessage)
            and (group_msg := msg.additional_kwargs.get("raw"))
            and isinstance(group_msg, UserMessage)
            and group_msg.message_id == message_id.strip()
        ):
            message = group_msg
            break
    if not message:
        return "没有找到对应的消息。请检查 message_id 是否正确。"
    if not message.images:
        return "指定 message_id 指向的消息没有图片。"
    return "对指定消息的图片进行询问的结果为：\n" + get_image_analyzer().invoke(
        {"images": [img.base64 for img, _ in message.images], "query": query}
    )
