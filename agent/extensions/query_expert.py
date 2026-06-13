import os
from functools import cache
from typing import Any

from dotenv import load_dotenv
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field

from agent.base import ManagerContext, ManagerState, UserMessage

load_dotenv()

MAX_HISTORY_COUNT = 10

DEFAULT_MODEL = "gemini-3.1-pro-preview"


SYSTEM_PROMPT = """
你是一位资深全领域专业顾问，具备跨学科的深厚知识储备和结构化问题解决能力。

## 核心职责
基于用户提供的上下文信息，给出准确、深入、可直接执行的专业分析与建议。

## 思维框架（回答前请内在遵循）
1. **问题解构**：识别查询的核心诉求、隐含约束和关键变量。
2. **上下文对齐**：评估对话历史中的脉络与关联参考中的具体信息，确定最相关的知识域。
3. **推理与综合**：基于事实进行逻辑推导，避免臆测；若信息不足，明确说明缺失点而非编造。
4. **输出构建**：以用户最易理解的方式组织答案，优先给出结论，再视需要展开细节。

## 输出规范
- 使用中文，语言专业且自然，避免机械感。
- 优先采用"结论先行 + 要点展开"的结构。语言简洁清晰，不要过于冗长。
- 若引用关联参考中的具体信息，请自然融入叙述，无需标注来源标签。
- 禁止输出 XML 标签、HTML 标签或任何格式标记。
- 保持简洁，去除冗余修饰，确保信息密度。
- 若上下文不足以回答问题，直接说明信息缺口，可基于常识给出方向性建议（需标注为"基于一般性知识推断"）。
""".strip()

CONTEXT_TEMPLATE = """
<dialogue_history>
{history}
</dialogue_history>

<<related_references>
{related}
</related_references>

<current_query>
{query}
</current_query>


## 上下文处理规则
1. **优先级**：<related_references> 中的信息与当前问题直接相关，应优先参考；<<dialogue_history> 提供背景脉络，用于理解用户偏好和前文逻辑。
2. **冲突处理**：若 <related_references> 与 <dialogue_history> 存在信息冲突，以 <related_references> 为准。
3. **缺失处理**：若上下文不足以完整回答问题，请明确说明信息缺口，并基于已有信息给出最佳推断（需标注为推断）。
4. **聚焦原则**：严格围绕 <current_query> 作答，不发散至无关领域，不重复用户问题。

请直接给出专业回答。""".strip()


@cache
def get_expert() -> Runnable[dict[str, Any], str]:
    def _compact_output(text: str) -> str:
        return " ".join(text.strip().split())

    model = ChatOpenAI(
        model=DEFAULT_MODEL,
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=0.6,
        max_retries=2,
    )

    prompt = ChatPromptTemplate(
        messages=[
            ("system", SYSTEM_PROMPT),
            ("human", CONTEXT_TEMPLATE),
        ]
    )
    parser = StrOutputParser() | RunnableLambda(_compact_output)

    return prompt | model | parser


class QuerExpertInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    query: str = Field(
        description="你需要进行询问的内容。请用简洁的语言描述你想要了解什么，使用平文本格式。"
    )
    related_msg_ids: list[str] = Field(
        description="问题相关的消息的 message_id 列表，这是可选的，不需要严格提供。",
        default_factory=list,
    )


@tool(
    args_schema=QuerExpertInput,
    description="对于某条消息中的图片进行详细询问。",
)
def query_expert(
    query: str,
    related_msg_ids: list[str],
    runtime: ToolRuntime[ManagerContext, ManagerState],
) -> str:
    query = query.strip()
    if not query:
        return "请输入需要进行询问的问题。"

    all_messages = [
        group_msg
        for msg in runtime.state["histories"]
        if isinstance(msg, HumanMessage)
        and (group_msg := msg.additional_kwargs.get("raw"))
        and isinstance(group_msg, UserMessage)
    ] + runtime.state["inputs"]

    recent_history = all_messages[-MAX_HISTORY_COUNT:]
    history_block = "\n".join(
        f"[{i + 1}] {msg.as_plain_content()}" for i, msg in enumerate(recent_history)
    )

    related_messages = list(
        reversed(
            [
                msg
                for msg in reversed(all_messages)
                if msg.message_id in [id.strip() for id in related_msg_ids]
            ]
        )
    )
    related_block = (
        "\n".join(f"• {msg.as_plain_content()}" for msg in related_messages)
        if related_messages
        else "（未提供直接关联的参考消息）"
    )

    return get_expert().invoke(
        {
            "history": history_block,
            "related": related_block,
            "query": query,
        }
    )
