from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain.agents.middleware.types import ToolCallRequest
from langchain.messages import ToolMessage
from langchain.tools import BaseTool, ToolRuntime, tool
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from agent.base import ManagerContext, ManagerState

SYSTEM_REMINDER = """
<system-reminder>
以下工具当前未直接暴露给你，但可以通过 tool_search 工具发现并在后续轮次中使用：
{deferred_tools}

如需其中某个工具，请先调用 tool_search。tool_search 只负责发现工具，不直接执行业务。
</system-reminder>
"""


class SearchTool(BaseModel):
    query: str = Field(description="要搜索的工具名、前缀或关键词。")
    limit: int = Field(default=3, description="最多返回多少个匹配工具", gt=1, le=10)


class DeferredToolMiddleware(AgentMiddleware[ManagerState, ManagerContext, Any]):
    tools: Sequence[BaseTool]

    def __init__(
        self,
        deferred_tools: Sequence[BaseTool],
    ) -> None:
        self.deferred_tools_by_name = {tool.name: tool for tool in deferred_tools}
        self.deferred_tools = list(self.deferred_tools_by_name.values())
        self.tools = [
            tool(
                "search_tool",
                args_schema=SearchTool,
                description="在 deferred tools 列表中按名称或关键词搜索工具，并将命中的工具加入后续轮次的可用工具列表。",
            )(self._search_tool)
        ]

    def _get_display_tools_dict(self, messages: Sequence[Any]) -> dict[str, BaseTool]:
        display_tools_dict: dict[str, BaseTool] = {}
        for msg in messages:
            if not isinstance(msg, ToolMessage):
                continue
            if (
                names := msg.additional_kwargs.get("search_tool_result_names")
            ) and isinstance(names, list):
                display_tools_dict.update(
                    {
                        name: tool
                        for name in names
                        if isinstance(name, str)
                        and (tool := self.deferred_tools_by_name.get(name))
                    }
                )
            elif (
                results := msg.additional_kwargs.get("search_tool_results")
            ) and isinstance(results, list):
                display_tools_dict.update(
                    {tool.name: tool for tool in results if isinstance(tool, BaseTool)}
                )
        return display_tools_dict

    def _search_tool(
        self,
        query: str,
        limit: int,
        runtime: ToolRuntime[ManagerContext, ManagerState],
    ) -> Any:
        if not query.strip():
            return "tool_search 需要提供非空的 `query` 字符串参数。"

        display_tools_dict = self._get_display_tools_dict(
            runtime.state.get("messages", [])
        )
        normalized_query = query.strip().casefold()
        scored_matches: list[tuple[int, str, BaseTool]] = []
        query_terms = [
            term.casefold()
            for term in query.replace("_", " ").replace("-", " ").split()
            if term
        ]

        for tool_name, tool_spec in self.deferred_tools_by_name.items():
            lower_name = tool_name.casefold()
            lower_description = tool_spec.description.casefold()
            score = 0

            if normalized_query == lower_name:
                score += 1000
            if lower_name.startswith(normalized_query):
                score += 300
            if normalized_query in lower_name:
                score += 200
            if normalized_query in lower_description:
                score += 100

            for query_term in query_terms:
                if query_term in lower_name:
                    score += 25
                if query_term in lower_description:
                    score += 10

            if score <= 0:
                continue

            scored_matches.append((score, tool_name, tool_spec))

        scored_matches.sort(key=lambda item: (-item[0], item[1]))
        matched_tool_specs = [
            tool_spec for _, _, tool_spec in scored_matches[: max(1, limit)]
        ]
        matched_tool_names = [tool_spec.name for tool_spec in matched_tool_specs]
        newly_display_tool_names = [
            name for name in matched_tool_names if name not in display_tools_dict
        ]
        if not matched_tool_names:
            return (
                "未找到匹配的 deferred tools，请尝试更完整的工具名、前缀或其他关键词。"
            )

        content_lines: list[str] = [
            f"已找到 {len(matched_tool_names)} 个 deferred tools，它们会在后续轮次中加入可用工具列表：",
            *[
                f"- {tool_name}{'（本次新发现）' if tool_name in newly_display_tool_names else '（此前已发现）'}"
                for tool_name in matched_tool_names
            ],
        ]
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="\n".join(content_lines),
                        tool_call_id=runtime.tool_call_id,
                        additional_kwargs={
                            "search_tool_result_names": matched_tool_names
                        },
                    )
                ],
            }
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> Any:
        if request.tool is None:
            display_tools_by_name = self._get_display_tools_dict(
                request.state.get("messages", [])
            )
            if tool := display_tools_by_name.get(request.tool_call["name"]):
                return handler(request.override(tool=tool))
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[Any]],
    ) -> Any:
        if request.tool is None:
            display_tools_by_name = self._get_display_tools_dict(
                request.state.get("messages", [])
            )
            if tool := display_tools_by_name.get(request.tool_call["name"]):
                return await handler(request.override(tool=tool))
        return await handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest[ManagerContext],
        handler: Callable[
            [ModelRequest[ManagerContext]], Awaitable[ModelResponse[Any]]
        ],
    ) -> ModelResponse[Any] | AIMessage:
        display_tools_dict = self._get_display_tools_dict(request.messages)
        deferred_tools = list(
            {
                tool.name: tool
                for tool in self.deferred_tools
                if tool.name not in display_tools_dict
            }.values()
        )
        display_tools = list(display_tools_dict.values())
        remind_message = HumanMessage(
            content=SYSTEM_REMINDER.format(
                deferred_tools="\n".join(
                    [f"- {tool.name}: {tool.description}" for tool in deferred_tools]
                )
            )
        )
        return await handler(
            request.override(
                tools=request.tools + display_tools,
                messages=request.messages + [remind_message],
            )
        )
