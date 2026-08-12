# Formula Chat Client - OpenAI chat with official tools
# Uses MOONSHOT_BASE_URL and MOONSHOT_API_KEY for OpenAI client

import json
import os
from typing import Any

import httpx
from langchain.tools import BaseTool, tool

DEFAULT_MOONSHOT_BASE_URL = "https://api.moonshot.cn/v1"


class FormulaChatClient:
    def __init__(
        self,
        moonshot_base_url: str = DEFAULT_MOONSHOT_BASE_URL,
        api_key: str | None = None,
    ) -> None:
        if api_key is None:
            api_key = os.getenv("MOONSHOT_API_KEY")
        self.httpx = httpx.AsyncClient(
            base_url=moonshot_base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0,
        )

    async def get_tools(self, formula_uri: str):
        response = await self.httpx.get(f"/formulas/{formula_uri}/tools")
        return response.json().get("tools", [])

    async def call_tool(self, formula_uri: str, function: str, args: dict):
        response = await self.httpx.post(
            f"/formulas/{formula_uri}/fibers",
            json={"name": function, "arguments": json.dumps(args)},
        )
        fiber = response.json()

        if fiber.get("status", "") == "succeeded":
            return fiber["context"].get("output") or fiber["context"].get(
                "encrypted_output"
            )

        if "error" in fiber:
            return f"Error: {fiber['error']}"
        if "error" in fiber.get("context", {}):
            return f"Error: {fiber['context']['error']}"
        if "output" in fiber.get("context", {}):
            return f"Error: {fiber['context']['output']}"
        return "Error: Unknown error"


def normalize_formula_uri(uri: str) -> str:
    """Normalize formula URI with default namespace and tag"""
    if "/" not in uri:
        uri = f"moonshot/{uri}"
    if ":" not in uri:
        uri = f"{uri}:latest"
    return uri


async def get_langchain_tool(formula_uri: str) -> type[BaseTool]:
    client = FormulaChatClient()
    formula_uri = normalize_formula_uri(formula_uri)
    tools = await client.get_tools(formula_uri)
    if not tools:
        raise ValueError(f"No tools found for formula {formula_uri}")

    async def _run_tool(**kwargs: Any):
        return await client.call_tool(
            formula_uri, function=tools[0]["function"]["name"], args=kwargs
        )

    return tool(
        tools[0]["function"]["name"],
        description=tools[0]["function"]["description"],
        args_schema=tools[0],
    )(_run_tool)
