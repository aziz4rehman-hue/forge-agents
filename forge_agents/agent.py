"""One agent: system role, tools, and a bounded tool loop."""

from __future__ import annotations

from forge_agents.types import LLMResponse, Message, Tool, ToolCall


class Agent:
    def __init__(
        self,
        name: str,
        role: str,
        llm,
        tools: list[Tool] | None = None,
        max_steps: int = 6,
    ) -> None:
        self.name = name
        self.role = role
        self.llm = llm
        self.tools = list(tools or [])
        self.max_steps = max_steps
        self._tools = {tool.name: tool for tool in self.tools}

    def run(self, task: str) -> str:
        messages = [
            Message(role="system", content=self.role),
            Message(role="user", content=task),
        ]
        for _ in range(self.max_steps):
            response: LLMResponse = self.llm.complete(messages, tools=self.tools)
            if response.tool_calls:
                messages.append(
                    Message(
                        role="assistant",
                        content=response.content or "",
                        tool_calls=response.tool_calls,
                    )
                )
                for call in response.tool_calls:
                    messages.append(
                        Message(
                            role="tool",
                            content=self._invoke(call),
                            name=call.name,
                            tool_call_id=call.id,
                        )
                    )
                continue
            text = (response.content or "").strip()
            return text or "(empty response)"
        return "stopped: reached max steps before a final answer"

    def _invoke(self, call: ToolCall) -> str:
        tool = self._tools.get(call.name)
        if tool is None:
            return f"unknown tool: {call.name}"
        try:
            return tool(**call.arguments)
        except Exception as exc:  # tool boundary: surface the error to the model
            return f"error: {exc}"
