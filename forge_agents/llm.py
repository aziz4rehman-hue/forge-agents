"""LLM providers. HeuristicLLM needs no network and no API key."""

from __future__ import annotations

import ast
import json
import os
import re
import urllib.error
import urllib.request
from typing import Protocol

from forge_agents.types import LLMResponse, Message, Tool, ToolCall

_EXPR = re.compile(
    r"(?:\d+(?:\.\d+)?|\([^()]+\))"
    r"(?:\s*(?:\*\*|[\+\-\*/%])\s*(?:\d+(?:\.\d+)?|\([^()]+\)))+"
)


def find_expression(text: str) -> str | None:
    """Return the longest arithmetic expression in text, if any."""
    best: str | None = None
    for match in _EXPR.findall(text.replace("^", "**")):
        candidate = match.strip()
        try:
            tree = ast.parse(candidate, mode="eval")
        except SyntaxError:
            continue
        if not any(isinstance(node, ast.BinOp) for node in ast.walk(tree)):
            continue
        if best is None or len(candidate) > len(best):
            best = candidate
    return best


class LLM(Protocol):
    def complete(self, messages: list[Message], tools: list[Tool] | None = None) -> LLMResponse:
        """Return the next assistant turn."""


class ScriptedLLM:
    """Replay fixed responses. Used by tests and demos."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def complete(self, messages: list[Message], tools: list[Tool] | None = None) -> LLMResponse:
        if self.calls >= len(self._responses):
            raise RuntimeError("ScriptedLLM ran out of responses")
        response = self._responses[self.calls]
        self.calls += 1
        return response


class HeuristicLLM:
    """Rule-based provider so the crew runs offline.

    It is not a model. It detects arithmetic, memory requests, and the
    planner/critic roles from the system prompt.
    """

    def complete(self, messages: list[Message], tools: list[Tool] | None = None) -> LLMResponse:
        system = next((m.content for m in messages if m.role == "system"), "").lower()
        visible = "\n".join(m.content for m in messages if m.role in {"user", "tool"})
        names = {tool.name for tool in tools or []}

        if "planner" in system:
            return LLMResponse(content=json.dumps({"steps": _plan(visible)}))
        if "critic" in system:
            return LLMResponse(
                content="Final answer based on the crew trace:\n" + visible[-1500:].strip()
            )

        last_tool = next((m for m in reversed(messages) if m.role == "tool"), None)
        if last_tool is not None:
            return LLMResponse(content=f"{last_tool.name} -> {last_tool.content}")

        expression = find_expression(visible)
        if expression and "calculator" in names:
            return LLMResponse(
                tool_calls=[
                    ToolCall(id="call_calc", name="calculator", arguments={"expression": expression})
                ],
                finish_reason="tool_calls",
            )

        remembered = re.search(r"remember(?:\s+that)?\s+(.+)", visible, re.IGNORECASE)
        if remembered and "memory_write" in names:
            note = remembered.group(1).strip().split("\n")[0][:500]
            return LLMResponse(
                tool_calls=[ToolCall(id="call_mem", name="memory_write", arguments={"note": note})],
                finish_reason="tool_calls",
            )

        if "recall" in visible.lower() and "memory_read" in names:
            return LLMResponse(
                tool_calls=[ToolCall(id="call_read", name="memory_read", arguments={"query": ""})],
                finish_reason="tool_calls",
            )

        return LLMResponse(content=f"No tool needed. Noted: {visible.strip()[:400]}")


def _plan(task: str) -> list[str]:
    steps: list[str] = []
    expression = find_expression(task)
    if expression:
        steps.append(f"Calculate {expression}")
    remembered = re.search(r"remember(?:\s+that)?\s+(.+)", task, re.IGNORECASE)
    if remembered:
        fact = remembered.group(1).strip().split("\n")[0]
        steps.append(f"remember {fact}")
    if "recall" in task.lower():
        steps.append("recall saved notes")
    if not steps:
        steps.append("Answer the task directly in one short paragraph")
    return steps


class OpenAICompatibleLLM:
    """Optional adapter for any OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        timeout: float = 60,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        self.timeout = timeout

    def complete(self, messages: list[Message], tools: list[Tool] | None = None) -> LLMResponse:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        payload: dict = {
            "model": self.model,
            "messages": [message.to_api() for message in messages],
            "temperature": 0,
        }
        if tools:
            payload["tools"] = [
                {"type": "function", "function": tool.spec()} for tool in tools
            ]
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            raise RuntimeError(f"LLM request failed ({exc.code}): {detail}") from exc

        message = body["choices"][0]["message"]
        calls: list[ToolCall] = []
        for raw in message.get("tool_calls") or []:
            arguments = raw.get("function", {}).get("arguments") or "{}"
            calls.append(
                ToolCall(
                    id=raw.get("id") or "call",
                    name=raw["function"]["name"],
                    arguments=json.loads(arguments),
                )
            )
        return LLMResponse(
            content=message.get("content"),
            tool_calls=calls,
            finish_reason=body["choices"][0].get("finish_reason") or "stop",
        )
