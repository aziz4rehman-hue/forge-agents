"""Planner, worker, critic. One shared memory, one shared provider."""

from __future__ import annotations

import json

from forge_agents.agent import Agent
from forge_agents.tools import Memory, default_tools

PLANNER_ROLE = (
    "You are the planner in a small agent crew. "
    'Return only JSON of the form {"steps": ["..."]}. Do not call tools.'
)
WORKER_ROLE = (
    "You are the worker in a small agent crew. "
    "Use a tool when the step needs arithmetic or memory. Then report the result."
)
CRITIC_ROLE = (
    "You are the critic in a small agent crew. "
    "Review the worker results and write the final answer."
)


def parse_steps(raw: str, fallback: list[str]) -> list[str]:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return fallback
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return fallback
    steps = data.get("steps")
    if not isinstance(steps, list):
        return fallback
    cleaned = [str(step).strip() for step in steps if str(step).strip()]
    return cleaned or fallback


class Crew:
    def __init__(self, llm, memory: Memory | None = None, max_steps: int = 6) -> None:
        self.llm = llm
        self.memory = memory or Memory()
        tools = default_tools(self.memory)
        self.planner = Agent("planner", PLANNER_ROLE, llm, tools=[], max_steps=max_steps)
        self.worker = Agent("worker", WORKER_ROLE, llm, tools=tools, max_steps=max_steps)
        self.critic = Agent("critic", CRITIC_ROLE, llm, tools=[], max_steps=max_steps)

    def run(self, task: str) -> dict:
        plan_raw = self.planner.run(
            "Break this task into steps.\nTask: " + task.strip()
        )
        steps = parse_steps(plan_raw, fallback=[task.strip()])
        traces: list[dict[str, str]] = []
        for step in steps:
            traces.append({"step": step, "result": self.worker.run(step)})
        answer = self.critic.run(
            "Original task:\n"
            + task.strip()
            + "\n\nWorker trace:\n"
            + json.dumps(traces, indent=2)
        )
        return {"plan": steps, "traces": traces, "answer": answer, "notes": list(self.memory.notes)}
