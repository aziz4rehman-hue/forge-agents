"""Command line entry. Default provider is offline."""

from __future__ import annotations

import argparse
import json

from forge_agents import __version__
from forge_agents.agent import Agent
from forge_agents.crew import WORKER_ROLE, Crew
from forge_agents.llm import HeuristicLLM, OpenAICompatibleLLM
from forge_agents.tools import Memory, default_tools


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forge-agents",
        description="Run a local agent or a planner/worker/critic crew.",
    )
    parser.add_argument("task", nargs="?", default="What is (12 * 8) + 5?")
    parser.add_argument("--crew", action="store_true", help="Run planner, worker, and critic")
    parser.add_argument("--provider", choices=["heuristic", "openai"], default="heuristic")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--version", action="version", version=f"forge-agents {__version__}")
    return parser


def make_llm(provider: str, model: str, base_url: str):
    if provider == "openai":
        return OpenAICompatibleLLM(model=model, base_url=base_url)
    return HeuristicLLM()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    llm = make_llm(args.provider, args.model, args.base_url)
    if args.crew:
        result = Crew(llm).run(args.task)
        print(json.dumps(result, indent=2))
        return 0
    agent = Agent("worker", WORKER_ROLE, llm, tools=default_tools(Memory()))
    print(agent.run(args.task))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
