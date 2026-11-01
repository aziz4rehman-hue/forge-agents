"""Local-first multi-agent runtime. Runs with no API key."""

__version__ = "0.1.0"

from forge_agents.agent import Agent
from forge_agents.crew import Crew
from forge_agents.llm import HeuristicLLM, OpenAICompatibleLLM, ScriptedLLM

__all__ = [
    "Agent",
    "Crew",
    "HeuristicLLM",
    "OpenAICompatibleLLM",
    "ScriptedLLM",
    "__version__",
]
