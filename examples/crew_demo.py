"""Run the offline crew from Python. No API key."""

from forge_agents import Crew, HeuristicLLM

result = Crew(HeuristicLLM()).run(
    "What is (12 * 8) + 5? Remember that agents need tools to touch the world."
)
print("plan:", result["plan"])
print("notes:", result["notes"])
print(result["answer"])
