import unittest

from forge_agents.agent import Agent
from forge_agents.crew import Crew, parse_steps
from forge_agents.llm import HeuristicLLM, ScriptedLLM, find_expression
from forge_agents.tools import Memory, calculate, default_tools
from forge_agents.types import LLMResponse, ToolCall


class CalculatorTests(unittest.TestCase):
    def test_arithmetic(self):
        self.assertEqual(calculate("(12 * 8) + 5"), "101")
        self.assertEqual(calculate("2**10"), "1024")
        self.assertEqual(calculate("10 / 4"), "2.5")

    def test_rejects_unsafe_input(self):
        for expression in ["__import__('os')", "a + 1", "abs(-1)", "1/0", "2**40"]:
            with self.assertRaises(ValueError):
                calculate(expression)

    def test_find_expression_picks_longest(self):
        self.assertEqual(find_expression("What is (12 * 8) + 5?"), "(12 * 8) + 5")
        self.assertEqual(find_expression("2**10"), "2**10")
        self.assertIsNone(find_expression("no math here"))


class AgentTests(unittest.TestCase):
    def test_scripted_tool_loop(self):
        llm = ScriptedLLM(
            [
                LLMResponse(
                    tool_calls=[ToolCall("1", "calculator", {"expression": "2+2"})],
                    finish_reason="tool_calls",
                ),
                LLMResponse(content="The answer is 4"),
            ]
        )
        agent = Agent("math", "worker", llm, tools=default_tools(Memory()))
        self.assertEqual(agent.run("add"), "The answer is 4")

    def test_unknown_tool_does_not_crash(self):
        llm = ScriptedLLM(
            [
                LLMResponse(tool_calls=[ToolCall("1", "nope", {})], finish_reason="tool_calls"),
                LLMResponse(content="recovered"),
            ]
        )
        agent = Agent("math", "worker", llm, tools=default_tools(Memory()))
        self.assertEqual(agent.run("go"), "recovered")

    def test_max_steps(self):
        llm = ScriptedLLM(
            [
                LLMResponse(tool_calls=[ToolCall("1", "calculator", {"expression": "1+1"})]),
                LLMResponse(tool_calls=[ToolCall("2", "calculator", {"expression": "1+1"})]),
            ]
        )
        agent = Agent("math", "worker", llm, tools=default_tools(Memory()), max_steps=2)
        self.assertIn("max steps", agent.run("loop"))

    def test_heuristic_solves_math(self):
        agent = Agent("worker", "worker", HeuristicLLM(), tools=default_tools(Memory()))
        self.assertIn("101", agent.run("What is (12 * 8) + 5?"))


class CrewTests(unittest.TestCase):
    def test_parse_steps_fallback(self):
        self.assertEqual(parse_steps("not json", ["fallback"]), ["fallback"])
        self.assertEqual(parse_steps('{"steps": ["a", "b"]}', ["x"]), ["a", "b"])

    def test_heuristic_crew(self):
        crew = Crew(HeuristicLLM())
        result = crew.run("What is 2**10? Remember that transformers use attention.")
        self.assertIn("1024", result["answer"])
        self.assertTrue(any("attention" in note for note in result["notes"]))
        self.assertGreaterEqual(len(result["plan"]), 2)


if __name__ == "__main__":
    unittest.main()
