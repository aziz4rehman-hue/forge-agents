"""Built-in tools. The calculator only evaluates arithmetic."""

from __future__ import annotations

import ast
import operator

from forge_agents.types import Tool

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_ALLOWED = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.Pow,
    ast.UAdd,
    ast.USub,
    ast.Load,
)


def _eval(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("unsupported expression")
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        if isinstance(node.op, ast.Pow):
            exponent = _eval(node.right)
            if abs(exponent) > 16:
                raise ValueError("exponent too large")
        return _BINOPS[type(node.op)](_eval(node.left), _eval(node.right))
    raise ValueError("unsupported expression")


def _format_number(value: int | float) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def calculate(expression: str) -> str:
    """Evaluate a pure arithmetic expression. Names and calls are rejected."""
    text = expression.strip().replace("^", "**")
    if not text or len(text) > 100:
        raise ValueError("expression missing or too long")
    tree = ast.parse(text, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            raise ValueError("unsupported expression")
    try:
        return _format_number(_eval(tree))
    except ZeroDivisionError as exc:
        raise ValueError("division by zero") from exc


class Memory:
    """Short list of notes shared by a crew."""

    def __init__(self) -> None:
        self.notes: list[str] = []

    def write(self, note: str) -> str:
        text = " ".join(note.split())
        if not text:
            raise ValueError("empty note")
        self.notes.append(text[:500])
        return f"saved note {len(self.notes)}"

    def read(self, query: str = "") -> str:
        if not self.notes:
            return "memory empty"
        if not query.strip():
            return "\n".join(f"- {note}" for note in self.notes[-10:])
        hits = [note for note in self.notes if query.lower() in note.lower()]
        return "\n".join(f"- {note}" for note in hits) if hits else "no match"


def default_tools(memory: Memory) -> list[Tool]:
    return [
        Tool(
            name="calculator",
            description="Evaluate arithmetic such as (12 * 8) + 5 or 2**10.",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Arithmetic expression"},
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
            fn=lambda expression: calculate(str(expression)),
        ),
        Tool(
            name="memory_write",
            description="Save a short fact for later steps.",
            parameters={
                "type": "object",
                "properties": {
                    "note": {"type": "string", "description": "Fact to remember"},
                },
                "required": ["note"],
                "additionalProperties": False,
            },
            fn=lambda note: memory.write(str(note)),
        ),
        Tool(
            name="memory_read",
            description="Read saved notes. Optional query filters by substring.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Optional filter"},
                },
                "required": [],
                "additionalProperties": False,
            },
            fn=lambda query="": memory.read(str(query)),
        ),
    ]
