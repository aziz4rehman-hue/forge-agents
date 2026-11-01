# forge-agents

Local-first multi-agent runtime in Python. A planner splits the task, a worker calls tools, and a critic writes the final answer. The default provider is a deterministic heuristic, so the whole crew runs with **no API key and no network**.

```
python -m forge_agents "What is (12 * 8) + 5?"
python -m forge_agents --crew "What is 2**10? Remember that transformers use attention."
```

## What you get

- A bounded agent loop (`think -> tool -> observe -> answer`)
- Three tools: safe arithmetic, memory write, memory read
- A crew: planner, worker, critic, one shared memory
- `ScriptedLLM` for tests and `OpenAICompatibleLLM` if you want a real model later
- Zero third-party dependencies

The calculator parses the expression with Python's `ast` module and only allows numbers and `+ - * / % **`. It will not run names, calls, or imports.

## Quick start

```bash
git clone https://github.com/aziz4rehman-hue/forge-agents.git
cd forge-agents
python -m forge_agents "What is (12 * 8) + 5?"
```

```
calculator -> 101
```

Crew mode prints the plan, each worker result, saved notes, and the critic's answer:

```bash
python -m forge_agents --crew "What is 2**10? Remember that transformers use attention."
```

```json
{
  "plan": [
    "Calculate 2**10",
    "remember transformers use attention."
  ],
  "traces": [
    { "step": "Calculate 2**10", "result": "calculator -> 1024" },
    { "step": "remember transformers use attention.", "result": "memory_write -> saved note 1" }
  ],
  "notes": ["transformers use attention."]
}
```

From Python:

```python
from forge_agents import Crew, HeuristicLLM

result = Crew(HeuristicLLM()).run("What is (12 * 8) + 5?")
print(result["answer"])
```

See [examples/crew_demo.py](examples/crew_demo.py).

## Use a real model

Any OpenAI-compatible chat endpoint works. This is optional.

```bash
export OPENAI_API_KEY=sk-...
python -m forge_agents --provider openai --model gpt-4o-mini "Draft a 3-step plan for testing a Python CLI"
```

Custom base URL (local server, proxy, other vendor):

```bash
python -m forge_agents --provider openai --base-url http://127.0.0.1:11434/v1 --model llama3.2 "Summarize tool calling"
```

## How a run works

```
task
  │
  ▼
planner ── JSON steps
  │
  ▼
worker ── calculator / memory_write / memory_read
  │         (max 6 steps, then it stops)
  ▼
critic ── final answer
```

`HeuristicLLM` is not a language model. It exists so the architecture is real and testable before you pay for tokens. Swap in `OpenAICompatibleLLM` (or your own object with a `complete(messages, tools)` method) without changing the agents.

## Tests

```bash
python -m unittest discover -s tests -v
```

Requires Python 3.10+.

## License

MIT. Built by [Aziz Rehman](https://github.com/aziz4rehman-hue).
