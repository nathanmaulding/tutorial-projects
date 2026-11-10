"""Tools: Python functions the model can call.

A tool is a name, a description the model reads, a JSON schema for its input, and a function.
The registry produces the `tools=[...]` list for the API and runs calls safely: errors come back to
the model as error results it can react to, never as crashes of the loop.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable

MAX_RESULT_CHARS = 6000  # long outputs waste tokens; the model gets the head and a note


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    fn: Callable[..., Any]

    def spec(self) -> dict:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}


class ToolError(Exception):
    """Raise inside a tool to send a clean error message back to the model."""


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.add(t)

    def add(self, tool: Tool):
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[dict]:
        # Fixed order: the tool list is part of the cached prompt prefix, so it must not change.
        return [self._tools[n].spec() for n in self._tools]

    def execute(self, name: str, tool_input: dict) -> tuple[str, bool, float]:
        """Returns (result_text, is_error, seconds)."""
        t0 = time.time()
        if name not in self._tools:
            return f"Unknown tool {name!r}. Available: {', '.join(self._tools)}", True, 0.0
        try:
            out = self._tools[name].fn(**tool_input)
            text = out if isinstance(out, str) else json.dumps(out, indent=1, default=str)
            is_error = False
        except ToolError as e:
            text, is_error = f"Error: {e}", True
        except TypeError as e:  # wrong or missing arguments
            text, is_error = f"Error: bad arguments for {name}: {e}", True
        except Exception as e:  # noqa: BLE001 - any tool failure goes back to the model
            text, is_error = f"Error: {type(e).__name__}: {e}", True
        if len(text) > MAX_RESULT_CHARS:
            text = text[:MAX_RESULT_CHARS] + f"\n... [truncated {len(text) - MAX_RESULT_CHARS} chars]"
        return text, is_error, time.time() - t0
