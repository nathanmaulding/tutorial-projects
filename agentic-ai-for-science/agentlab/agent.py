"""The agent loop.

An "agent" is a model in a loop: it reads the conversation, either answers or asks to call tools,
we run the tools and append their results, and repeat. Everything interesting (planning, retrieval,
hypothesis testing) is behaviour the model chooses inside this loop, steered by the system prompt
and by which tools exist.

The loop also enforces the guard rails: a step limit, a per-run and total dollar cap, one reminder
if the model stops without delivering, and a full trace of every step.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from .budget import BudgetExceeded, Ledger
from .config import RunConfig
from .tools import ToolRegistry
from .trace import Trace


@dataclass
class AgentResult:
    status: str          # done | no_submission | max_steps | budget | refusal | max_tokens | api_error
    steps: int
    tool_calls: int
    usd: float
    final_text: str


def run_agent(*, client, tools: ToolRegistry, system: str, task_prompt: str, cfg: RunConfig,
              ledger: Ledger, trace: Trace, run_id: str,
              is_done: Callable[[], bool] = lambda: False,
              done_reminder: str = "You stopped without delivering a result. Finish the task now.",
              ) -> AgentResult:
    messages: list[dict] = [{"role": "user", "content": task_prompt}]
    tool_specs = tools.specs()
    run_usd, n_tool_calls, reminded, final_text = 0.0, 0, False, ""
    trace.log("run_start", run_id=run_id, model=client.model, config=cfg.to_dict(),
              system=system, task_prompt=task_prompt, tools=[t["name"] for t in tool_specs])

    def finish(status: str, steps: int) -> AgentResult:
        trace.log("run_end", status=status, steps=steps, tool_calls=n_tool_calls,
                  usd=round(run_usd, 6), final_text=final_text)
        return AgentResult(status, steps, n_tool_calls, run_usd, final_text)

    for step in range(1, cfg.max_steps + 1):
        prompt_chars = len(system) + len(json.dumps(messages, default=str)) + len(json.dumps(tool_specs))
        try:
            ledger.check(client.model, prompt_chars, cfg.max_tokens, run_usd, cfg.per_run_usd_cap)
            resp = client.create(system=system, messages=messages, tools=tool_specs,
                                 max_tokens=cfg.max_tokens)
        except BudgetExceeded as e:
            trace.log("budget_stop", reason=str(e))
            return finish("budget", step - 1)
        except Exception as e:  # noqa: BLE001 - API errors end the run but are recorded
            trace.log("api_error", error=f"{type(e).__name__}: {e}")
            return finish("api_error", step - 1)

        usd = ledger.record(resp.model if resp.model in _priced() else client.model, resp.usage, run_id)
        run_usd += usd
        trace.log("llm_response", step=step, stop_reason=resp.stop_reason, usage=resp.usage,
                  usd=round(usd, 6), content=resp.content, stop_details=resp.stop_details)
        # Append the assistant turn exactly as received (thinking blocks included, unmodified).
        messages.append({"role": "assistant", "content": resp.content})
        if resp.text:
            final_text = resp.text

        if resp.stop_reason == "refusal":
            return finish("refusal", step)
        if resp.stop_reason == "max_tokens":
            return finish("max_tokens", step)

        calls = resp.tool_calls
        if not calls:
            if is_done():
                return finish("done", step)
            if reminded:
                return finish("no_submission", step)
            reminded = True
            messages.append({"role": "user", "content": done_reminder})
            continue

        # Run every requested tool, then return all results together in one user message.
        results = []
        for call in calls:
            out, is_error, secs = tools.execute(call["name"], call.get("input", {}))
            n_tool_calls += 1
            trace.log("tool_result", step=step, name=call["name"], input=call.get("input", {}),
                      output=out, is_error=is_error, seconds=round(secs, 2))
            results.append({"type": "tool_result", "tool_use_id": call["id"], "content": out,
                            **({"is_error": True} if is_error else {})})
        if is_done():
            return finish("done", step)
        left = cfg.max_steps - step
        if left <= 3:
            results.append({"type": "text", "text": f"[{left} model steps left. Deliver your result soon.]"})
        messages.append({"role": "user", "content": results})

    return finish("max_steps", cfg.max_steps)


def _priced():
    from .models import PRICING
    return PRICING
