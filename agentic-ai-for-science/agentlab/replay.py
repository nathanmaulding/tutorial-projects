"""Show a recorded agent run step by step, without calling any model."""
from __future__ import annotations

import json
import textwrap

from .trace import load_trace


def _short(text: str, n: int) -> str:
    text = str(text)
    return text if len(text) <= n else text[: n - 20] + f" ... [{len(text) - n + 20} more chars]"


def render(path, max_output=700, max_code=900, show_thinking=True) -> str:
    """Return a readable, markdown-formatted transcript of a trace file."""
    lines = []
    for e in load_trace(path):
        k = e["kind"]
        if k == "run_start":
            lines.append(f"**Run** `{e['run_id']}`  \n**Model:** `{e['model']}`  \n"
                         f"**Tools available:** {', '.join(e['tools']) or 'none'}\n")
            lines.append("**Task given to the agent:**\n\n> " + e["task_prompt"].replace("\n", "\n> ") + "\n")
        elif k == "llm_response":
            u = e["usage"]
            lines.append(f"---\n### Step {e['step']}  \n<sub>{u.get('input_tokens', 0) + u.get('cache_read_input_tokens', 0) + u.get('cache_creation_input_tokens', 0):,} input tokens "
                         f"({u.get('cache_read_input_tokens', 0):,} from cache), {u.get('output_tokens', 0):,} output, ${e['usd']:.4f}</sub>\n")
            for b in e["content"]:
                if b["type"] == "thinking" and show_thinking and b.get("thinking"):
                    lines.append("*Reasoning (summary):* " + _short(b["thinking"], 600).replace("\n", " ") + "\n")
                elif b["type"] == "text" and b["text"].strip():
                    lines.append("**Says:** " + b["text"].strip() + "\n")
                elif b["type"] == "tool_use":
                    inp = dict(b["input"])
                    code = inp.pop("custom_code", None) or inp.pop("code", None)
                    lines.append(f"**Calls `{b['name']}`** `{_short(json.dumps(inp), 400)}`\n")
                    if code:
                        lines.append("```python\n" + _short(code, max_code) + "\n```\n")
        elif k == "tool_result":
            tag = "Error from" if e["is_error"] else "Result of"
            lines.append(f"<details><summary>{tag} <code>{e['name']}</code></summary>\n\n```\n"
                         + _short(e["output"], max_output) + "\n```\n</details>\n")
        elif k in ("budget_stop", "api_error", "predict_error"):
            lines.append(f"**{k}:** {e.get('reason') or e.get('error')}\n")
        elif k == "run_end":
            lines.append(f"---\n**Run ended:** {e['status']} after {e.get('steps', '?')} steps, "
                         f"{e.get('tool_calls', 0)} tool calls, ${e.get('usd', 0):.4f}\n")
        elif k == "scored":
            r = e["result"]
            lines.append(f"**Scored against the hidden truth:** submitted `{r.get('label')}`; "
                         f"mechanism correct: {r.get('mechanism_ok')}; held-out error {r.get('heldout_nrmse')} "
                         f"(threshold {r.get('threshold')}); **success: {r.get('success')}**\n")
    return "\n".join(lines)


def show(path, **kw):
    """Display a trace in a notebook."""
    from IPython.display import Markdown, display
    display(Markdown(render(path, **kw)))


def summary_line(path) -> str:
    ev = load_trace(path)
    end = next((e for e in ev if e["kind"] == "run_end"), {})
    return textwrap.shorten(f"{end.get('status')} in {end.get('steps')} steps, ${end.get('usd', 0):.4f}", 80)
