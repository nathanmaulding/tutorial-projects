"""A small, safe wrapper around Claude for in-context learning experiments.

* Every response is cached in runs/part1/cache.jsonl, keyed by a hash of the request. Re-running an
  experiment replays from the cache, so the notebook works with no API key and costs nothing.
* A spending ledger (runs/part1/spend.jsonl) refuses any call that could take total spend above the cap
  (ICL_TOTAL_CAP, default $5).
* The API key is only ever read by the Anthropic SDK from your environment; it is never written anywhere.
* No extended thinking: we want to measure what the model infers from the examples in one pass, not what it
  can work out by reasoning at length.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs" / "part1"
CACHE = RUNS / "cache.jsonl"
LEDGER = RUNS / "spend.jsonl"
TOTAL_CAP = float(os.environ.get("ICL_TOTAL_CAP", "5"))

# Dollars per million tokens (check https://claude.com/pricing if these look stale)
PRICING = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.00, 10.00), "claude-opus-5-5": (4.00, 20.00)}

# How to switch thinking off for each model (checked against the API on 2026-11-14)
NO_THINKING = {"claude-haiku-5-5": {"type": "disabled"}, "claude-sonnet-5-5": {"type": "between_tools"}}

_lock = threading.Lock()
_cache: dict | None = None


class BudgetExceeded(RuntimeError):
    pass


def _key(model: str, system: str, prompt: str, max_tokens: int, sample: int, json_number: bool = False) -> str:
    parts = [model, system, prompt, max_tokens, sample] + (["json_number"] if json_number else [])
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()


def _load_cache() -> dict:
    global _cache
    if _cache is None:
        _cache = {}
        if CACHE.exists():
            for line in CACHE.open():
                if line.strip():
                    r = json.loads(line)
                    _cache[r["key"]] = r
    return _cache


def spent() -> float:
    return sum(json.loads(l)["usd"] for l in LEDGER.open() if l.strip()) if LEDGER.exists() else 0.0


NUMBER_FORMAT = {"type": "json_schema", "schema": {"type": "object", "properties": {"y": {"type": "number"}},
                                                    "required": ["y"], "additionalProperties": False}}


def ask(prompt: str, model: str = "claude-haiku-5-5", system: str = "", max_tokens: int = 20, sample: int = 0,
        live: bool = True, json_number: bool = False) -> str:
    """One question, one short answer. `sample` lets you draw several independent answers to one prompt.
    json_number=True constrains the reply to {"y": <number>} (structured output), so the model can't reason
    out loud first: we get its one-pass guess."""
    key = _key(model, system, prompt, max_tokens, sample, json_number)
    with _lock:
        hit = _load_cache().get(key)
    if hit is not None:
        return hit["text"]
    if not live or not os.environ.get("ANTHROPIC_API_KEY"):
        raise LookupError("not in the replay cache, and live calls are off (set ANTHROPIC_API_KEY to run live)")
    p_in, p_out = PRICING[model]
    worst = (len(system) + len(prompt)) / 3 * p_in / 1e6 + max_tokens * p_out / 1e6
    with _lock:
        if spent() + worst > TOTAL_CAP:
            raise BudgetExceeded(f"spent ${spent():.3f}; this call could exceed the ${TOTAL_CAP:.2f} cap")
    import anthropic
    client = anthropic.Anthropic(max_retries=6)
    # Thinking off: Claude 5 models think by default, and we want the one-pass in-context answer.
    # The setting differs by model; Opus 5.5 can't turn thinking off, so it isn't used here.
    if model not in NO_THINKING:
        raise ValueError(f"{model} can't answer without thinking; use one of {list(NO_THINKING)}")
    kwargs = dict(model=model, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}],
                  thinking=NO_THINKING[model])
    if system:
        kwargs["system"] = system
    if json_number:
        kwargs["output_config"] = {"format": NUMBER_FORMAT}
    resp = client.messages.create(**kwargs)
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    usd = (resp.usage.input_tokens * p_in + resp.usage.output_tokens * p_out) / 1e6
    with _lock:
        RUNS.mkdir(parents=True, exist_ok=True)
        rec = {"key": key, "model": model, "text": text, "stop_reason": resp.stop_reason,
               "in": resp.usage.input_tokens, "out": resp.usage.output_tokens}
        with CACHE.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        with LEDGER.open("a") as f:
            f.write(json.dumps({"ts": time.time(), "model": model, "usd": usd}) + "\n")
        _load_cache()[key] = rec
    return text
