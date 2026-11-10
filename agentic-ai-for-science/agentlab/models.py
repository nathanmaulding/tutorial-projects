"""The model layer: the only module that knows about specific models.

Everything else calls `client.create(system=..., messages=..., tools=...)` and gets back a plain
`LLMResponse`. Swapping Haiku for Opus is a config change; adding another provider means adding one
class here with the same `create` method.

Your API key is never read, stored or logged here. The Anthropic SDK reads ANTHROPIC_API_KEY from
the environment by itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

# Dollars per million tokens. Check https://claude.com/pricing if these look stale.
# cache_read defaults to 10% of input and cache_write to 125% of input when not listed.
PRICING: dict[str, dict[str, float]] = {
    "claude-haiku-5-5":  {"input": 0.10, "output": 0.50},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_read": 0.10},
    "claude-opus-5-5":   {"input": 4.00, "output": 20.00, "cache_read": 0.20},
    "claude-fable-5-2":  {"input": 8.00, "output": 40.00, "cache_read": 0.20},
}

# Models that support the server-side refusal fallback (re-runs a declined request on a fallback
# model in the same call). Science prompts can occasionally trip a safety classifier by mistake.
FALLBACK_DEFAULT_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-2"}


def price_of(model: str) -> dict[str, float]:
    if model not in PRICING:
        raise KeyError(f"No price listed for {model!r}; add it to agentlab/models.py PRICING "
                       "so the spending cap stays accurate.")
    p = dict(PRICING[model])
    p.setdefault("cache_read", p["input"] * 0.10)
    p.setdefault("cache_write", p["input"] * 1.25)
    return p


def cost_usd(model: str, usage: dict) -> float:
    p = price_of(model)
    return (usage.get("input_tokens", 0) * p["input"]
            + usage.get("cache_creation_input_tokens", 0) * p["cache_write"]
            + usage.get("cache_read_input_tokens", 0) * p["cache_read"]
            + usage.get("output_tokens", 0) * p["output"]) / 1e6


@dataclass
class LLMResponse:
    content: list[dict]          # content blocks as plain dicts, passed back to the API unchanged
    stop_reason: str
    usage: dict
    model: str
    stop_details: dict | None = None

    @property
    def tool_calls(self) -> list[dict]:
        return [b for b in self.content if b.get("type") == "tool_use"]

    @property
    def text(self) -> str:
        return "\n".join(b["text"] for b in self.content if b.get("type") == "text")


class LLMClient(Protocol):
    model: str
    def create(self, *, system: str, messages: list[dict], tools: list[dict],
               max_tokens: int) -> LLMResponse: ...


def _usage_dict(usage) -> dict:
    keys = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    return {k: int(getattr(usage, k, 0) or 0) for k in keys}


class AnthropicClient:
    """Claude via the Anthropic SDK. Settings that differ by model are handled here."""

    def __init__(self, model: str, effort: str = "medium", thinking_display: str = "summarized"):
        import anthropic  # imported lazily so replay mode works without the SDK
        price_of(model)   # fail early if the spending cap can't price this model
        self.model = model
        self.effort = effort
        self.thinking_display = thinking_display
        self._client = anthropic.Anthropic(max_retries=4)

    def create(self, *, system: str, messages: list[dict], tools: list[dict],
               max_tokens: int) -> LLMResponse:
        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            thinking={"type": "adaptive", "display": self.thinking_display},
            output_config={"effort": self.effort},
            cache_control={"type": "ephemeral"},  # caches the growing prefix: system, tools, history
        )
        if tools:
            kwargs["tools"] = tools
        if self.model in FALLBACK_DEFAULT_MODELS:
            resp = self._client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
        else:
            resp = self._client.messages.create(**kwargs)
        details = getattr(resp, "stop_details", None)
        return LLMResponse(
            content=[b.model_dump(mode="json", exclude_none=True) for b in resp.content],
            stop_reason=resp.stop_reason,
            usage=_usage_dict(resp.usage),
            model=getattr(resp, "model", self.model),
            stop_details=details.model_dump(mode="json") if details is not None else None,
        )


@dataclass
class ScriptedClient:
    """A fake model for tests and demos: each step is a function of the conversation so far.

    It costs nothing, needs no key, and lets the agent loop and tools be tested deterministically.
    """
    steps: list[Callable[[list[dict]], list[dict]]]
    model: str = "scripted"
    calls: int = field(default=0, init=False)

    def create(self, *, system, messages, tools, max_tokens) -> LLMResponse:
        if self.calls >= len(self.steps):
            blocks = [{"type": "text", "text": "(script finished)"}]
        else:
            blocks = self.steps[self.calls](messages)
        self.calls += 1
        stop = "tool_use" if any(b["type"] == "tool_use" for b in blocks) else "end_turn"
        return LLMResponse(content=blocks, stop_reason=stop,
                           usage={"input_tokens": 0, "output_tokens": 0}, model=self.model)


def make_client(model: str, effort: str = "medium", thinking_display: str = "summarized") -> LLMClient:
    if model.startswith("claude-"):
        return AnthropicClient(model, effort=effort, thinking_display=thinking_display)
    raise ValueError(f"Unknown model {model!r}. Add a client class for its provider in agentlab/models.py.")
