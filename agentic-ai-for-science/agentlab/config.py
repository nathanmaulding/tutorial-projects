"""Run settings. The model is a setting, never hard-coded elsewhere.

Precedence: explicit argument > environment variable > default.
    AGENTLAB_MODEL      e.g. claude-haiku-5-5, claude-sonnet-5-5, claude-opus-5-5
    AGENTLAB_EFFORT     low | medium | high | xhigh | max
    AGENTLAB_TOTAL_CAP  total dollars this checkout may spend, across all runs
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict

DEFAULT_MODEL = "claude-haiku-5-5"


@dataclass
class RunConfig:
    model: str = field(default_factory=lambda: os.environ.get("AGENTLAB_MODEL", DEFAULT_MODEL))
    effort: str = field(default_factory=lambda: os.environ.get("AGENTLAB_EFFORT", "medium"))
    thinking_display: str = "summarized"   # readable reasoning summaries in traces; billed the same
    max_tokens: int = 16000                # per model response
    max_steps: int = 20                    # model calls per agent run
    per_run_usd_cap: float = 0.75          # one run can never cost more than this
    total_usd_cap: float = field(
        default_factory=lambda: float(os.environ.get("AGENTLAB_TOTAL_CAP", "10.0")))

    def to_dict(self) -> dict:
        return asdict(self)
