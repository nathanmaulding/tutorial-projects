"""A spending ledger with a hard cap.

Every real model call is appended to a JSONL file with its token counts and cost. Before each call
the ledger checks that the total so far, plus a worst-case estimate for the next call, stays under
the cap. If not, it raises BudgetExceeded and nothing is sent.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from .models import cost_usd, price_of


class BudgetExceeded(RuntimeError):
    pass


class Ledger:
    def __init__(self, path: str | Path, total_cap_usd: float):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.total_cap = total_cap_usd
        self._lock = threading.Lock()

    def spent(self) -> float:
        if not self.path.exists():
            return 0.0
        with self.path.open() as f:
            return sum(json.loads(line)["usd"] for line in f if line.strip())

    @staticmethod
    def worst_case(model: str, prompt_chars: int, max_tokens: int) -> float:
        """Upper bound for one call: every prompt token uncached (~3 chars/token), full max_tokens out."""
        p = price_of(model)
        return (prompt_chars / 3 * p["input"] + max_tokens * p["output"]) / 1e6

    def check(self, model: str, prompt_chars: int, max_tokens: int, run_spent: float, run_cap: float):
        if model == "scripted":
            return
        nxt = self.worst_case(model, prompt_chars, max_tokens)
        with self._lock:
            total = self.spent()
        if total + nxt > self.total_cap:
            raise BudgetExceeded(f"Total cap ${self.total_cap:.2f} would be exceeded "
                                 f"(spent ${total:.4f}, next call up to ${nxt:.4f}).")
        if run_spent + nxt > run_cap:
            raise BudgetExceeded(f"Per-run cap ${run_cap:.2f} would be exceeded "
                                 f"(this run ${run_spent:.4f}, next call up to ${nxt:.4f}).")

    def record(self, model: str, usage: dict, run_id: str) -> float:
        if model == "scripted":
            return 0.0
        usd = cost_usd(model, usage)
        with self._lock, self.path.open("a") as f:
            f.write(json.dumps({"ts": round(time.time(), 1), "run_id": run_id, "model": model,
                                "usd": round(usd, 6), **usage}) + "\n")
        return usd
