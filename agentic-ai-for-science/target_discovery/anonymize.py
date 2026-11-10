"""Gene-name anonymization: does the agent analyse the data, or recall the answer?

Positive controls such as MTAP loss -> PRMT5 are famous; a language model knows them from training.
So "the agent rediscovered 12/12 controls" partly measures memory. This module hides every gene
symbol behind a random code (MTAP -> G04711) at the tool boundary:

  * tool outputs and tool descriptions: every gene symbol is replaced by its code
  * tool inputs: codes are translated back to real symbols; a REAL symbol typed by the model (from
    memory) is blocked, so knowledge can't leak in through queries
  * the literature search and free-form Python tools are removed (they would expose real names)

The comparison that isolates memory is anonymized vs a REDUCED session: same tools removed, real
names kept. Scoring happens on real names after the session.
"""
from __future__ import annotations

import re

import numpy as np

from agentlab.tools import Tool, ToolRegistry

from .stats import load, split

TOKEN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?")
CODE = re.compile(r"^G\d{5}$")
REMOVED_TOOLS = {"search_pubmed", "run_python"}


class Anonymizer:
    def __init__(self, seed: int = 0):
        genes = set(split("discovery").genes)
        genes |= {c.split("_", 1)[1] for c in load("features").columns if not c.startswith("LIN_")}
        genes = sorted(g for g in genes if len(g) >= 2)
        codes = np.random.default_rng(seed).permutation(len(genes))
        self.enc = {g: f"G{c:05d}" for g, c in zip(genes, codes)}
        self.dec = {v: k for k, v in self.enc.items()}

    def encode(self, text: str) -> str:
        return TOKEN.sub(lambda m: self.enc.get(m.group(), m.group()), text)

    def decode(self, text: str) -> str:
        def one(m):
            t = m.group()
            if CODE.match(t):
                return self.dec.get(t, t)
            return "BLOCKED" if t in self.enc else t      # real symbols from memory are not allowed in
        return TOKEN.sub(one, text)

    def decode_obj(self, x):
        if isinstance(x, str):
            return self.decode(x)
        if isinstance(x, list):
            return [self.decode_obj(v) for v in x]
        if isinstance(x, dict):
            return {k: self.decode_obj(v) for k, v in x.items()}
        return x

    def wrap(self, reg: ToolRegistry) -> ToolRegistry:
        import json
        tools = []
        for name in reg.names():
            if name in REMOVED_TOOLS:
                continue
            t = reg._tools[name]

            def fn(_f=t.fn, **kw):
                out = _f(**self.decode_obj(kw))
                text = out if isinstance(out, str) else json.dumps(out, indent=1, default=str)
                return self.encode(text)
            tools.append(Tool(t.name, self.encode(t.description), t.input_schema, fn))
        return ToolRegistry(tools)


def reduced(reg: ToolRegistry) -> ToolRegistry:
    """The same tool set as an anonymized session, with real gene names."""
    return ToolRegistry([reg._tools[n] for n in reg.names() if n not in REMOVED_TOOLS])


def describe(biomarker: str, code: str) -> str:
    """A description of a positive-control biomarker that names no gene and no known biology."""
    kind = biomarker.lstrip("~").split("_", 1)[0]
    neg = biomarker.startswith("~")
    base = {"LOSS": "loss of expression of gene {g}", "MUT": "damaging mutation in gene {g}",
            "HOT": "hotspot mutation in gene {g}"}[kind].format(g=code)
    return f"NO {base}" if neg else base
