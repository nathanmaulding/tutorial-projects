"""agentlab: a small, readable framework for building and evaluating scientific agents.

Pieces (each is one short module, written to be read):
    models    - talks to the model; the one place that knows model names, prices and request settings
    budget    - a spending ledger with a hard dollar cap
    tools     - a registry that turns Python functions into tools the model can call
    sandbox   - runs model-written Python in a separate process with no secrets in its environment
    retrieval - a tiny BM25 search engine over markdown documents
    trace     - records every model call and tool call to a replayable JSONL file (with key redaction)
    agent     - the agent loop itself
"""
from .config import RunConfig
from .models import make_client, PRICING
from .budget import Ledger, BudgetExceeded
from .tools import Tool, ToolRegistry
from .trace import Trace, load_trace
from .agent import run_agent, AgentResult

__all__ = [
    "RunConfig", "make_client", "PRICING", "Ledger", "BudgetExceeded",
    "Tool", "ToolRegistry", "Trace", "load_trace", "run_agent", "AgentResult",
]
