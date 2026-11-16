"""Makes the tutorial folder importable (icl_llm, icl_bio) when pytest runs from the repo root."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
