# Agentic AI for Science

Building, benchmarking and stress-testing AI agents that do scientific modelling: tool use, retrieval, code execution, planning, iterative hypothesis testing, agent evaluation and scientific reliability, in one small framework you can read end to end.

| Part | Notebook | What it covers | Run it |
|---|---|---|---|
| **1a** | [Building a scientific agent](01_building_a_science_agent.ipynb) | an agent that discovers mechanistic models (enzyme kinetics, pharmacokinetics, drug effects) from data; every component explained and shown working | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nathanmaulding/tutorial-projects/blob/main/agentic-ai-for-science/01_building_a_science_agent.ipynb) |
| **1b** | [Benchmarking and reliability](02_benchmarking_and_reliability.ipynb) | 18 tasks with known answers, baselines and ablations, failure analysis, consistency, calibration, cost | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nathanmaulding/tutorial-projects/blob/main/agentic-ai-for-science/02_benchmarking_and_reliability.ipynb) |
| **2** | [Agentic target discovery](03_agentic_target_discovery.ipynb) | an agent hunting for selective cancer dependencies (candidate drug targets) in DepMap CRISPR screens, with pre-registration, replication in held-out lines and an independent lab's screens, literature novelty checks and null-world controls | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nathanmaulding/tutorial-projects/blob/main/agentic-ai-for-science/03_agentic_target_discovery.ipynb) |

## What we found

**Part 1 (18 tasks × 10 seeds, 1,458 recorded runs):**
- The Haiku agent solves 96% of runs for $0.003 each; the Opus agent solves 35 of 36.
- **The agent loop stands in for model capability:** Haiku with one call plus a fitting step solves 10% of new-mechanism datasets, and the Haiku agent solves 92%. Opus with one call plus fitting already solves them all.
- Retrieval helped on new mechanisms only where a card pointed at the answer.

**Part 2 (DepMap):**
- Zero false discoveries in null worlds.
- Positive controls rediscovered even with every gene name hidden.
- Candidates tested prospectively (cell lines screened after the data release, new CRISPR system) and under RNAi, all pre-registered.
- The best-supported candidates (DNAJC15 → DNAJC19, TTC7B → TTC7A) hold under every test but turn out to be **published**, as does the original lead (ATP6V0E2 → ATP6V0E1, found by a full-text search after an abstract-only search missed it). NABP1 → NABP2 and AGPAT2 → AGPAT1 have no paper but were computationally predicted in 2019-2020 supplementary tables. This project adds independent held-out, prospective and RNAi evidence for them, not a discovery.

The notebooks **replay recorded agent runs by default, so they need no API key and cost nothing.** Each one also has a live mode that uses *your own* key.

## Layout

```
agentlab/                 the framework (about 650 lines, written to be read)
  agent.py                the agent loop and its guard rails
  tools.py                Python functions -> tools the model can call
  retrieval.py            BM25 search over markdown documents
  sandbox.py              runs model-written code in a keyless subprocess
  models.py               model names, prices, request settings: the only model-specific code
  budget.py, trace.py     hard dollar caps; replayable JSONL traces with key redaction
  replay.py               renders a trace in a notebook
mechanism_discovery/      Part 1
  library.py              candidate mechanisms (rate laws, PK and PD models)
  cards/                  the retrieval corpus: one card per mechanism or method, with references
  tasks.py                18 benchmark tasks with hidden ground truth and held-out tests
  fitting.py              nonlinear fitting with uncertainty, identifiability and residual diagnostics
  agent_tools.py          the agent's tools, per arm (ablations remove tools honestly)
  run.py                  baselines, the agent arms, and the benchmark runner
  analysis.py             the tables in notebook 02
target_discovery/         Part 2
  prep.py                 builds the discovery / held-out / Sanger split from DepMap 24Q4
  stats.py                lineage-adjusted association tests; null worlds
  agent_tools.py          exploration tools + pre-registration and a fixed replication budget
  pubmed.py               literature retrieval (NCBI E-utilities, cached)
  run.py                  positive controls, discovery missions, null worlds, baseline pipeline
runs/                     recorded traces, results and the spend ledger
tests/                    offline tests (no key, no cost)
```

## Running it yourself

```bash
cd agentic-ai-for-science
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                                      # offline tests, free
export ANTHROPIC_API_KEY=...                # your own key, only for live runs
python -m mechanism_discovery.run --arms agent_full --tasks hard --model claude-haiku-5-5
```

Part 2 also needs the DepMap 24Q4 data (about 1.6 GB, not in the repo); notebook 03, section 9 lists the files and commands.

**Choosing a model.** The model is a setting, never hard-coded: pass `--model`, or set `AGENTLAB_MODEL` (default `claude-haiku-5-5`). Prices for the spending cap live in `agentlab/models.py`; another provider can be added there as one class with the same `create` method.

**Spending caps.** Every call is checked before it is sent against a per-run cap and a total cap across runs (`AGENTLAB_TOTAL_CAP`, default $10), using the spend recorded in `runs/spend_ledger.jsonl`. A Haiku agent run costs about $0.003.

## Your API key stays private

- The key is only ever read from your environment by the Anthropic SDK. It is never written into code, notebooks, configs or traces.
- Model-written code runs in a subprocess whose environment contains **no keys**.
- Every trace line passes through a redaction filter before it is written.
- A secret scan (`scripts/check_secrets.py`) runs as a pre-commit hook (`git config core.hooksPath .githooks`) and on every push in GitHub Actions.
- Public notebooks run in replay mode. Anyone running live mode uses their own key.

Recommended: create a separate API key for this project in the Anthropic Console with a low monthly spending limit.
