# In-Context Learning

How a model can learn a new task from the examples in its prompt, without changing a single weight: first in a transformer small enough to train on a laptop, then in Claude, then in a tabular foundation model pointed at real cancer data.

| Part | Notebook | What it covers | Run it |
|---|---|---|---|
| **1a** | [Inside a transformer](01_icl_inside_a_transformer.ipynb) | train a 150k-parameter transformer to do linear regression in context; compare it with least squares and gradient descent; test that it reads the examples; watch it fail out of distribution | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nathanmaulding/tutorial-projects/blob/main/in-context-learning/01_icl_inside_a_transformer.ipynb) |
| **1b** | [In Claude](02_icl_in_claude.ipynb) | regression from examples, flipped and meaningless labels (examples vs prior knowledge), majority-label bias; how to measure ICL fairly | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nathanmaulding/tutorial-projects/blob/main/in-context-learning/02_icl_in_claude.ipynb) |
| **2** | [For biology: TabPFN on DepMap](03_icl_for_biology_tabpfn.ipynb) | a pre-registered test of whether an in-context learner predicts cancer gene dependencies better than standard methods when labelled cell lines are scarce, with a prospective test on newly screened lines | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nathanmaulding/tutorial-projects/blob/main/in-context-learning/03_icl_for_biology_tabpfn.ipynb) |

Every notebook runs in seconds from saved results, with **no API key, no GPU and no data download**.

## What we found

- **1a:** with frozen weights, the tiny transformer's error falls as the prompt holds more examples, tracking least squares closely and beating one step of gradient descent at every prompt length. Scrambling the (x, y) pairs destroys it, and inputs 3x larger than in training break it.
- **1b:** Claude learns a new line from examples but stays 2–4x less accurate than least squares. Sonnet follows flipped labels every time while Haiku reverts to the real meaning 21% of the time. Haiku drifts toward the majority label on neutral inputs; Sonnet barely does, but calls most neutral reviews negative.
- **2:** TabPFN, with no training at all, matches a random forest at every data size and in a prospective test on new cell lines, and beats ridge regression. **The pre-registered few-shot advantage did not appear**: with 64 labelled cell lines, random forest was better.

## Layout

```
icl_llm/          Part 1b: Claude wrapper (replay cache, spending cap), experiments, analysis
icl_bio/          Part 2: the pre-registered study (study.py) and its analysis
models/           the trained tiny transformer (0.7 MB)
runs/part1/       recorded Claude answers, the spend ledger and results
runs/part2/       pre-registration (+ SHA-256), clarifications, every model's predictions scored
tests/            offline tests (no key, no data, no weights)
```

## Running it yourself

```bash
cd in-context-learning
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
```

**Part 1b live:** set `ANTHROPIC_API_KEY` (your own key; ideally a separate key with a monthly spending limit) and run `python -m icl_llm.experiments all`. Cached answers replay for free; a cap (`ICL_TOTAL_CAP`, default $5) is checked before every call. Re-running everything costs about $0.40. Thinking is switched off for these measurements; Opus 5.5 doesn't allow that, so Haiku and Sonnet are used.

**Part 2 live** needs:
1. DepMap 24Q4 (Figshare) processed by `agentic-ai-for-science/target_discovery/prep.py`, then converted to `.npz` (see `icl_bio/study.py`), and for the prospective test the DepMap 26Q3 files (`ScreenGeneEffect.csv`, `CRISPRScreenMap.csv`, `Model.csv`, the protein-coding expression matrix) from the DepMap portal.
2. The TabPFN v2 regressor weights, `tabpfn-v2-regressor.ckpt` (44 MB) from [Prior-Labs/TabPFN-v2-reg](https://huggingface.co/Prior-Labs/TabPFN-v2-reg).
3. `python -m icl_bio.study curve | lineage | prospective` (about 4 hours on a laptop CPU).

`TABPFN_DISABLE_TELEMETRY=1` is set by the code, so TabPFN never contacts Prior Labs' telemetry service.

**Intel Macs:** PyTorch stopped publishing Intel-Mac builds after 2.2.2, which needs numpy 1.x; `requirements.txt` handles this automatically.

## Credits and licences

- TabPFN: Hollmann et al. (2025), *Nature*. Weights under the Prior Labs License 1.1 (Apache 2.0 with an attribution requirement). Built with TabPFN.
- DepMap data: CC BY 4.0 (DepMap 24Q4 and 26Q3 Public releases). Project Score: Behan et al. (2019), *Nature*.
- Garg et al. (2022), "What can transformers learn in-context?"; von Oswald et al. (2023), "Transformers learn in-context by gradient descent"; Wei et al. (2023), "Larger language models do in-context learning differently"; Zhao et al. (2021), "Calibrate before use".
