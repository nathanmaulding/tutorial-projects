# Tutorial Projects

Hands-on notebooks for learning and demonstrating concepts in machine learning, AI and quantitative methods. Each tutorial pairs **runnable code** with **plain-language explanations** of the ideas behind it, and comes with a video walkthrough.

## Tutorials

| # | Tutorial | What you'll learn | Run it | Video |
|---|----------|-------------------|--------|-------|
| 01 | [Gradient Descent from Scratch](gradient-descent/) | Loss functions, gradients, learning rates, SGD, all in plain numpy | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nathanmaulding/tutorial-projects/blob/main/gradient-descent/gradient_descent_from_scratch.ipynb) | coming soon |

| 02 | [Agentic AI for Science](agentic-ai-for-science/) | Build an AI agent that discovers mechanistic models from data, benchmark how far to trust it, then point it at real cancer CRISPR screens to hunt for drug targets with pre-registration and independent replication | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nathanmaulding/tutorial-projects/blob/main/agentic-ai-for-science/01_building_a_science_agent.ipynb) | coming soon |

*Planned: in-context learning in transformers.*

## How to run a tutorial

**Easiest:** click **Open in Colab**. It runs in your browser, nothing to install.

**Locally:**

```bash
git clone https://github.com/nathanmaulding/tutorial-projects.git
cd tutorial-projects/gradient-descent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

## How each tutorial is laid out

```
<tutorial-name>/
├── README.md           # summary, prerequisites, runtime, video link
├── <name>.ipynb        # the notebook: explanation, then code, then "what just happened"
└── requirements.txt    # pinned dependencies for this tutorial only
```

Every notebook is run automatically on each push (see `.github/workflows/test-notebooks.yml`), so the code in the repo always works.

## Adding a tutorial

Copy `_template/` to a new folder and follow `_template/README.md`, or in Claude Code run the `new-tutorial` skill.
