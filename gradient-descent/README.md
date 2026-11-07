# Gradient Descent from Scratch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nathanmaulding/tutorial-projects/blob/main/gradient-descent/gradient_descent_from_scratch.ipynb)

Build the algorithm behind nearly all of modern machine learning by hand, using only numpy, on the simplest possible problem: fitting a line to noisy points.

| | |
|---|---|
| **Level** | Beginner (high-school algebra; a little calculus helps) |
| **Runtime** | Under a minute, CPU only |
| **Libraries** | numpy, matplotlib |
| **Video** | coming soon |

## What's covered

1. **The problem:** models and parameters, fitting $y = wx + b$
2. **Loss functions:** mean squared error, drawn as a landscape
3. **Gradients:** deriving them, and checking them numerically
4. **Gradient descent:** the update rule, and watching it walk downhill
5. **Learning rate:** too small, just right, too big (divergence)
6. **Sanity check:** comparing with the exact closed-form answer (OLS)
7. **Stochastic gradient descent:** mini-batches, the version used in practice

The notebook ends with four exercises.

## Run locally

```bash
pip install -r requirements.txt
jupyter lab gradient_descent_from_scratch.ipynb
```
