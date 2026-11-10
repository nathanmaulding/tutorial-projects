id: first_order_regime
title: First-order (linear) regime of enzyme kinetics, S much less than Km
keywords: linear first-order low substrate Vmax/Km specificity constant identifiability unsaturated
library_model: linear_rate
---
Equation: v = k * S, with k = Vmax / Km (the specificity constant).

When it applies: every substrate concentration is well below Km, so the Michaelis-Menten curve is still in its linear part.

Data signatures:
- v is proportional to S with no visible curvature.
- A Michaelis-Menten fit returns huge, highly uncertain Km and Vmax that are almost perfectly correlated; only their ratio is determined.

What to report: the linear model with k, and state that Km and Vmax are not identifiable from these data (a lower bound on Km may be given). Predictions within the measured range are reliable; extrapolation to high S is not.

Reference: Cornish-Bowden (2012) ch. 2; Raue A et al. (2009) Bioinformatics 25:1923-1929 (practical identifiability).
