id: null_model
title: Null (no-effect) model and false discoveries
keywords: null no effect baseline constant false positive overfitting placebo noise significance
library_model: pd_no_effect
---
Equation: E = E0 (a constant: the drug has no effect on the measurement).

Always fit the null model alongside mechanistic models. A more complex model must earn its extra parameters: if its BIC is not clearly lower (by roughly 6 or more) and its effect parameters are poorly determined, the evidence for any effect is weak.

Signs that an apparent effect is noise:
- Effect parameters (Emax, Imax) have huge relative standard errors or confidence intervals that include zero.
- The fitted "effect" is small compared with the measurement noise and has no consistent pattern across doses or time.
- Residual diagnostics of the null model show no structure.

Reporting "no detectable effect" is a valid scientific result. Claiming an exposure-response relationship from noise is a false discovery.

Reference: Burnham KP, Anderson DR, Model Selection and Multimodel Inference, 2nd ed (2002).
