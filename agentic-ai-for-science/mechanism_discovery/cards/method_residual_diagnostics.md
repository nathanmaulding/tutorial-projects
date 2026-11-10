id: method_residual_diagnostics
title: Method: reading residuals to find a missing mechanism
keywords: residuals diagnostics systematic misfit runs test trend structure missing mechanism model inadequacy
library_model: none
---
If a model is right, residuals look like noise: no trend with any input, signs that alternate randomly, and similar size across groups.

What patterns suggest:
- Residuals trend with an input (S, I, dose, t): the model gets that input's effect wrong.
- Long runs of same-sign residuals along S or t: wrong curve shape (missing curvature, a second phase, saturation).
- One group (an inhibitor level or a dose) fit badly on average: the model's dependence on that factor is wrong. If the misfit grows at the extremes (highest inhibitor, highest dose), extrapolation will be poor.

When every library model leaves structured residuals, the true mechanism is probably not in the library. Look at what the data do at the extremes (for example, does inhibition level off rather than go to zero? does the response go below the floor a candidate model allows?) and write a custom model that captures it.

Reference: Gabrielsson & Weiner (2016) ch. 7; Bates DM, Watts DG, Nonlinear Regression Analysis (1988).
