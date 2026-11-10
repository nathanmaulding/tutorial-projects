id: pd_emax
title: Direct Emax and sigmoid Emax concentration-effect models
keywords: pharmacodynamics Emax EC50 sigmoid Hill concentration effect direct response baseline E0
library_model: pd_direct_emax, pd_sigmoid_emax
---
Equations (C = plasma concentration):
- Direct Emax:  E = E0 + Emax * C / (EC50 + C)
- Sigmoid Emax: E = E0 + Emax * C^gamma / (EC50^gamma + C^gamma)

Assumption: the effect responds instantly to plasma concentration (no delay).

Data signatures:
- Plotting E against C gives a single curve: points from rising and falling concentrations, and from different doses, fall on the same line, with no loop.
- Effect peaks at the same time as concentration.

Pitfalls: If E plotted against C traces a loop (hysteresis), the effect lags concentration: use an effect-compartment or indirect-response model instead. Gamma is poorly determined unless concentrations span EC50 well.

Reference: Holford NHG, Sheiner LB (1981) Clin Pharmacokinet 6:429-453; Gabrielsson & Weiner (2016) ch. 4.
