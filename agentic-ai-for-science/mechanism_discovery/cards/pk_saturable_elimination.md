id: pk_saturable_elimination
title: Saturable (Michaelis-Menten) elimination, nonlinear pharmacokinetics
keywords: nonlinear pharmacokinetics saturable elimination Michaelis-Menten Vmax Km dose-dependent half-life capacity limited
library_model: pk_1cmt_iv_mm
---
Equation (ODE): dC/dt = -(Vmax/V) * C / (Km + C), C(0) = Dose/V.
Vmax: maximum elimination rate (mg/h); Km: concentration at half-maximal elimination (mg/L).

Data signatures:
- Dose-normalised curves do NOT overlap: at the higher dose, concentrations fall more slowly at first (elimination is saturated), so exposure rises more than proportionally with dose.
- At high concentration the decline is roughly linear in C (zero order); at low concentration it becomes exponential (first order).
- A linear one-compartment model fit to all doses together leaves residuals that differ systematically by dose.

Why it matters: predictions for a higher, untested dose can be badly wrong if elimination is assumed linear.

Reference: Gabrielsson & Weiner (2016), ch. 3 (nonlinear kinetics); phenytoin is the classic clinical example.
