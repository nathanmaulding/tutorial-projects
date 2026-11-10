id: pk_two_compartment
title: Two-compartment pharmacokinetics, IV bolus (distribution then elimination)
keywords: two compartment biexponential distribution phase alpha beta peripheral intercompartmental clearance Q
library_model: pk_2cmt_iv
---
Equation: C(t) = Dose/V1 * (A*exp(-alpha*t) + B*exp(-beta*t)), with micro-constants k10 = CL/V1, k12 = Q/V1, k21 = Q/V2; alpha and beta are the roots of x^2 - (k10+k12+k21) x + k10*k21 = 0.

Parameters: CL (elimination clearance), V1 (central volume), Q (intercompartmental clearance), V2 (peripheral volume).

Data signatures:
- log C against t is biphasic: a steep early distribution phase then a shallower terminal phase.
- A one-compartment fit leaves residuals that are positive early, negative in the middle and positive late (a runs pattern over time).

Pitfalls: Needs early samples to capture the distribution phase; with sparse early data Q and V1 are poorly determined.

Reference: Gabrielsson & Weiner (2016), ch. 2-3.
