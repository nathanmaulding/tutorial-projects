id: pk_oral_absorption
title: One-compartment pharmacokinetics with first-order oral absorption
keywords: oral absorption ka Bateman bioavailability Cmax Tmax apparent clearance CL/F
library_model: pk_1cmt_oral
---
Equation (Bateman function): C(t) = Dose*ka / (V*(ka - ke)) * (exp(-ke*t) - exp(-ka*t)), ke = CL/V.

With oral data alone, bioavailability F cannot be separated from CL and V: the fitted CL and V are apparent values (CL/F and V/F).

Data signatures:
- Concentration starts at zero, rises to a peak (Tmax), then declines.
- Flip-flop kinetics: if ka < ke the terminal slope reflects absorption, not elimination; ka and ke can swap without changing the fit.

Reference: Gabrielsson & Weiner (2016), ch. 2.
