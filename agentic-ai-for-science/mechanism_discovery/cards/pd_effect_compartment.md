id: pd_effect_compartment
title: Effect-compartment (link) model for delayed drug effect
keywords: hysteresis delay effect compartment ke0 link model counterclockwise lag distribution to site of action
library_model: pd_effect_compartment
---
Equations: dCe/dt = ke0 * (C - Ce),  E = E0 + Emax * Ce / (EC50 + Ce)

Mechanism: the drug must distribute to a hypothetical effect site; ke0 sets how fast the effect-site concentration Ce follows plasma concentration C.

Data signatures:
- Counter-clockwise hysteresis loop in E against C (effect is lower on the way up than on the way down).
- Peak effect occurs after peak concentration, at a delay that does not depend on dose.

Contrast with indirect response: in indirect-response models the time of peak effect shifts later as dose increases, and the response returns to baseline at a rate set by kout.

Reference: Sheiner LB et al. (1979) Clin Pharmacol Ther 25:358-371.
