id: pd_indirect_response
title: Indirect response model I (drug inhibits production of the response)
keywords: indirect response turnover kin kout inhibition of production delayed response biomarker IC50 Imax
library_model: pd_indirect_inhibition
---
Equation: dR/dt = kin * (1 - Imax*C/(IC50 + C)) - kout*R,  R(0) = R0 = kin/kout.

Mechanism: the response variable R (a biomarker) is continuously produced at rate kin and removed at rate kout*R. The drug inhibits its production, so R falls slowly towards a new level and recovers as the drug washes out.

Data signatures:
- Response falls below baseline after the dose, with a delay, and recovers over a time scale of about 1/kout.
- Time of maximal effect shifts later at higher doses.
- The maximal achievable drop is limited: R cannot fall below R0*(1 - Imax).

Note: Dayneka et al. describe four basic indirect-response models: the drug can inhibit or stimulate either production (kin) or loss (kout). This library implements only model I (inhibition of production). The others must be written as custom models.

Reference: Dayneka NL, Garg V, Jusko WJ (1993) J Pharmacokinet Biopharm 21:457-478.
