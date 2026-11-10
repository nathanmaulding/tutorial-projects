id: hill_cooperativity
title: Hill equation (cooperative substrate binding, sigmoidal kinetics)
keywords: cooperativity sigmoidal allosteric Hill coefficient n K half-saturation
library_model: hill
---
Equation: v = Vmax * S^n / (K^n + S^n)

Mechanism: an empirical description of cooperative binding across multiple sites. n > 1 means positive cooperativity (sigmoidal curve), n = 1 reduces to Michaelis-Menten, n < 1 means negative cooperativity.

Data signatures:
- Sigmoidal v against S: an initial lag at low S, then a steep rise, then saturation.
- Michaelis-Menten fits leave residuals that are negative at low S, positive in the middle, negative near saturation.
- The Hill plot log(v/(Vmax-v)) against log(S) has slope n.

Pitfalls: n is poorly determined unless several S values fall near K. Hill is phenomenological; n is not the number of binding sites.

Reference: Hill AV (1910) J Physiol 40:iv-vii; Cornish-Bowden (2012) ch. 12.
