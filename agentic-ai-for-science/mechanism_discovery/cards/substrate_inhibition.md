id: substrate_inhibition
title: Substrate inhibition (rate falls at high substrate)
keywords: substrate inhibition Haldane rate decreases high substrate Ksi dead-end
library_model: substrate_inhibition
---
Equation: v = Vmax * S / (Km + S + S^2 / Ksi)

Mechanism: a second substrate molecule binds the enzyme-substrate complex to form an unproductive ESS complex.

Data signatures:
- Rate rises, peaks, then declines as S increases further.
- Michaelis-Menten residuals become strongly negative at the highest S values.
- Peak rate occurs at S = sqrt(Km * Ksi).

Pitfalls: Needs measurements beyond the peak to determine Ksi; extrapolating to higher S is sensitive to Ksi.

Reference: Haldane JBS, Enzymes (1930); Cornish-Bowden (2012) ch. 6.
