id: michaelis_menten
title: Michaelis-Menten kinetics (single substrate, hyperbolic)
keywords: enzyme initial rate substrate saturation hyperbolic Vmax Km
library_model: mm
---
Equation: v = Vmax * S / (Km + S)

Mechanism: E + S <-> ES -> E + P with rapid equilibrium or steady state. Vmax = kcat * [E]total; Km is the substrate concentration giving half-maximal rate.

Data signatures:
- v rises linearly at low S and saturates at high S (a hyperbola).
- A Lineweaver-Burk plot (1/v against 1/S) is a straight line.
- Rate at S = Km is exactly Vmax/2.

Identifiability: Km and Vmax are only both determined if S spans roughly 0.2*Km to 5*Km. If every S is far below Km, only the ratio Vmax/Km is determined (see card first_order_regime).

Pitfalls: Sigmoidal curves (Hill) or curves that fall at high S (substrate inhibition) are not Michaelis-Menten; check residuals against S.

Reference: Michaelis L, Menten ML (1913) Biochem Z 49:333-369; Cornish-Bowden A, Fundamentals of Enzyme Kinetics, 4th ed (2012).
