id: reversible_inhibition
title: Reversible inhibition: competitive, uncompetitive and noncompetitive (mixed with alpha = 1)
keywords: inhibitor Ki competitive uncompetitive noncompetitive mixed inhibition apparent Km apparent Vmax
library_model: competitive_inhibition, uncompetitive_inhibition, noncompetitive_inhibition
---
Equations (I = inhibitor concentration, Ki = inhibition constant):
- Competitive (inhibitor binds free enzyme only):  v = Vmax*S / (Km*(1 + I/Ki) + S)
- Uncompetitive (binds ES complex only):            v = Vmax*S / (Km + S*(1 + I/Ki))
- Noncompetitive (binds E and ES equally):          v = Vmax*S / ((Km + S)*(1 + I/Ki))

Data signatures (compare curves at different I):
- Competitive: apparent Km increases with I, Vmax unchanged; inhibition is overcome at high S. Lineweaver-Burk lines intersect on the y-axis.
- Uncompetitive: apparent Vmax and apparent Km both decrease by the same factor; parallel Lineweaver-Burk lines.
- Noncompetitive: apparent Vmax decreases, Km unchanged; lines intersect on the x-axis.

All three models are "complete" inhibition: as I goes to infinity the rate goes to zero. If activity levels off at a nonzero fraction at saturating inhibitor, none of these models apply.

Fit all three to the full data set (all I levels together) and compare with AIC/BIC and residuals grouped by I.

Reference: Cornish-Bowden (2012) ch. 6; Segel IH, Enzyme Kinetics (1975).
