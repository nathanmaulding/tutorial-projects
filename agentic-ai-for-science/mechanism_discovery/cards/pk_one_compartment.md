id: pk_one_compartment
title: One-compartment pharmacokinetics, IV bolus, linear elimination
keywords: pharmacokinetics IV bolus clearance volume half-life monoexponential linear dose proportional
library_model: pk_1cmt_iv
---
Equation: C(t) = Dose / V * exp(-(CL/V) * t)

Assumptions: the drug distributes instantly into one well-mixed volume V; elimination is first order with clearance CL. Half-life = 0.693 * V / CL.

Data signatures:
- log C against t is a straight line.
- Dose-normalised curves (C/dose) from different doses overlap (linear, dose-proportional PK).

Pitfalls: A bend in log C against t (fast early fall, slower late fall) indicates a second compartment. Dose-normalised curves that do not overlap indicate nonlinear (for example saturable) elimination.

Error model: assay error is usually proportional, so fit on a log scale.

Reference: Gabrielsson J, Weiner D, Pharmacokinetic and Pharmacodynamic Data Analysis, 5th ed (2016), ch. 2.
