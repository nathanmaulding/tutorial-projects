id: method_identifiability
title: Method: practical identifiability of parameters
keywords: identifiability standard error correlation profile likelihood poorly determined parameters sloppy
library_model: none
---
A parameter is practically non-identifiable when the data cannot pin it down: its standard error is huge, or it is almost perfectly correlated with another parameter so that only a combination is determined.

What to do:
- Report which parameters are not identifiable rather than quoting their point estimates.
- Prefer a reduced model with only the identifiable combination (for example v = k*S instead of Michaelis-Menten when S is much less than Km), unless the extra parameters are needed for the intended predictions.
- Predictions inside the range of the data can still be reliable; extrapolations that depend on a non-identifiable parameter are not.

Reference: Raue A et al. (2009) Bioinformatics 25:1923-1929.
