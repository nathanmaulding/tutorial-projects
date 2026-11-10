id: method_model_selection
title: Method: comparing candidate models with AIC and BIC
keywords: model selection AIC BIC information criterion parsimony overfitting compare models likelihood
library_model: none
---
AIC = n*log(RSS/n) + 2k;  BIC = n*log(RSS/n) + k*log(n). Lower is better; k is the number of parameters.

Rules of thumb for the difference from the best model (delta):
- delta < 2: essentially equivalent support.
- delta 2 to 6: positive evidence for the better model.
- delta > 6 (BIC) or > 10 (AIC): strong evidence.

Only compare models fitted to the same data with the same error model (additive vs proportional residuals change the scale).

A lower information criterion is necessary but not sufficient: also check residuals for structure and parameters for identifiability. The best model in a list can still be wrong if the true mechanism is not in the list.

Reference: Burnham & Anderson (2002).
