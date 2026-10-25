# Validation scope

Tests cover transform power invariance, motor power balance, exact versus integrated transitions, independent MTPA and constrained field-weakening references, battery coulomb and RC solutions, CC-CV voltage limits, isolated thermal energy, sensor residuals, observer covariance, and forward-drive limits.

Published field-weakening masks retain unavailable points. The no-load voltage check can open a drive that cannot produce a feasible controlled operating point. It is not a detailed diode freewheel transient model.

The road-load Monte Carlo study varies physical mass, drag area and rolling resistance for a prescribed speed. Its wheel-work interval is conditional on those distributions; it is not a confidence interval on real EV range.

Small algebraic roundoff residuals do not prove model accuracy. Hardware, calibrated cell chemistry, road tests, regulatory compliance and switching EMI validation remain outside this numerical laboratory.

CI checks Python 3.11/3.13 on Linux/Windows, formatting, and an installed wheel run from outside the source tree.
