# Cell diagnostics and observability
The capacity bench regresses integrated discharged charge against independently observed SOC change. It rejects unexcited data and nonphysical fitted capacity. It cannot infer both unknown SOC and capacity from coulomb count alone.

The pulse bench fits V=OCV-r0*I with an unknown intercept and estimates residual-based slope uncertainty. Its samples assume unchanged SOC/polarization and instantaneous pulses. A long pulse contains RC relaxation and violates that simplified regression.

The relaxed-OCV bench inverts a monotone synthetic OCV curve, retaining out-of-range flags instead of extrapolating SOC. It assumes zero current and full relaxation. Sensor noise is seeded; fit truth is used only for synthesis and scoring.

Independent tests differentiate stored chemical energy against OCV, verify shunt KCL, close nonlinear energy balances, recover analytic regression parameters and test first-cell termination. See [cell balancing](cell-balancing.md) and [computed evidence](results.md).
