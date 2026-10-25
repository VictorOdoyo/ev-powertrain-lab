# Battery and charging

Positive current discharges the pack. The pack combines Ns series groups and Np parallel cells. Capacity scales with Np, resistances with Ns/Np, and polarization capacitance with Np/Ns.

SOC integrates coulombs, including 0.995 charge efficiency. RC polarization uses the exact held-current transition with tau=R1 C1. Terminal voltage is OCV - vp - R0 I. The OCV curve and temperature derivative are illustrative functions, not a fitted cell dataset.

The stable low-current quadratic root resolves constant-power load. Infeasible discriminants are rejected. Hardware current, instantaneous voltage, and SOC headroom constrain current. Individual series-group checks demonstrate why average pack voltage cannot certify every cell.

Heat separates ohmic, RC relaxation, reversible entropic, and charge-inefficiency terms. The aging API reports throughput-based equivalent cycles with an illustrative temperature acceleration; it has no calendar-aging calibration.

CC-CV uses predicted future SOC, RC state and terminal voltage. Temperature can stop charging entirely. Grid draw includes conversion efficiency and auxiliary consumption. Charge-completion dwell is a separate API.

Structural reference: [PyBaMM Thevenin implementation](https://docs.pybamm.org/en/pybamm-v26.10.0.0/_modules/pybamm/models/full_battery_models/equivalent_circuit/thevenin.html). No numerical cross-validation against PyBaMM is claimed.
