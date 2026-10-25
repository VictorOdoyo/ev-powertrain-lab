# Thermal coupling

A lumped node follows C dT/dt = Q - (T-Ta)/R and uses its exact held-input exponential transition. The network model constructs a symmetric conduction Laplacian and ambient conductance vector, then advances an augmented linear system exactly.

An isolated network must conserve sum(C T). Conductances must be nonnegative and finite, capacities positive. The published isolated-network study audits this invariant.

Pump power follows flow cubed; cooling conductance follows flow to the power 0.8. These are illustrative engineering laws. A thermostat adds on/off hysteresis and minimum dwell. Its clock cannot move backward.

The vehicle replay feeds battery heat and motor copper loss into separate thermal nodes. It does not include a spatial motor temperature field, semiconductor junction network, or calibrated coolant circuit. Motor current rating tapers between 80 and 110 deg C; battery discharge and charge use their separately declared envelopes.
