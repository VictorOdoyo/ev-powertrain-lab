# Extended benches

The voltage ellipse is the quadratic form of the PMSM steady voltage map. Its center and matrix are checked against independently evaluated d-q voltages.

Thevenin impedance is R0 + R1/(1+j omega R1 C1). Complex least squares fits positive R0, R1 and tau using logarithmic parameters. Broadband excitation is required. Perfect recovery from synthetic impedance is an identifiability check, not a real battery fit.

The standalone battery hysteresis state moves toward a signed voltage target with a throughput-dependent exponential. It freezes at rest. The illustrative iron-loss function separates terms proportional to frequency and frequency squared. Neither hysteresis nor iron loss is included in the main drive replay.

Chassis APIs expose axle weight transfer, a friction circle and a smooth slip-force law. These are separate benches; the main longitudinal replay does not simulate a lateral vehicle or tire slip dynamics.

The DC precharge model includes capacitor energy, source energy and resistor heat. The contactor model latches on persistent overcurrent with explicit zero-current reset. These do not claim an automotive safety qualification.

The deadbeat current reference inverts the exact held-speed electrical transition then applies a voltage clamp. Exact one-step attainment requires that the unclipped solution is feasible. The jerk limiter bounds acceleration and its per-step change; it does not anticipate stopping distance and can overshoot a changed target.
