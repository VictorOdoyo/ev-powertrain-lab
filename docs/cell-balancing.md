# Heterogeneous cells and passive balancing
The standalone cell-bank extension models unequal series-cell capacities and ohmic resistances with one RC polarization state per cell. It does not replace the existing forward vehicle pack.

For string current I and shunt conductance g, terminal voltage is (OCV-vp-r0*I)/(1+r0*g). Cell current is I+g*V. Shunt dissipation is g*V², not a prescribed constant current. A measured-voltage hysteresis controller limits simultaneous branches. Measurements are idealized as instantaneous shunt-off samples; the noisy case perturbs those observations.

At fixed 25 C and unit coulombic efficiency, chemical energy is capacity_c times the analytic integral of OCV(SOC). Adding C1*vp²/2 yields a stored-energy derivative equal to minus load, shunt, ohmic and relaxation powers. The adaptive ODE integrates those same powers and terminates at the first SOC boundary.

The SOC-cutoff studies isolate that boundary; they are not complete voltage/thermal battery-protection algorithms. Balancing dissipates energy and cannot repair capacity loss. [TI passive-balancing application guidance](https://www.ti.com/lit/pdf/sluaap7) supplies hardware context; no TI hardware is simulated or validated.

Run `evlab run --case cell-passive-balancing --output outputs/cells`. Compare with `cell-unbalanced-reference` and `cell-noisy-balancing` in the [gallery](results.md).
