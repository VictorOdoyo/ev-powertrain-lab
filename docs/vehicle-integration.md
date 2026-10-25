# Forward vehicle replay

The route reference drives a PI wheel-force request. The solver applies motor current/voltage/temperature constraints and battery charge/discharge limits, converts delivered torque through directional gear efficiency, and advances actual speed.

Effective mass includes motor and wheel inertias. Road force combines relative-wind drag, smooth low-speed rolling resistance, and grade gravity. Grade is rise/run; sine and cosine use atan(grade). Main replay is forward-only and prevents negative speed.

Regenerative braking is limited by battery headroom, temperature and motor feasibility. Friction braking fills negative force demand up to its limit. Disabled-regeneration runs allocate braking to friction explicitly.

The mechanical audit compares actual kinetic-energy change with midpoint wheel-minus-road work. Battery power uses the previous battery state for the held operating point; the sampled terminal voltage is recorded after the state update. Do not multiply the latter by current and assume it equals the frozen-point battery power exactly.

Synthetic urban, highway and hill references are not WLTP or regulatory drive cycles. The primary replay uses quasi-steady motor operating points; fast FOC behavior is resolved in separate benches.
