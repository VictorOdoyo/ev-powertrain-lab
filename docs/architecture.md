# Architecture

| Layer | Modules | Responsibility |
|---|---|---|
| Electrical | transforms, motor, inverter | dq physics, voltage/current feasibility, modulation and losses |
| Stored energy | battery, charging, balancing | SOC, polarization, heat, charging, series-group checks |
| Mechanical and thermal | vehicle, thermal, powertrain | road load, gear conversion, cooling and coupled limits |
| Digital | control, estimation, diagnostics | sampled control, observer, parameter and sensor residuals |
| Replays | cycles, simulation, benches, statistics | synthetic drives, fast current loop, charging and uncertainty |
| Evidence | studies_drive, studies_bench, experiments | validated reproducible physical studies |
| Artifacts | results, reporting, batch, cli | sealed arrays, data exports, rendering, replay and verification |

Result/report/catalog infrastructure is adapted from this author's MIT-licensed adaptive-grid-protection project. EV model equations, numerical benches and studies are independently authored for this repository. No protection-network implementation is copied into the EV solver.
