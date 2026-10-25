# EV Powertrain Lab

[![Engineering checks](https://github.com/VictorOdoyo/ev-powertrain-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/VictorOdoyo/ev-powertrain-lab/actions/workflows/ci.yml)

An electrical engineering laboratory that connects PMSM torque and voltage limits, inverter losses, battery polarization and charge limits, thermal derating, and achieved vehicle motion. Separate fast benches resolve sampled current control; 24 reproducible studies retain raw arrays and source provenance.

![Forward urban drive](docs/gallery/urban-drive/figure.png)

## Run it

```sh
uv sync --locked --extra dev
uv run evlab list
uv run evlab run --case urban-drive --output outputs/urban
uv run evlab run --case field-weakening --output outputs/weakening
uv run evlab verify outputs/urban
```

Or install a built wheel with pip and use the same `evlab` CLI. Python 3.11 or newer is supported.

## What you can investigate

- Motor d-q power conservation, independent MTPA references, exact frozen-speed dynamics, coupled rotor integration and polynomial field weakening.
- SVPWM, switching, dead-time voltage error, conduction/switching losses, PI anti-windup, delays, ADC and encoder quantization, and deadbeat control.
- Battery SOC and RC polarization, voltage-collapse and current limits, series-cell imbalance, balancing, CC-CV charging, impedance fitting, observer gating and online resistance.
- Road load, gear losses, limited acceleration, regenerative/friction braking, cooling, thermal shutdown, and conditional road-load uncertainty.
- Standalone tire-force envelopes, axle transfer, DC-link precharge and contactor latching.

The main drive replay uses quasi-steady feasible motor points and forward vehicle/battery/thermal integration. Standalone hysteresis, iron-loss, chassis, precharge and advanced-control benches are explicitly separated from it. All coefficients and driving references are illustrative; no calibrated EV range or hardware-safety claim is made.

## Evidence and methods

The [computed gallery](docs/results.md) contains all 24 studies, CSV/JSON exports, NPZ arrays, plots, HTML reports and SHA-256 manifests. Infeasible operating points are masked rather than presented as attained motion.

| Read | Purpose |
|---|---|
| [Motor model](docs/motor-model.md) | Peak d-q equations, conservation and constraints |
| [Battery model](docs/battery-model.md) | Signs, exact RC, charging and heat |
| [Vehicle integration](docs/vehicle-integration.md) | Achieved speed and brake allocation |
| [Digital controls](docs/digital-controls.md) | Saturation, delays and observer separation |
| [Thermal model](docs/thermal-model.md) | Cooling, conservation and derating |
| [Extended benches](docs/extended-benches.md) | Optional models and their limits |
| [Validation](docs/validation.md) | Independent checks and validation scope |
| [Reproducibility](docs/reproducibility.md) | Source freezing, archives and replay |
| [Architecture](docs/architecture.md) | Module map and infrastructure attribution |

## Reproduce the catalog

```sh
uv run evlab batch --data-only --output outputs/catalog
uv run evlab render outputs/catalog
uv run evlab verify outputs/catalog
uv run python examples/build_gallery.py outputs/catalog --destination docs/gallery
```

The gallery builder requires the complete default catalog captured from one clean committed source revision. Report replay preserves that original revision.

## Development

```sh
uv run pytest -q
uv run ruff check src tests examples
uv run ruff format --check src tests examples
uv build
```

CI covers Linux/Windows, Python 3.11/3.13, formatting and an installed-wheel report outside the checkout. See [contributing](CONTRIBUTING.md) and [release notes](CHANGELOG.md). MIT licensed.
