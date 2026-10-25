"""Validated reproducible run requests and lazy study dispatch."""

import json
from dataclasses import dataclass
from pathlib import Path

CASES = {
    "road-load-uncertainty": ["studies_bench", "energy_uncertainty", "Conditional road-load ensemble"],
    "observer-outlier-gate": ["studies_bench", "observer_gate", "Gated voltage outlier rejection"],
    "urban-drive": ["studies_drive", "urban", "Forward urban drive"],
    "highway-drive": ["studies_drive", "highway", "Voltage-limited highway motion"],
    "hill-drive": ["studies_drive", "hill", "Road grade and electrical load"],
    "friction-braking": ["studies_drive", "friction_only", "Friction-only brake allocation"],
    "full-pack-braking": ["studies_drive", "full_pack", "Regeneration near full SOC"],
    "hot-pack-shutdown": ["studies_drive", "hot_pack", "Battery thermal shutdown"],
    "biased-current-observer": ["studies_drive", "biased", "Sensor bias and SOC observer"],
    "heavy-vehicle": ["studies_drive", "heavy", "Increased vehicle mass"],
    "hot-motor-derating": ["studies_drive", "hot_motor", "Motor derating without coolant"],
    "foc-step": ["studies_bench", "foc", "Fast exact-plant current control"],
    "foc-delay": ["studies_bench", "foc_delay", "Sampled current-loop delay"],
    "cccv-charge": ["studies_bench", "charging", "CC-CV charging with grid loss"],
    "field-weakening": ["studies_bench", "field_weakening", "Current and voltage feasibility"],
    "mtpa-reference": ["studies_bench", "mtpa_curve", "Independent MTPA references"],
    "battery-pulse": ["studies_bench", "battery_pulse", "Exact RC polarization recovery"],
    "thermal-network": ["studies_bench", "thermal_network", "Conservative thermal coupling"],
    "svpwm-average": ["studies_bench", "modulation", "Averaged modulation voltage closure"],
}


@dataclass(frozen=True)
class ExperimentConfig:
    case: str = "urban-drive"
    resolution_scale: int = 1
    seed: int = 42

    def __post_init__(self):
        if not isinstance(self.case, str) or self.case not in CASES:
            raise ValueError("unknown experiment case")
        if type(self.resolution_scale) is not int or not 1 <= self.resolution_scale <= 3:
            raise ValueError("resolution_scale must be an integer from 1 to 3")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")

    def to_dict(self):
        return {"case": self.case, "resolution_scale": self.resolution_scale, "seed": self.seed}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) - {"case", "resolution_scale", "seed"}:
            raise ValueError("request must be an object containing only case, resolution_scale and seed")
        return cls(**value)


def load_config(path):
    return ExperimentConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def run(config, progress=None):
    from importlib import import_module

    from .reporting import provenance, stamp
    from .results import validate

    if not isinstance(config, ExperimentConfig):
        raise ValueError("validated ExperimentConfig required")
    configuration = config.to_dict()
    source = provenance(configuration)
    module, function, _ = CASES[config.case]
    if progress is not None:
        progress("Solving " + config.case)
    result = getattr(import_module("evlab." + module), function)(
        scale=config.resolution_scale, seed=config.seed
    )
    from dataclasses import asdict

    from .battery import Pack
    from .motor import PMSM
    from .vehicle import Vehicle

    result["parameters"]["model_defaults"] = {
        "motor": asdict(PMSM()),
        "pack": asdict(Pack()),
        "vehicle": asdict(Vehicle()),
    }
    validate(result)
    if result["name"] != config.case:
        raise ValueError("study returned a different case name")
    return stamp(result, configuration, source)
