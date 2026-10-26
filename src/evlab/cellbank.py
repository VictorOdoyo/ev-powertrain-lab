from dataclasses import dataclass

import numpy as np

from .core import finite, positive


@dataclass(frozen=True)
class Cells:
    capacity_ah: tuple = (2.0, 2.1, 1.9, 2.0)
    r0_ohm: tuple = (0.02, 0.02, 0.025, 0.02)
    r1_ohm: float = 0.01
    c1_f: float = 3000

    def __post_init__(self):
        q = finite(self.capacity_ah, real=True, ndim=1)
        r = finite(self.r0_ohm, real=True, ndim=1)
        if len(q) < 2 or q.shape != r.shape or np.any(q <= 0) or np.any(r <= 0):
            raise ValueError("aligned positive cell capacities and resistances required")
        positive(self.r1_ohm)
        positive(self.c1_f)
        object.__setattr__(self, "capacity_ah", tuple(q))
        object.__setattr__(self, "r0_ohm", tuple(r))

    @property
    def count(self):
        return len(self.capacity_ah)


def validate_state(value, model=None):
    p = Cells() if model is None else model
    x = finite(value, real=True)
    if x.shape != (2, p.count) or np.any((x[0] < 0) | (x[0] > 1)):
        raise ValueError("SOC and polarization per cell required")
    return x


def ports(value, string_current_a, shunt_s=None, model=None):
    p = Cells() if model is None else model
    x = validate_state(value, p)
    current = float(finite(string_current_a, real=True))
    conductance = np.zeros(p.count) if shunt_s is None else finite(shunt_s, real=True)
    if conductance.shape != (p.count,) or np.any(conductance < 0):
        raise ValueError("nonnegative shunt conductance per cell required")
    ocv = 3.1 + x[0] + 0.06 * np.tanh(10 * (x[0] - 0.5))
    resistance = np.asarray(p.r0_ohm)
    voltage = (ocv - x[1] - resistance * current) / (1 + resistance * conductance)
    cell_current = current + conductance * voltage
    return {
        "voltage_v": voltage,
        "cell_current_a": cell_current,
        "load_w": float(current * sum(voltage)),
        "bleed_w": float(sum(conductance * voltage**2)),
        "ohmic_w": float(sum(resistance * cell_current**2)),
        "relaxation_w": float(sum(x[1] ** 2 / p.r1_ohm)),
    }
