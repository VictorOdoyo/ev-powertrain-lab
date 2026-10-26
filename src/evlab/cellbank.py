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
