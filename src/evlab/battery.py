from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Pack:
    series: int = 96
    parallel: int = 3
    cell_capacity_ah: float = 50.0
    cell_r0_ohm: float = 0.002
    cell_r1_ohm: float = 0.001
    cell_c1_f: float = 30000.0
    minimum_cell_v: float = 2.8
    maximum_cell_v: float = 4.25

    def __post_init__(self):
        from .core import integer, positive

        integer(self.series)
        integer(self.parallel)
        for key in (
            "cell_capacity_ah",
            "cell_r0_ohm",
            "cell_r1_ohm",
            "cell_c1_f",
            "minimum_cell_v",
            "maximum_cell_v",
        ):
            positive(getattr(self, key), key)
        if self.maximum_cell_v <= self.minimum_cell_v:
            raise ValueError("ordered cell voltage limits required")

    @property
    def capacity_c(self):
        return self.parallel * self.cell_capacity_ah * 3600

    @property
    def r0(self):
        return self.cell_r0_ohm * self.series / self.parallel

    @property
    def r1(self):
        return self.cell_r1_ohm * self.series / self.parallel

    @property
    def c1(self):
        return self.cell_c1_f * self.parallel / self.series


def ocv(soc, temperature_c=25.0, model=None):
    from .core import finite

    p = Pack() if model is None else model
    z = finite(soc, real=True)
    t = finite(temperature_c, real=True)
    if np.any((z < 0) | (z > 1)):
        raise ValueError("SOC outside [0,1]")
    return p.series * (3.1 + 1.0 * z + 0.06 * np.tanh(10 * (z - 0.5)) - 0.0002 * (t - 25))
