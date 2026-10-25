from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Config:
    cycle: str = "urban"
    duration_s: float = 60.0
    dt_s: float = 0.2
    initial_soc: float = 0.8
    observer_soc: float = 0.76
    ambient_c: float = 25.0
    battery_c: float = 25.0
    motor_c: float = 25.0
    mass_scale: float = 1.0
    cooling: bool = True
    regen: bool = True
    current_bias_a: float = 0.0
    voltage_bias_v: float = 0.0
    noise_v: float = 0.1
    seed: int = 42

    def __post_init__(self):
        from .core import integer, positive

        if self.cycle not in ("urban", "highway", "hill"):
            raise ValueError("known cycle required")
        for key in ("duration_s", "dt_s", "mass_scale"):
            positive(getattr(self, key), key)
        if self.dt_s > 1 or self.duration_s < 2 * self.dt_s:
            raise ValueError("resolved drive timing required")
        if not 0 < self.initial_soc < 1 or not 0 < self.observer_soc < 1:
            raise ValueError("interior initial SOC required")
        if not np.all(
            np.isfinite(
                [self.ambient_c, self.battery_c, self.motor_c, self.current_bias_a, self.voltage_bias_v]
            )
        ):
            raise ValueError("finite replay conditions required")
        if min(self.ambient_c, self.battery_c, self.motor_c) <= -273.15:
            raise ValueError("temperature below absolute zero")
        positive(self.noise_v, allow_zero=True)
        integer(self.seed, 0)
        if self.seed >= 2**32 or type(self.cooling) is not bool or type(self.regen) is not bool:
            raise ValueError("boolean switches and 32-bit seed required")

    def to_dict(self):
        return asdict(self)
