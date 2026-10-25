from dataclasses import dataclass


@dataclass(frozen=True)
class Vehicle:
    mass_kg: float = 1800.0
    wheel_radius_m: float = 0.31
    gear: float = 9.0
    gear_efficiency: float = 0.96
    drag_area_m2: float = 0.62
    rolling: float = 0.01
    air_density: float = 1.225
    motor_inertia: float = 0.08
    wheel_inertia: float = 4.0

    def __post_init__(self):
        from .core import positive

        for key, value in vars(self).items():
            positive(
                value, key, allow_zero=key in ("motor_inertia", "wheel_inertia", "rolling", "drag_area_m2")
            )
        if self.gear_efficiency > 1:
            raise ValueError("gear efficiency cannot exceed one")

    @property
    def effective_mass(self):
        return (
            self.mass_kg + (self.motor_inertia * self.gear**2 + self.wheel_inertia) / self.wheel_radius_m**2
        )
