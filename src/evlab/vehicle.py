from dataclasses import dataclass

import numpy as np


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


def road_forces(speed_m_s, grade=0.0, wind_m_s=0.0, model=None):
    from .core import finite

    p = Vehicle() if model is None else model
    speed = finite(speed_m_s, real=True)
    relative = speed - wind_m_s
    angle = np.arctan(grade)
    aero = 0.5 * p.air_density * p.drag_area_m2 * relative * abs(relative)
    rolling = p.rolling * p.mass_kg * 9.80665 * np.cos(angle) * np.tanh(speed / 0.1)
    gravity = p.mass_kg * 9.80665 * np.sin(angle)
    return {"aero_n": aero, "rolling_n": rolling, "grade_n": gravity, "total_n": aero + rolling + gravity}


def motor_torque(force_n, speed_m_s, model=None):
    p = Vehicle() if model is None else model
    force = np.asarray(force_n, float)
    drive = force * np.asarray(speed_m_s) >= 0
    torque = force * p.wheel_radius_m / p.gear
    return np.where(drive, torque / p.gear_efficiency, torque * p.gear_efficiency)


def wheel_force(torque_nm, speed_m_s, model=None):
    p = Vehicle() if model is None else model
    torque = np.asarray(torque_nm, float)
    drive = torque * np.asarray(speed_m_s) >= 0
    force = torque * p.gear / p.wheel_radius_m
    return np.where(drive, force * p.gear_efficiency, force / p.gear_efficiency)
