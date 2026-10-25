import numpy as np

from .core import finite, positive


def friction_circle(longitudinal_n, lateral_n, normal_n, coefficient=0.8):
    capacity = positive(normal_n) * positive(coefficient)
    lateral = float(finite(lateral_n, real=True))
    if abs(lateral) > capacity:
        raise ValueError("lateral force exceeds tire friction circle")
    remaining = np.sqrt(max(0, capacity**2 - lateral**2))
    return float(np.clip(longitudinal_n, -remaining, remaining))


def axle_loads(mass_kg, acceleration_m_s2, wheelbase_m=2.8, cg_front_distance_m=1.3, cg_height_m=0.5):
    mass = positive(mass_kg)
    base = positive(wheelbase_m)
    height = positive(cg_height_m, allow_zero=True)
    if not 0 < cg_front_distance_m < base or not np.isfinite(acceleration_m_s2):
        raise ValueError("interior CG and finite acceleration required")
    rear = (mass * 9.80665 * cg_front_distance_m + mass * acceleration_m_s2 * height) / base
    front = mass * 9.80665 - rear
    if min(front, rear) < 0:
        raise ValueError("requested acceleration lifts an axle")
    return {"front_n": front, "rear_n": rear}
