import numpy as np

from .core import finite, positive


def friction_circle(longitudinal_n, lateral_n, normal_n, coefficient=0.8):
    capacity = positive(normal_n) * positive(coefficient)
    lateral = float(finite(lateral_n, real=True))
    if abs(lateral) > capacity:
        raise ValueError("lateral force exceeds tire friction circle")
    remaining = np.sqrt(max(0, capacity**2 - lateral**2))
    return float(np.clip(longitudinal_n, -remaining, remaining))
