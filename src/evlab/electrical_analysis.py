import numpy as np

from .core import positive


def voltage_ellipse(speed_rad_s, bus_v, model=None):
    from .motor import PMSM

    p = PMSM() if model is None else model
    we = p.pole_pairs * speed_rad_s
    matrix = np.array([[p.resistance_ohm, -we * p.lq_h], [we * p.ld_h, p.resistance_ohm]])
    offset = np.array([0, we * p.flux_wb])
    center = -np.linalg.solve(matrix, offset)
    return {"center_a": center, "quadratic": matrix.T @ matrix, "radius_squared_v2": positive(bus_v) ** 2 / 3}
