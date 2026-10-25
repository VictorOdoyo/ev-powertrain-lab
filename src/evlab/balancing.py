import numpy as np


def bleed(soc_groups, cell_voltage_v, maximum_a=0.2, threshold_soc=0.01):
    from .core import finite, positive

    z = finite(soc_groups, real=True, ndim=1)
    voltage = finite(cell_voltage_v, real=True)
    if voltage.shape != z.shape or np.any((z < 0) | (z > 1)) or np.any(voltage <= 0):
        raise ValueError("valid SOC and voltage per group required")
    current = np.where(z > z.min() + positive(threshold_soc), positive(maximum_a), 0.0)
    return {"bleed_a": current, "heat_w": float(voltage @ current)}


def step(soc_groups, pack_current_a, bleed_a, dt_s, group_capacity_ah=150.0):
    from .core import finite, positive

    z = finite(soc_groups, real=True, ndim=1)
    current = finite(bleed_a, real=True)
    if current.shape != z.shape or np.any(current < 0) or not np.isfinite(pack_current_a):
        raise ValueError("aligned nonnegative bleed required")
    result = z - (pack_current_a + current) * positive(dt_s) / (positive(group_capacity_ah) * 3600)
    if np.any((result < 0) | (result > 1)):
        raise ValueError("balancing crosses SOC boundary")
    return result
