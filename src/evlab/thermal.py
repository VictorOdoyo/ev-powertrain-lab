import numpy as np


def lumped_step(temperature_c, heat_w, ambient_c, capacity_j_k, resistance_k_w, dt_s):
    from .core import positive

    capacity = positive(capacity_j_k)
    resistance = positive(resistance_k_w)
    dt = positive(dt_s)
    if not np.all(np.isfinite([temperature_c, heat_w, ambient_c])):
        raise ValueError("finite thermal inputs required")
    equilibrium = ambient_c + heat_w * resistance
    return equilibrium + (temperature_c - equilibrium) * np.exp(-dt / (capacity * resistance))
