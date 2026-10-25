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


def network(capacities, edges, ambient_conductance):
    from .core import finite

    c = finite(capacities, real=True, ndim=1)
    g = finite(ambient_conductance, real=True, ndim=1)
    if c.shape != g.shape or np.any(c <= 0) or np.any(g < 0):
        raise ValueError("positive capacities and nonnegative conductances required")
    k = np.diag(g)
    for a, b, value in edges:
        if a == b or not 0 <= a < len(c) or not 0 <= b < len(c) or value < 0:
            raise ValueError("valid thermal edge required")
        k[a, a] += value
        k[b, b] += value
        k[a, b] -= value
        k[b, a] -= value
    return {"capacity": c, "conductance": k, "ambient": g}
