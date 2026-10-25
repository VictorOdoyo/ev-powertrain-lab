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
        if a == b or not 0 <= a < len(c) or not 0 <= b < len(c) or not np.isfinite(value) or value < 0:
            raise ValueError("valid thermal edge required")
        k[a, a] += value
        k[b, b] += value
        k[a, b] -= value
        k[b, a] -= value
    return {"capacity": c, "conductance": k, "ambient": g}


def network_step(temperature, heat_w, ambient_c, dt_s, model):
    from scipy.linalg import expm

    from .core import finite, positive

    t = finite(temperature, real=True, ndim=1)
    q = finite(heat_w, real=True, ndim=1)
    c, k, g = model["capacity"], model["conductance"], model["ambient"]
    if t.shape != c.shape or q.shape != c.shape:
        raise ValueError("matching thermal nodes required")
    n = len(c)
    aug = np.zeros((n + 1, n + 1))
    aug[:n, :n] = -k / c[:, None]
    aug[:n, n] = (q + g * ambient_c) / c
    return (expm(aug * positive(dt_s)) @ np.r_[t, 1])[:n]


def derate(temperature_c, onset_c=80.0, shutdown_c=110.0):
    from .core import finite

    if not np.isfinite(onset_c + shutdown_c) or onset_c >= shutdown_c:
        raise ValueError("ordered thermal limits required")
    return np.clip((shutdown_c - finite(temperature_c, real=True)) / (shutdown_c - onset_c), 0, 1)


def coolant(flow_fraction, maximum_conductance_w_k=100.0, maximum_pump_w=300.0):
    from .core import positive

    if not np.isfinite(flow_fraction) or not 0 <= flow_fraction <= 1:
        raise ValueError("flow fraction in [0,1] required")
    return {
        "conductance_w_k": positive(maximum_conductance_w_k) * flow_fraction**0.8,
        "pump_w": positive(maximum_pump_w) * flow_fraction**3,
    }


class Thermostat:
    def __init__(self, on_c=45.0, off_c=40.0, minimum_dwell_s=5.0):
        from .core import positive

        if off_c >= on_c:
            raise ValueError("hysteresis temperatures required")
        self.on_c, self.off_c = on_c, off_c
        self.dwell = positive(minimum_dwell_s)
        self.on = False
        self.changed = -np.inf
        self.time = -np.inf

    def update(self, time_s, temperature_c):
        if not np.isfinite(time_s + temperature_c) or time_s < self.time:
            raise ValueError("monotone finite thermal clock required")
        self.time = time_s
        target = temperature_c > self.on_c if not self.on else temperature_c > self.off_c
        if target != self.on and time_s - self.changed >= self.dwell:
            self.on = target
            self.changed = time_s
        return self.on
