import numpy as np


class PI:
    def __init__(self, kp, ki, limit, tracking_s=0.005):
        from .core import positive

        self.kp = positive(kp, allow_zero=True)
        self.ki = positive(ki, allow_zero=True)
        self.limit = positive(limit)
        self.tracking = positive(tracking_s)
        self.integral = 0.0

    def update(self, error, dt_s, feedforward=0.0):
        from .core import positive

        dt = positive(dt_s)
        if not np.isfinite(error + feedforward):
            raise ValueError("finite controller inputs required")
        raw = self.kp * error + self.integral + feedforward
        applied = float(np.clip(raw, -self.limit, self.limit))
        self.integral += dt * (self.ki * error + (applied - raw) / self.tracking)
        return applied


def current_gains(model, bandwidth_hz=500.0):
    from .core import positive

    omega = 2 * np.pi * positive(bandwidth_hz)
    return {"kp": omega * np.array([model.ld_h, model.lq_h]), "ki": omega * model.resistance_ohm}


class CurrentLoop:
    def __init__(self, model=None, bandwidth_hz=500.0, tracking_s=0.001):
        from .core import positive
        from .motor import PMSM

        self.model = PMSM() if model is None else model
        self.gains = current_gains(self.model, bandwidth_hz)
        self.integral = np.zeros(2)
        self.tracking = positive(tracking_s)

    def update(self, reference, current, omega_m, dc_voltage_v, dt_s):
        from .core import finite, positive
        from .inverter import limit_vector

        p = self.model
        dt = positive(dt_s)
        target = finite(reference, real=True)
        measured = finite(current, real=True)
        if target.shape != (2,) or measured.shape != (2,):
            raise ValueError("d-q current vectors required")
        error = target - measured
        we = p.pole_pairs * omega_m
        feed = np.array([-we * p.lq_h * measured[1], we * (p.ld_h * measured[0] + p.flux_wb)])
        raw = self.gains["kp"] * error + self.integral + feed
        applied = limit_vector(raw, dc_voltage_v)
        self.integral += dt * (self.gains["ki"] * error + (applied - raw) / self.tracking)
        return applied


class Delay:
    def __init__(self, samples, initial):
        from collections import deque

        from .core import finite, integer

        self.count = integer(samples, 0)
        self.queue = deque([finite(initial, real=True).copy() for _ in range(samples)])
        self.shape = np.asarray(initial).shape

    def update(self, value):
        from .core import finite

        v = finite(value, real=True)
        if v.shape != self.shape:
            raise ValueError("delayed command shape changed")
        self.queue.append(v.copy())
        return self.queue.popleft()


def quantize_current(current_a, full_scale_a=500.0, bits=12):
    from .core import finite, integer, positive

    v = finite(current_a, real=True)
    rail = positive(full_scale_a)
    integer(bits, 2)
    if bits > 24:
        raise ValueError("at most 24 ADC bits supported")
    step = 2 * rail / (2**bits - 1)
    return {
        "value": np.rint((np.clip(v, -rail, rail) + rail) / step) * step - rail,
        "clipped": abs(v) > rail,
        "lsb_a": step,
    }
