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
