import numpy as np


def predict(state, covariance, current_a, dt_s, model=None, process_std=(1e-5, 0.01)):
    from .battery import Pack, step
    from .core import finite, positive

    p = Pack() if model is None else model
    x = finite(state, real=True)
    covariance = finite(covariance, real=True, ndim=2)
    if x.shape != (2,) or covariance.shape != (2, 2) or np.linalg.eigvalsh(covariance).min() < -1e-12:
        raise ValueError("valid battery state covariance required")
    dt = positive(dt_s)
    a = np.exp(-dt / (p.r1 * p.c1))
    f = np.diag([1.0, a])
    q = np.asarray(process_std, float) ** 2 * dt
    if q.shape != (2,) or np.any(q < 0):
        raise ValueError("two process noise scales required")
    return {
        "state": step(x, current_a, dt, p),
        "covariance": f @ covariance @ f.T + np.diag(q),
        "transition": f,
    }


def correct(prediction, voltage_v, current_a, temperature_c=25.0, model=None, voltage_std_v=0.2, gate=16.0):
    from .battery import Pack, ocv_jacobian, terminal_voltage
    from .core import positive

    p = Pack() if model is None else model
    x = prediction["state"].copy()
    covariance = prediction["covariance"].copy()
    if not np.isfinite(voltage_v):
        raise ValueError("finite voltage sample required")
    h = np.array([float(ocv_jacobian(x[0], p)), -1.0])
    r = positive(voltage_std_v) ** 2
    innovation = voltage_v - float(terminal_voltage(x[0], x[1], current_a, temperature_c, p))
    s = float(h @ covariance @ h + r)
    nis = innovation**2 / s
    accepted = nis <= positive(gate)
    if accepted:
        gain = covariance @ h / s
        candidate = x + gain * innovation
        if not 0 <= candidate[0] <= 1:
            accepted = False
        else:
            x = candidate
            identity = np.eye(2) - np.outer(gain, h)
            covariance = identity @ covariance @ identity.T + np.outer(gain, gain) * r
            covariance = (covariance + covariance.T) / 2
    return {
        "state": x,
        "covariance": covariance,
        "innovation_v": innovation,
        "nis": nis,
        "accepted": bool(accepted),
    }
