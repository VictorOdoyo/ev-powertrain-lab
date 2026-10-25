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
