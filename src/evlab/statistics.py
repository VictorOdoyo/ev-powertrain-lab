import numpy as np

from .core import integer, stream


def correlated_samples(mean, covariance, count=1000, seed=42):
    from .core import finite

    mu = finite(mean, real=True, ndim=1)
    cov = finite(covariance, real=True, ndim=2)
    if cov.shape != (len(mu), len(mu)) or not np.allclose(cov, cov.T) or np.min(np.linalg.eigvalsh(cov)) < 0:
        raise ValueError("symmetric positive semidefinite covariance required")
    integer(count, 2)
    return stream(seed, "physical-uncertainty").multivariate_normal(mu, cov, count)


def route_energy(time_s, speed_m_s, grade, vehicle=None):
    from .core import finite, timebase
    from .vehicle import Vehicle, road_forces

    p = Vehicle() if vehicle is None else vehicle
    t = timebase(time_s)
    v = finite(speed_m_s, real=True, ndim=1)
    g = np.broadcast_to(grade, v.shape)
    if v.shape != t.shape or np.any(v < 0):
        raise ValueError("aligned forward route required")
    mid = (v[1:] + v[:-1]) / 2
    dt = np.diff(t)
    force = road_forces(mid, (g[1:] + g[:-1]) / 2, model=p)["total_n"]
    kinetic = 0.5 * p.effective_mass * np.diff(v * v)
    road = force * mid * dt
    work = kinetic + road
    return {
        "traction_j": float(np.sum(np.maximum(work, 0))),
        "braking_j": float(-np.sum(np.minimum(work, 0))),
        "road_j": float(np.sum(road)),
        "kinetic_change_j": float(np.sum(kinetic)),
    }


def interval(values, coverage=0.9):
    from .core import finite

    a = finite(values, real=True, ndim=1)
    if not 0 < coverage < 1 or len(a) < 2:
        raise ValueError("nonempty ensemble and interior coverage required")
    return np.quantile(a, [(1 - coverage) / 2, 0.5, (1 + coverage) / 2])


def sensitivity(function, parameters, relative_step=0.001):
    from .core import finite, positive

    p = finite(parameters, real=True, ndim=1)
    h = positive(relative_step) * np.maximum(abs(p), 1)
    result = []
    for k in range(len(p)):
        d = np.zeros(len(p))
        d[k] = h[k]
        result.append((function(p + d) - function(p - d)) / (2 * h[k]))
    return np.asarray(result)
