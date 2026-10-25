import numpy as np

from .core import finite, positive


def voltage_ellipse(speed_rad_s, bus_v, model=None):
    from .motor import PMSM

    p = PMSM() if model is None else model
    we = p.pole_pairs * speed_rad_s
    matrix = np.array([[p.resistance_ohm, -we * p.lq_h], [we * p.ld_h, p.resistance_ohm]])
    offset = np.array([0, we * p.flux_wb])
    center = -np.linalg.solve(matrix, offset)
    return {"center_a": center, "quadratic": matrix.T @ matrix, "radius_squared_v2": positive(bus_v) ** 2 / 3}


def pack_impedance(frequency_hz, model=None):
    from .battery import Pack

    p = Pack() if model is None else model
    f = finite(frequency_hz, real=True)
    if np.any(f < 0):
        raise ValueError("nonnegative impedance frequencies required")
    return p.r0 + p.r1 / (1 + 2j * np.pi * f * p.r1 * p.c1)


def fit_impedance(frequency_hz, impedance_ohm):
    from scipy.optimize import least_squares

    f = finite(frequency_hz, real=True, ndim=1)
    z = finite(impedance_ohm, ndim=1)
    if z.shape != f.shape or len(f) < 5 or np.any(f <= 0) or np.ptp(np.log(f)) < 1:
        raise ValueError("broadband positive-frequency impedance record required")

    def residual(logp):
        r0, r1, tau = np.exp(logp)
        error = r0 + r1 / (1 + 2j * np.pi * f * tau) - z
        return np.r_[error.real, error.imag]

    fit = least_squares(residual, np.log([max(min(z.real), 1e-6), max(np.ptp(z.real), 1e-6), 1.0]))
    r0, r1, tau = np.exp(fit.x)
    return {"r0_ohm": r0, "r1_ohm": r1, "tau_s": tau, "rms_ohm": float(np.sqrt(np.mean(fit.fun**2)))}
