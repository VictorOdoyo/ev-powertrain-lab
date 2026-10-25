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


class BatteryEKF:
    def __init__(self, soc=0.6, model=None, voltage_std_v=0.2):
        from .battery import Pack, ocv

        self.model = Pack() if model is None else model
        ocv(soc, model=self.model)
        self.state = np.array([soc, 0.0])
        self.covariance = np.diag([0.02**2, 1.0])
        self.voltage_std = voltage_std_v

    def update(self, voltage_v, current_a, dt_s, temperature_c=25.0):
        prediction = predict(self.state, self.covariance, current_a, dt_s, self.model)
        result = correct(prediction, voltage_v, current_a, temperature_c, self.model, self.voltage_std)
        self.state = result["state"].copy()
        self.covariance = result["covariance"].copy()
        return result


def pulse_resistance(
    before_voltage_v, after_voltage_v, before_current_a, after_current_a, minimum_step_a=5.0
):
    from .core import positive

    values = np.array([before_voltage_v, after_voltage_v, before_current_a, after_current_a])
    if not np.all(np.isfinite(values)):
        raise ValueError("finite pulse samples required")
    change = after_current_a - before_current_a
    if abs(change) < positive(minimum_step_a):
        raise ValueError("pulse current step is unobservable")
    resistance = -(after_voltage_v - before_voltage_v) / change
    if resistance <= 0:
        raise ValueError("pulse edge implies nonphysical resistance")
    return resistance


def fit_relaxation(time, voltage):
    from scipy.optimize import least_squares

    from .core import finite, timebase

    t = timebase(time)
    y = finite(voltage, real=True, ndim=1)
    if y.shape != t.shape or len(t) < 5:
        raise ValueError("sufficient aligned rest samples required")
    elapsed = t - t[0]

    def residual(parameters):
        offset, amplitude, log_tau = parameters
        return offset + amplitude * np.exp(-elapsed / np.exp(log_tau)) - y

    fit = least_squares(residual, [y[-1], y[0] - y[-1], np.log(max(elapsed[-1] / 3, 1e-6))], max_nfev=1000)
    return {
        "offset_v": float(fit.x[0]),
        "amplitude_v": float(fit.x[1]),
        "tau_s": float(np.exp(fit.x[2])),
        "residual_v": float(np.linalg.norm(fit.fun) / np.sqrt(len(t))),
    }


def capacity_from_excursion(time, current_a, soc_start, soc_end):
    from .core import integrate

    difference = soc_start - soc_end
    if not 0 <= soc_start <= 1 or not 0 <= soc_end <= 1 or abs(difference) < 0.05:
        raise ValueError("at least five-percent valid SOC excursion required")
    charge = float(integrate(time, current_a).real)
    capacity = charge / (3600 * difference)
    if capacity <= 0:
        raise ValueError("inconsistent SOC and current signs")
    return capacity
