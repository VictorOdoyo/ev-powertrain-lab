from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PMSM:
    resistance_ohm: float = 0.04
    ld_h: float = 0.0003
    lq_h: float = 0.0006
    flux_wb: float = 0.08
    pole_pairs: int = 4
    inertia_kg_m2: float = 0.08
    friction_nm_s: float = 0.001
    current_limit_a: float = 250.0

    def __post_init__(self):
        from .core import integer, positive

        integer(self.pole_pairs)
        for k in ("resistance_ohm", "ld_h", "lq_h", "flux_wb", "inertia_kg_m2", "current_limit_a"):
            positive(getattr(self, k), k)
        positive(self.friction_nm_s, allow_zero=True)


def torque(current, model=None):
    from .core import finite

    model = PMSM() if model is None else model
    i = finite(current, real=True)
    if i.shape[-1] != 2:
        raise ValueError("d-q current required")
    return 1.5 * model.pole_pairs * (model.flux_wb + (model.ld_h - model.lq_h) * i[..., 0]) * i[..., 1]


def electrical_rhs(current, voltage, omega_m, model=None):
    from .core import finite

    model = PMSM() if model is None else model
    i, v = finite(current, real=True), finite(voltage, real=True)
    if i.shape != (2,) or v.shape != (2,) or not np.isfinite(omega_m):
        raise ValueError("finite d-q vectors and mechanical speed required")
    we = model.pole_pairs * omega_m
    return np.array(
        [
            (v[0] - model.resistance_ohm * i[0] + we * model.lq_h * i[1]) / model.ld_h,
            (v[1] - model.resistance_ohm * i[1] - we * (model.ld_h * i[0] + model.flux_wb)) / model.lq_h,
        ]
    )


def steady_voltage(current, omega_m, model=None):
    from .core import finite

    p = PMSM() if model is None else model
    i = finite(current, real=True)
    if i.shape[-1] != 2:
        raise ValueError("d-q current required")
    we = p.pole_pairs * np.asarray(omega_m)
    return np.stack(
        (
            p.resistance_ohm * i[..., 0] - we * p.lq_h * i[..., 1],
            p.resistance_ohm * i[..., 1] + we * (p.ld_h * i[..., 0] + p.flux_wb),
        ),
        axis=-1,
    )


def power_balance(current, voltage, omega_m, model=None):
    p = PMSM() if model is None else model
    i = np.asarray(current, float)
    derivative = electrical_rhs(i, voltage, omega_m, p)
    electrical = 1.5 * float(np.dot(current, voltage))
    copper = 1.5 * p.resistance_ohm * float(i @ i)
    storage = 1.5 * (p.ld_h * i[0] * derivative[0] + p.lq_h * i[1] * derivative[1])
    shaft = float(torque(i, p) * omega_m)
    return {
        "electrical_w": electrical,
        "copper_w": copper,
        "storage_rate_w": storage,
        "conversion_w": shaft,
        "defect_w": electrical - copper - storage - shaft,
    }


def mechanical_rhs(omega_m, electromagnetic_nm, load_nm, model=None):
    p = PMSM() if model is None else model
    if not np.all(np.isfinite([omega_m, electromagnetic_nm, load_nm])):
        raise ValueError("finite mechanical inputs required")
    return (electromagnetic_nm - load_nm - p.friction_nm_s * omega_m) / p.inertia_kg_m2


def exact_step(current, voltage, omega_m, dt_s, model=None):
    from scipy.linalg import expm

    from .core import positive

    p = PMSM() if model is None else model
    dt = positive(dt_s)
    we = p.pole_pairs * omega_m
    a = np.array(
        [
            [-p.resistance_ohm / p.ld_h, we * p.lq_h / p.ld_h],
            [-we * p.ld_h / p.lq_h, -p.resistance_ohm / p.lq_h],
        ]
    )
    b = np.array([voltage[0] / p.ld_h, (voltage[1] - we * p.flux_wb) / p.lq_h])
    aug = np.zeros((3, 3))
    aug[:2, :2] = a
    aug[:2, 2] = b
    return (expm(aug * dt) @ np.r_[current, 1])[:2]


def stored_energy(state, model=None):
    from .core import finite

    p = PMSM() if model is None else model
    x = finite(state, real=True)
    if x.shape[-1] < 3:
        raise ValueError("id iq omega state required")
    return 0.75 * (p.ld_h * x[..., 0] ** 2 + p.lq_h * x[..., 1] ** 2) + 0.5 * p.inertia_kg_m2 * x[..., 2] ** 2


def coupled_step(state, voltage, load_nm, dt_s, model=None):
    from .core import finite, positive

    p = PMSM() if model is None else model
    x = finite(state, real=True)
    if x.shape != (4,):
        raise ValueError("id iq omega mechanical-angle state required")
    dt = positive(dt_s)

    def rhs(y):
        electrical = electrical_rhs(y[:2], voltage, y[2], p)
        return np.r_[electrical, mechanical_rhs(y[2], torque(y[:2], p), load_nm, p), y[2]]

    k1 = rhs(x)
    k2 = rhs(x + dt * k1 / 2)
    k3 = rhs(x + dt * k2 / 2)
    k4 = rhs(x + dt * k3)
    result = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    if not np.all(np.isfinite(result)):
        raise RuntimeError("nonfinite motor integration")
    return result


def mtpa(torque_nm, model=None):
    from scipy.optimize import minimize_scalar

    p = PMSM() if model is None else model
    if not np.isfinite(torque_nm):
        raise ValueError("finite torque request required")

    def iq(id_value):
        return torque_nm / (1.5 * p.pole_pairs * (p.flux_wb + (p.ld_h - p.lq_h) * id_value))

    if p.ld_h > p.lq_h:
        raise ValueError("this MTPA branch requires Ld<=Lq")
    fit = minimize_scalar(
        lambda id_value: id_value**2 + iq(id_value) ** 2, bounds=(-p.current_limit_a, 0), method="bounded"
    )
    candidate = np.array([fit.x, iq(fit.x)])
    if np.linalg.norm(candidate) > p.current_limit_a * (1 + 1e-8):
        raise ValueError("requested torque exceeds current-limited MTPA capability")
    return candidate


def torque_capability(model=None):
    from scipy.optimize import minimize_scalar

    p = PMSM() if model is None else model

    def operating(id_value):
        return float(torque([id_value, np.sqrt(max(0, p.current_limit_a**2 - id_value**2))], p))

    fit = minimize_scalar(lambda value: -operating(value), bounds=(-p.current_limit_a, 0), method="bounded")
    return {
        "maximum_nm": operating(fit.x),
        "id_a": float(fit.x),
        "iq_a": float(np.sqrt(p.current_limit_a**2 - fit.x**2)),
    }


def current_reference(torque_nm, omega_m, dc_voltage_v, model=None):
    from scipy.optimize import minimize

    from .core import positive

    p = PMSM() if model is None else model
    limit = positive(dc_voltage_v) / np.sqrt(3)
    guess = mtpa(torque_nm, p)
    constraints = [
        {"type": "eq", "fun": lambda i: float(torque(i, p) - torque_nm)},
        {"type": "ineq", "fun": lambda i: p.current_limit_a**2 - float(i @ i)},
        {
            "type": "ineq",
            "fun": lambda i: limit**2 - float(steady_voltage(i, omega_m, p) @ steady_voltage(i, omega_m, p)),
        },
    ]
    fit = minimize(
        lambda i: float(i @ i),
        guess,
        method="SLSQP",
        bounds=[(-p.current_limit_a, p.current_limit_a)] * 2,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 150},
    )
    i = fit.x
    feasible = (
        abs(torque(i, p) - torque_nm) < 1e-5
        and np.linalg.norm(i) <= p.current_limit_a + 1e-5
        and np.linalg.norm(steady_voltage(i, omega_m, p)) <= limit + 1e-5
    )
    if not feasible:
        raise ValueError("torque request infeasible under current and voltage limits")
    return i


def at_temperature(temperature_c, model=None, copper_alpha=0.00393, magnet_alpha=-0.001):
    from dataclasses import replace

    p = PMSM() if model is None else model
    if not np.isfinite(temperature_c) or temperature_c <= -273.15:
        raise ValueError("physical temperature required")
    return replace(
        p,
        resistance_ohm=p.resistance_ohm * (1 + copper_alpha * (temperature_c - 25)),
        flux_wb=p.flux_wb * (1 + magnet_alpha * (temperature_c - 25)),
    )


def mtpa_stationary(torque_nm, model=None):
    from scipy.optimize import brentq

    p = PMSM() if model is None else model
    if not np.isfinite(torque_nm):
        raise ValueError("finite torque required")
    delta = p.ld_h - p.lq_h
    constant = 1.5 * p.pole_pairs
    if abs(torque_nm) < 1e-14:
        return np.zeros(2)
    if abs(delta) < 1e-14:
        current = np.array([0.0, torque_nm / (constant * p.flux_wb)])
    else:
        target = (torque_nm / constant) ** 2 * delta

        def function(d):
            return d * (p.flux_wb + delta * d) ** 3 - target

        bounds = (-p.current_limit_a, 0) if delta < 0 else (0, p.current_limit_a)
        direct = brentq(function, *bounds)
        current = np.array([direct, torque_nm / (constant * (p.flux_wb + delta * direct))])
    if np.linalg.norm(current) > p.current_limit_a + 1e-8:
        raise ValueError("MTPA current exceeds rating")
    return current
