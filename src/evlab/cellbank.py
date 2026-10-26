from dataclasses import dataclass

import numpy as np

from .core import finite, positive


@dataclass(frozen=True)
class Cells:
    capacity_ah: tuple = (2.0, 2.1, 1.9, 2.0)
    r0_ohm: tuple = (0.02, 0.02, 0.025, 0.02)
    r1_ohm: float = 0.01
    c1_f: float = 3000

    def __post_init__(self):
        q = finite(self.capacity_ah, real=True, ndim=1)
        r = finite(self.r0_ohm, real=True, ndim=1)
        if len(q) < 2 or q.shape != r.shape or np.any(q <= 0) or np.any(r <= 0):
            raise ValueError("aligned positive cell capacities and resistances required")
        positive(self.r1_ohm)
        positive(self.c1_f)
        object.__setattr__(self, "capacity_ah", tuple(q))
        object.__setattr__(self, "r0_ohm", tuple(r))

    @property
    def count(self):
        return len(self.capacity_ah)


def validate_state(value, model=None):
    p = Cells() if model is None else model
    x = finite(value, real=True)
    if x.shape != (2, p.count) or np.any((x[0] < 0) | (x[0] > 1)):
        raise ValueError("SOC and polarization per cell required")
    return x


def ports(value, string_current_a, shunt_s=None, model=None):
    p = Cells() if model is None else model
    x = validate_state(value, p)
    current = float(finite(string_current_a, real=True))
    conductance = np.zeros(p.count) if shunt_s is None else finite(shunt_s, real=True)
    if conductance.shape != (p.count,) or np.any(conductance < 0):
        raise ValueError("nonnegative shunt conductance per cell required")
    ocv = 3.1 + x[0] + 0.06 * np.tanh(10 * (x[0] - 0.5))
    resistance = np.asarray(p.r0_ohm)
    voltage = (ocv - x[1] - resistance * current) / (1 + resistance * conductance)
    cell_current = current + conductance * voltage
    return {
        "voltage_v": voltage,
        "cell_current_a": cell_current,
        "load_w": float(current * sum(voltage)),
        "bleed_w": float(sum(conductance * voltage**2)),
        "ohmic_w": float(sum(resistance * cell_current**2)),
        "relaxation_w": float(sum(x[1] ** 2 / p.r1_ohm)),
    }


def stored_energy(value, model=None):
    p = Cells() if model is None else model
    x = validate_state(value, p)
    z = x[0]
    primitive = 3.1 * z + 0.5 * z * z + 0.006 * (np.log(np.cosh(10 * (z - 0.5))) - np.log(np.cosh(5)))
    return float(np.sum(np.asarray(p.capacity_ah) * 3600 * primitive + 0.5 * p.c1_f * x[1] ** 2))


def advance(value, current_a, duration_s, shunt_s=None, model=None):
    from scipy.integrate import solve_ivp

    p = Cells() if model is None else model
    x = validate_state(value, p)
    duration = positive(duration_s)
    n = p.count
    conductance = np.zeros(n) if shunt_s is None else np.asarray(shunt_s)

    def rhs(_, y):
        cells = y[: 2 * n].reshape(2, n)
        bounded = cells.copy()
        bounded[0] = np.clip(bounded[0], 0, 1)
        r = ports(bounded, current_a, conductance, p)
        ic = r["cell_current_a"]
        return np.r_[
            -ic / (np.asarray(p.capacity_ah) * 3600),
            (-cells[1] + p.r1_ohm * ic) / (p.r1_ohm * p.c1_f),
            r["load_w"],
            r["bleed_w"],
            r["ohmic_w"] + r["relaxation_w"],
        ]

    def lower(_, y):
        return min(y[:n])

    def upper(_, y):
        return 1 - max(y[:n])

    lower.terminal = True
    lower.direction = -1
    upper.terminal = True
    upper.direction = -1
    solved = solve_ivp(
        rhs,
        (0, duration),
        np.r_[x.ravel(), np.zeros(3)],
        events=(lower, upper),
        rtol=1e-10,
        atol=1e-11,
        max_step=min(duration, 30),
    )
    if not solved.success:
        raise RuntimeError(solved.message)
    final = solved.y[: 2 * n, -1].reshape(2, n)
    final[0] = np.clip(final[0], 0, 1)
    energies = solved.y[-3:, -1]
    residual = stored_energy(final, p) - stored_energy(x, p) + sum(energies)
    return {
        "state": final,
        "elapsed_s": float(solved.t[-1]),
        "boundary": bool(solved.status == 1),
        "load_j": float(energies[0]),
        "bleed_j": float(energies[1]),
        "dissipation_j": float(energies[2]),
        "energy_residual_j": float(residual),
    }
