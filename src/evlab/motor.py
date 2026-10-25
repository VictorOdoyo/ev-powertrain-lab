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
