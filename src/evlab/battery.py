from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Pack:
    series: int = 96
    parallel: int = 3
    cell_capacity_ah: float = 50.0
    cell_r0_ohm: float = 0.002
    cell_r1_ohm: float = 0.001
    cell_c1_f: float = 30000.0
    minimum_cell_v: float = 2.8
    maximum_cell_v: float = 4.25

    def __post_init__(self):
        from .core import integer, positive

        integer(self.series)
        integer(self.parallel)
        for key in (
            "cell_capacity_ah",
            "cell_r0_ohm",
            "cell_r1_ohm",
            "cell_c1_f",
            "minimum_cell_v",
            "maximum_cell_v",
        ):
            positive(getattr(self, key), key)
        if self.maximum_cell_v <= self.minimum_cell_v:
            raise ValueError("ordered cell voltage limits required")

    @property
    def capacity_c(self):
        return self.parallel * self.cell_capacity_ah * 3600

    @property
    def r0(self):
        return self.cell_r0_ohm * self.series / self.parallel

    @property
    def r1(self):
        return self.cell_r1_ohm * self.series / self.parallel

    @property
    def c1(self):
        return self.cell_c1_f * self.parallel / self.series


def ocv(soc, temperature_c=25.0, model=None):
    from .core import finite

    p = Pack() if model is None else model
    z = finite(soc, real=True)
    t = finite(temperature_c, real=True)
    if np.any((z < 0) | (z > 1)):
        raise ValueError("SOC outside [0,1]")
    return p.series * (3.1 + 1.0 * z + 0.06 * np.tanh(10 * (z - 0.5)) - 0.0002 * (t - 25))


def ocv_jacobian(soc, model=None):
    p = Pack() if model is None else model
    ocv(soc, model=p)
    return p.series * (1 + 0.6 / np.cosh(10 * (np.asarray(soc) - 0.5)) ** 2)


def terminal_voltage(soc, polarization_v, current_a, temperature_c=25.0, model=None):
    from .core import finite

    p = Pack() if model is None else model
    current = finite(current_a, real=True)
    vp = finite(polarization_v, real=True)
    return ocv(soc, temperature_c, p) - vp - p.r0 * current


def step(state, current_a, dt_s, model=None, charge_efficiency=0.995):
    from .core import finite, positive

    p = Pack() if model is None else model
    x = finite(state, real=True)
    if x.shape != (2,) or not np.isfinite(current_a) or not 0 < charge_efficiency <= 1:
        raise ValueError("SOC polarization state finite current and efficiency required")
    dt = positive(dt_s)
    effective = current_a if current_a >= 0 else current_a * charge_efficiency
    soc = x[0] - effective * dt / p.capacity_c
    if not 0 <= soc <= 1:
        raise ValueError("step would cross SOC boundary")
    decay = np.exp(-dt / (p.r1 * p.c1))
    return np.array([soc, decay * x[1] + p.r1 * current_a * (1 - decay)])


def heat(soc, polarization_v, current_a, temperature_c=25.0, model=None):
    p = Pack() if model is None else model
    ocv(soc, temperature_c, p)
    ohmic = p.r0 * current_a**2
    relaxation = polarization_v**2 / p.r1
    entropic = -current_a * (temperature_c + 273.15) * (-0.0002 * p.series)
    return {
        "ohmic_w": ohmic,
        "polarization_w": relaxation,
        "reversible_w": entropic,
        "total_w": ohmic + relaxation + entropic,
    }


def current_for_power(power_w, soc, polarization_v=0.0, temperature_c=25.0, model=None):
    from .core import finite

    p = Pack() if model is None else model
    demand = float(finite(power_w, real=True))
    available = float(ocv(soc, temperature_c, p) - polarization_v)
    discriminant = available**2 - 4 * p.r0 * demand
    if available <= 0 or discriminant < 0:
        raise ValueError("constant power exceeds battery voltage-collapse limit")
    return 2 * demand / (available + np.sqrt(discriminant))


def current_limits(state, dt_s, temperature_c=25.0, model=None, hardware_a=300.0):
    from .core import positive

    p = Pack() if model is None else model
    z, vp = state
    dt = positive(dt_s)
    ceiling = positive(hardware_a)
    available = float(ocv(z, temperature_c, p) - vp)
    discharge = min(ceiling, max(0, (available - p.series * p.minimum_cell_v) / p.r0), z * p.capacity_c / dt)
    charge = min(
        ceiling,
        max(0, (p.series * p.maximum_cell_v - available) / p.r0),
        (1 - z) * p.capacity_c / (dt * 0.995),
    )
    return {"discharge_a": discharge, "charge_a": charge}


def weakest_cell_limits(soc_cells, current_a, model=None):
    from .core import finite

    p = Pack() if model is None else model
    z = finite(soc_cells, real=True, ndim=1)
    if len(z) != p.series:
        raise ValueError("one SOC per series cell required")
    cell_voltage = ocv(z, model=p) / p.series - p.cell_r0_ohm * current_a / p.parallel
    return {
        "minimum_v": float(cell_voltage.min()),
        "maximum_v": float(cell_voltage.max()),
        "voltage_ok": bool(np.all((cell_voltage >= p.minimum_cell_v) & (cell_voltage <= p.maximum_cell_v))),
    }
