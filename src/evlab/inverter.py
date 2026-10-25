import numpy as np


def limit_vector(voltage, dc_voltage_v, modulation=0.98):
    from .core import finite, positive

    v = finite(voltage, real=True)
    if v.shape[-1] != 2 or not 0 < modulation <= 1:
        raise ValueError("d-q or alpha-beta vector and modulation in (0,1] required")
    limit = positive(dc_voltage_v) * modulation / np.sqrt(3)
    scale = np.minimum(1, limit / np.maximum(np.linalg.norm(v, axis=-1), 1e-30))
    return v * scale[..., None]


def svpwm(alpha_beta, dc_voltage_v):
    from .core import positive
    from .transforms import inverse_clarke

    v = np.asarray(alpha_beta, float)
    if v.shape != (2,):
        raise ValueError("one alpha-beta command required")
    dc = positive(dc_voltage_v)
    phases = inverse_clarke(np.r_[v, 0])
    if np.ptp(phases) > dc * (1 + 1e-12):
        raise ValueError("command outside modulation hexagon")
    offset = 0.5 - (phases.max() + phases.min()) / (2 * dc)
    return phases / dc + offset


def switching(time_s, duty, switch_hz=10000):
    from .core import finite, positive

    t = finite(time_s, real=True)
    d = finite(duty, real=True)
    if d.shape != (3,) or np.any((d < 0) | (d > 1)):
        raise ValueError("three bounded duties required")
    phase = (t * positive(switch_hz)) % 1
    carrier = 1 - abs(2 * phase - 1)
    return carrier[..., None] < d


def deadtime_error(phase_current_a, dc_voltage_v, deadtime_s, switch_hz):
    from .core import finite, positive

    i = finite(phase_current_a, real=True)
    if i.shape[-1] != 3:
        raise ValueError("three phase currents required")
    fraction = positive(deadtime_s, allow_zero=True) * positive(switch_hz)
    if fraction >= 0.5:
        raise ValueError("deadtime exceeds half switching period")
    error = -np.sign(i) * positive(dc_voltage_v) * fraction
    return error - np.mean(error, axis=-1, keepdims=True)


def losses(
    current_dq,
    dc_voltage_v,
    switch_hz=10000,
    temperature_c=60.0,
    on_resistance_ohm=0.005,
    switch_energy_j=0.003,
    reference_current_a=100.0,
):
    from .core import finite, positive

    current = finite(current_dq, real=True)
    peak = np.linalg.norm(current, axis=-1)
    resistance = positive(on_resistance_ohm) * (1 + 0.004 * (temperature_c - 25))
    if resistance <= 0:
        raise ValueError("invalid resistance temperature extrapolation")
    conduction = 1.5 * resistance * peak**2
    switching = (
        6
        * positive(switch_energy_j)
        * positive(switch_hz)
        * (positive(dc_voltage_v) / 400)
        * peak
        / positive(reference_current_a)
    )
    return {"conduction_w": conduction, "switching_w": switching, "total_w": conduction + switching}


def dc_power(ac_power_w, loss_w):
    from .core import finite

    power, loss = np.broadcast_arrays(finite(ac_power_w, real=True), finite(loss_w, real=True))
    if np.any(loss < 0):
        raise ValueError("nonnegative converter loss required")
    return power + loss
