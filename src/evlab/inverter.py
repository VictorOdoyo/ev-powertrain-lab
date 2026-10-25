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
