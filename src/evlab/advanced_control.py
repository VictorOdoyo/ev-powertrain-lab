import numpy as np

from .core import finite, positive


def deadbeat_voltage(current, target, speed_rad_s, dt_s, bus_v, model=None):
    from .inverter import limit_vector
    from .motor import PMSM, exact_step

    p = PMSM() if model is None else model
    dt = positive(dt_s)
    x = finite(current, real=True)
    r = finite(target, real=True)
    if x.shape != (2,) or r.shape != (2,):
        raise ValueError("two-axis currents required")
    base = exact_step(x, [0, 0], speed_rad_s, dt, p)
    columns = np.column_stack([exact_step(x, v, speed_rad_s, dt, p) - base for v in ([1, 0], [0, 1])])
    raw = np.linalg.solve(columns, r - base)
    applied = limit_vector(raw, bus_v)
    return {"raw_v": raw, "applied_v": applied, "saturated": bool(np.linalg.norm(raw - applied) > 1e-8)}
