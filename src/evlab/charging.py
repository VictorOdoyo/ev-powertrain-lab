def command(state, dt_s, temperature_c=25.0, model=None, maximum_a=200.0, target_cell_v=4.2):
    from scipy.optimize import brentq

    from .battery import Pack, step, terminal_voltage
    from .core import positive

    p = Pack() if model is None else model
    dt = positive(dt_s)
    maximum = positive(maximum_a)
    if not p.minimum_cell_v < target_cell_v <= p.maximum_cell_v:
        raise ValueError("charging voltage inside pack limits required")
    z, _vp = state
    headroom = max(0, (1 - z) * p.capacity_c / (dt * 0.995))
    upper = min(maximum, headroom)

    def future_voltage(charge):
        future = step(state, -charge, dt, p)
        return float(terminal_voltage(future[0], future[1], -charge, temperature_c, p))

    target = p.series * target_cell_v
    if upper == 0 or future_voltage(0) >= target:
        return 0.0
    if future_voltage(upper) <= target:
        return -upper
    return -float(brentq(lambda value: future_voltage(value) - target, 0, upper))
