import numpy as np


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


def grid_power(battery_power_w, efficiency=0.94, auxiliary_w=100.0):
    from .core import positive

    if not 0 < efficiency <= 1:
        raise ValueError("charging efficiency in (0,1] required")
    if battery_power_w > 0:
        raise ValueError("charger expects nonpositive battery discharge power")
    return -battery_power_w / efficiency + positive(auxiliary_w, allow_zero=True)


def thermal_current_limit(
    temperature_c, rated_a=200.0, cold_stop_c=0.0, warm_c=15.0, hot_on_c=40.0, hot_stop_c=50.0
):
    from .core import positive

    if not cold_stop_c < warm_c < hot_on_c < hot_stop_c:
        raise ValueError("ordered charge-temperature envelope required")
    if not np.isfinite(temperature_c):
        raise ValueError("finite temperature required")
    cold = np.clip((temperature_c - cold_stop_c) / (warm_c - cold_stop_c), 0, 1)
    hot = np.clip((hot_stop_c - temperature_c) / (hot_stop_c - hot_on_c), 0, 1)
    return float(positive(rated_a) * min(cold, hot))


class Completion:
    def __init__(self, taper_a=10.0, dwell_s=30.0, minimum_soc=0.95):
        from .core import positive

        self.taper = positive(taper_a)
        self.dwell = positive(dwell_s)
        if not 0 < minimum_soc <= 1:
            raise ValueError("SOC completion threshold required")
        self.minimum_soc = minimum_soc
        self.elapsed = 0.0
        self.complete = False

    def update(self, soc, current_a, dt_s):
        from .core import positive

        dt = positive(dt_s)
        if not 0 <= soc <= 1 or not np.isfinite(current_a):
            raise ValueError("valid charge state required")
        qualifying = soc >= self.minimum_soc and abs(current_a) <= self.taper
        self.elapsed = self.elapsed + dt if qualifying else 0.0
        self.complete |= self.elapsed >= self.dwell
        return self.complete
