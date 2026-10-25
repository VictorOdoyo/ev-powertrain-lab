import numpy as np

from .core import positive


class Contactor:
    def __init__(self, trip_a=350, dwell_s=0.01):
        self.trip = positive(trip_a)
        self.dwell = positive(dwell_s)
        self.elapsed = 0.0
        self.closed = True

    def update(self, current_a, dt_s):
        dt = positive(dt_s)
        if not np.isfinite(current_a):
            raise ValueError("finite current required")
        self.elapsed = self.elapsed + dt if abs(current_a) > self.trip else 0.0
        if self.elapsed >= self.dwell:
            self.closed = False
        return self.closed

    def reset(self, current_a):
        if not np.isfinite(current_a) or abs(current_a) > 0.1 * self.trip:
            raise ValueError("zero-current reset required")
        self.elapsed = 0.0
        self.closed = True


def precharge(bus_v, link_v, resistance_ohm, capacitance_f, dt_s):
    r = positive(resistance_ohm)
    c = positive(capacitance_f)
    dt = positive(dt_s)
    if not np.isfinite(bus_v + link_v) or min(bus_v, link_v) < 0:
        raise ValueError("nonnegative finite DC voltages required")
    delta = bus_v - link_v
    decay = np.exp(-dt / (r * c))
    end = bus_v - delta * decay
    heat = 0.5 * c * delta**2 * (1 - decay**2)
    source = c * bus_v * (end - link_v)
    stored = 0.5 * c * (end**2 - link_v**2)
    return {
        "link_v": end,
        "resistor_heat_j": heat,
        "source_j": source,
        "stored_j": stored,
        "defect_j": source - stored - heat,
    }


def dc_link_energy(voltage_before_v, voltage_after_v, capacitance_f):
    c = positive(capacitance_f)
    if not np.isfinite(voltage_before_v + voltage_after_v) or min(voltage_before_v, voltage_after_v) < 0:
        raise ValueError("finite nonnegative voltage endpoints required")
    return 0.5 * c * (voltage_after_v**2 - voltage_before_v**2)
