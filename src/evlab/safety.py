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
