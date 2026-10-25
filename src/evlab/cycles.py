from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Cycle:
    time_s: object
    speed_m_s: object
    grade: object = 0.0
    label: str = "synthetic"

    def __post_init__(self):
        from .core import finite, timebase

        t = timebase(self.time_s).copy()
        v = finite(self.speed_m_s, real=True, ndim=1).copy()
        g = np.broadcast_to(finite(self.grade, real=True), t.shape).copy()
        if v.shape != t.shape or np.any(v < 0) or np.any(abs(g) > 0.5):
            raise ValueError("nonnegative aligned speed and bounded slope required")
        for key, value in (("time_s", t), ("speed_m_s", v), ("grade", g)):
            value.setflags(write=False)
            object.__setattr__(self, key, value)
