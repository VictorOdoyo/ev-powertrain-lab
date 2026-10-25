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


def synthetic(kind="urban", dt_s=0.2, duration_s=120.0):
    from .core import positive

    dt = positive(dt_s)
    duration = positive(duration_s)
    t = np.arange(round(duration / dt) + 1) * dt
    phase = (t % 40) / 40
    if kind == "urban":
        v = 15 * np.maximum(0, np.sin(2 * np.pi * phase)) ** 2
    elif kind == "highway":
        v = 28 * (1 - np.exp(-t / 10)) * (1 - 0.1 * np.sin(2 * np.pi * t / 60))
    elif kind == "hill":
        v = 20 * (1 - np.exp(-t / 10))
    else:
        raise ValueError("known synthetic cycle required")
    grade = 0.07 * np.sin(2 * np.pi * t / duration) if kind == "hill" else np.zeros(len(t))
    return Cycle(t, v, grade, "synthetic-" + kind)
