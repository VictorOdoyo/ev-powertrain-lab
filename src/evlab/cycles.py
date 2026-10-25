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


def resample(cycle, dt_s):
    from .core import positive

    count = max(2, round((cycle.time_s[-1] - cycle.time_s[0]) / positive(dt_s)) + 1)
    t = np.linspace(cycle.time_s[0], cycle.time_s[-1], count)
    return Cycle(
        t, np.interp(t, cycle.time_s, cycle.speed_m_s), np.interp(t, cycle.time_s, cycle.grade), cycle.label
    )


def metrics(cycle):
    from .core import integrate

    distance = float(integrate(cycle.time_s, cycle.speed_m_s).real)
    vertical_speed = cycle.speed_m_s * cycle.grade / np.sqrt(1 + cycle.grade**2)
    return {
        "distance_m": distance,
        "elevation_change_m": float(integrate(cycle.time_s, vertical_speed).real),
        "duration_s": float(cycle.time_s[-1] - cycle.time_s[0]),
        "maximum_speed_m_s": float(cycle.speed_m_s.max()),
        "stop_sample_fraction": float(np.mean(cycle.speed_m_s < 0.1)),
    }


def write_csv(cycle, path):
    import csv
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["time_s", "speed_m_s", "grade"])
        writer.writerows(zip(cycle.time_s, cycle.speed_m_s, cycle.grade, strict=True))
    return path


def read_csv(path, label="imported"):
    import csv
    from pathlib import Path

    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["time_s", "speed_m_s", "grade"]:
            raise ValueError("exact SI cycle schema required")
        rows = [[float(row[k]) for k in reader.fieldnames] for row in reader]
    array = np.asarray(rows)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("nonempty cycle required")
    return Cycle(array[:, 0], array[:, 1], array[:, 2], label)
