import numpy as np


def identify_motor(current, derivative, voltage, omega_m, pole_pairs=4):
    from .core import finite, integer

    i, di, v = [finite(x, real=True, ndim=2) for x in (current, derivative, voltage)]
    speed = finite(omega_m, real=True, ndim=1)
    integer(pole_pairs)
    if i.shape != di.shape or i.shape != v.shape or i.shape != (len(speed), 2):
        raise ValueError("aligned d-q records required")
    we = speed * pole_pairs
    d = np.column_stack((i[:, 0], di[:, 0], -we * i[:, 1], np.zeros(len(i))))
    q = np.column_stack((i[:, 1], we * i[:, 0], di[:, 1], we))
    design = np.vstack((d, q))
    target = np.r_[v[:, 0], v[:, 1]]
    scale = np.linalg.norm(design, axis=0)
    if np.any(scale == 0) or np.linalg.matrix_rank(design / scale) < 4:
        raise ValueError("motor parameters are unobservable without independent excitation")
    normalized = design / scale
    coefficient = np.linalg.lstsq(normalized, target, rcond=None)[0] / scale
    if np.any(coefficient <= 0):
        raise ValueError("fit produced nonphysical motor parameters")
    residual = design @ coefficient - target
    return {
        "resistance_ohm": float(coefficient[0]),
        "ld_h": float(coefficient[1]),
        "lq_h": float(coefficient[2]),
        "flux_wb": float(coefficient[3]),
        "rms_residual_v": float(np.sqrt(np.mean(residual**2))),
        "scaled_condition": float(np.linalg.cond(normalized)),
    }


def current_sum(phase_current_a, noise_std_a=0.5, threshold_sigma=5.0):
    from .core import finite, positive

    i = finite(phase_current_a, real=True)
    if i.shape[-1] != 3:
        raise ValueError("three phase sensors required")
    residual = np.sum(i, axis=-1)
    normalized = residual / (np.sqrt(3) * positive(noise_std_a))
    return {
        "residual_a": residual,
        "normalized": normalized,
        "alarm": abs(normalized) > positive(threshold_sigma),
    }


class InnovationMonitor:
    def __init__(self, threshold=9.0, dwell_s=0.5):
        from .core import positive

        self.threshold = positive(threshold)
        self.dwell = positive(dwell_s)
        self.elapsed = 0.0

    def update(self, nis, dt_s, valid=True):
        from .core import positive

        dt = positive(dt_s)
        if not np.isfinite(nis) or nis < 0:
            raise ValueError("nonnegative normalized innovation required")
        self.elapsed = self.elapsed + dt if valid and nis > self.threshold else 0.0
        return self.elapsed >= self.dwell - 1e-12
