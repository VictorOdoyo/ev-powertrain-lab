import numpy as np


def clarke(abc):
    from .core import finite

    value = finite(abc, real=True)
    if value.shape[-1] != 3:
        raise ValueError("three phases required")
    matrix = np.array([[2, -1, -1], [0, np.sqrt(3), -np.sqrt(3)], [1, 1, 1]]) / 3
    return value @ matrix.T


def inverse_clarke(ab0):
    from .core import finite

    v = finite(ab0, real=True)
    if v.shape[-1] != 3:
        raise ValueError("alpha beta zero coordinates required")
    return v @ np.array([[1, 0, 1], [-0.5, np.sqrt(3) / 2, 1], [-0.5, -np.sqrt(3) / 2, 1]]).T


def park(ab, theta):
    from .core import finite

    v = finite(ab, real=True)
    angle = finite(theta, real=True)
    if v.shape[-1] != 2:
        raise ValueError("two stationary coordinates required")
    c, s = np.cos(angle), np.sin(angle)
    return np.stack((c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1]), axis=-1)


def inverse_park(dq, theta):
    return park(dq, -np.asarray(theta))


def dq_power(voltage, current):
    from .core import finite

    v, i = np.broadcast_arrays(finite(voltage, real=True), finite(current, real=True))
    if v.shape[-1] != 2:
        raise ValueError("d-q vectors required")
    return 1.5 * np.sum(v * i, axis=-1)


def wrap(angle):
    from .core import finite

    return (finite(angle, real=True) + np.pi) % (2 * np.pi) - np.pi
