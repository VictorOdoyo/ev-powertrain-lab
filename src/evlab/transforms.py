import numpy as np


def clarke(abc):
    from .core import finite

    value = finite(abc, real=True)
    if value.shape[-1] != 3:
        raise ValueError("three phases required")
    matrix = np.array([[2, -1, -1], [0, np.sqrt(3), -np.sqrt(3)], [1, 1, 1]]) / 3
    return value @ matrix.T
