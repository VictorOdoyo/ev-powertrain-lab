import numpy as np
import pytest

from evlab import inverter as m


def test_vector_limit_preserves_direction_and_linear_modulation_bound():
    command = np.array([300.0, 400.0])
    result = m.limit_vector(command, 400)
    assert np.linalg.norm(result) == pytest.approx(0.98 * 400 / np.sqrt(3))
    assert result[0] / result[1] == pytest.approx(0.75)
