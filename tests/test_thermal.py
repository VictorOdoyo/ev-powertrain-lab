import numpy as np
import pytest

from evlab import thermal as m


def test_lumped_temperature_matches_step_response():
    assert m.lumped_step(25, 100, 25, 1000, 0.1, 100) == pytest.approx(25 + 10 * (1 - np.exp(-1)))
