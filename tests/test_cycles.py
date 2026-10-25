import numpy as np
import pytest

from evlab import cycles as m


def test_cycle_rejects_unordered_time_and_negative_speed():
    with pytest.raises(ValueError):
        m.Cycle([0, 1], [-1, 2])
    with pytest.raises(ValueError):
        m.Cycle([0, 0], [1, 2])


def test_synthetic_profiles_are_deterministic_and_hill_grade_is_explicit():
    cycle = m.synthetic("hill")
    assert np.max(abs(cycle.grade)) > 0.06
    assert np.array_equal(m.synthetic().speed_m_s, m.synthetic().speed_m_s)
