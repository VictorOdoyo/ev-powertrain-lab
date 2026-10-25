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


def test_resampling_preserves_boundaries_and_linear_ramp():
    cycle = m.Cycle([0, 10], [0, 20])
    fine = m.resample(cycle, 0.1)
    assert fine.time_s[-1] == 10
    assert np.allclose(fine.speed_m_s, 2 * fine.time_s)


def test_route_grade_is_tangent_not_small_angle_height_identity():
    cycle = m.Cycle([0, 10], [10, 10], 0.1)
    result = m.metrics(cycle)
    assert result["distance_m"] == 100
    assert result["elevation_change_m"] == pytest.approx(10 / np.sqrt(1.01))
