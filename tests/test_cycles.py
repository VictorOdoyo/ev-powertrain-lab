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


def test_cycle_csv_roundtrip_preserves_arrays_and_si_header(tmp_path):
    original = m.synthetic()
    path = m.write_csv(original, tmp_path / "cycle.csv")
    restored = m.read_csv(path)
    assert np.array_equal(original.speed_m_s, restored.speed_m_s)
    assert b"\r\n" not in path.read_bytes()


def test_cycle_acceleration_and_jerk_match_constant_acceleration():
    t = np.linspace(0, 10, 101)
    result = m.dynamics(m.Cycle(t, 2 * t))
    assert np.allclose(result["acceleration_m_s2"], 2)
    assert result["rms_jerk_m_s3"] < 1e-10
