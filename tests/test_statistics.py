import numpy as np
import pytest

from evlab import statistics as m


def test_correlated_samples_preserve_covariance_and_repeat_seed():
    a = m.correlated_samples([0, 0], [[1, 0.8], [0.8, 1]], 10000)
    assert np.corrcoef(a.T)[0, 1] == pytest.approx(0.8, abs=0.02)
    assert np.array_equal(a, m.correlated_samples([0, 0], [[1, 0.8], [0.8, 1]], 10000))


def test_route_work_closes_cycle_energy_balance():
    t = np.linspace(0, 40, 201)
    v = 15 * np.sin(np.pi * t / 40) ** 2
    a = m.route_energy(t, v, 0)
    assert a["traction_j"] - a["braking_j"] == pytest.approx(a["road_j"] + a["kinetic_change_j"])


def test_sensitivity_matches_independent_quadratic_derivative():
    a = m.sensitivity(lambda p: p[0] ** 2 + 3 * p[1], [2.0, 4.0])
    assert np.allclose(a, [4, 3])
    assert np.allclose(m.interval(np.arange(101)), [5, 50, 95])
