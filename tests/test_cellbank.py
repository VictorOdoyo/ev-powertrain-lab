import numpy as np
import pytest

from evlab import cellbank as m


def test_heterogeneous_parameters():
    p = m.Cells()
    assert p.count == 4
    with pytest.raises(ValueError):
        m.Cells(capacity_ah=(2, 3), r0_ohm=(0.02,))
    assert m.validate_state([[0.5] * 4, [0] * 4]).shape == (2, 4)


def test_shunt_kcl_and_power():
    x = np.array([[0.7] * 4, [0.01] * 4])
    r = m.ports(x, 1, np.array([0.2, 0, 0, 0]))
    assert r["cell_current_a"][0] == pytest.approx(1 + 0.2 * r["voltage_v"][0])
    assert r["bleed_w"] == pytest.approx(0.2 * r["voltage_v"][0] ** 2)


def test_energy_gradient_matches_ocv():
    x = np.array([[0.4, 0.5, 0.6, 0.7], [0.01] * 4])
    eps = 1e-6
    plus = x.copy()
    minus = x.copy()
    plus[0, 0] += eps
    minus[0, 0] -= eps
    derivative = (m.stored_energy(plus) - m.stored_energy(minus)) / (2 * eps)
    assert derivative / (2 * 3600) == pytest.approx(3.1 + 0.4 + 0.06 * np.tanh(-1), rel=1e-8)


def test_nonlinear_balance_energy():
    x = np.array([[0.7, 0.6, 0.65, 0.62], [0] * 4])
    r = m.advance(x, 1, 60, [0.2, 0, 0, 0])
    assert abs(r["energy_residual_j"]) < 1e-6
    assert r["state"][0, 0] < 0.7
    assert r["bleed_j"] > 0


def test_balance_hysteresis():
    b = m.Balancer(4, maximum_active=1)
    assert np.array_equal(b.update([4, 3.9, 3.95, 3.9]), [True, False, False, False])
    assert b.update([3.915, 3.9, 3.9, 3.9])[0]
    assert not b.update([3.905, 3.9, 3.9, 3.9]).any()


def test_capacity_identification():
    q = np.linspace(0, 500, 30)
    r = m.capacity_fit(q, -q / (1.8 * 3600))
    assert r["capacity_ah"] == pytest.approx(1.8)
    with pytest.raises(ValueError):
        m.capacity_fit(q, np.zeros(30))


def test_pulse_resistance():
    i = np.array([-2.0, -1, 0, 1, 2])
    r = m.pulse_fit(i, 3.7 - 0.03 * i)
    assert r["resistance_ohm"] == pytest.approx(0.03)
    assert r["ocv_v"] == pytest.approx(3.7)
