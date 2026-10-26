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
