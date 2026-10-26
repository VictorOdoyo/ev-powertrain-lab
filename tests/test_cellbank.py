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
