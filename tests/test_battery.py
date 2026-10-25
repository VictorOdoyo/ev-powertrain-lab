import numpy as np
import pytest

from evlab import battery as m


def test_pack_scaling_preserves_polarization_time_constant():
    p = m.Pack()
    assert p.r1 * p.c1 == pytest.approx(p.cell_r1_ohm * p.cell_c1_f)
    assert p.capacity_c == 540000


def test_ocv_is_monotone_and_not_a_manufacturer_calibration():
    z = np.linspace(0, 1, 101)
    assert np.all(np.diff(m.ocv(z)) > 0)
    assert m.ocv(0.5, 35) - m.ocv(0.5, 25) == pytest.approx(-0.192)


def test_ocv_jacobian_matches_central_difference():
    z = 0.45
    h = 1e-6
    assert m.ocv_jacobian(z) == pytest.approx((m.ocv(z + h) - m.ocv(z - h)) / (2 * h), rel=1e-8)


def test_battery_discharge_reduces_and_charge_raises_terminal_voltage():
    assert m.terminal_voltage(0.7, 0, 100) < m.ocv(0.7)
    assert m.terminal_voltage(0.7, 0, -100) > m.ocv(0.7)


def test_held_battery_step_matches_coulomb_and_rc_solutions():
    p = m.Pack()
    x = m.step([0.8, 0], 100, 30, p)
    assert x[0] == pytest.approx(0.8 - 3000 / p.capacity_c)
    assert x[1] == pytest.approx(p.r1 * 100 * (1 - np.exp(-1)))
    with pytest.raises(ValueError):
        m.step([0, 0], 1, 1, p)
