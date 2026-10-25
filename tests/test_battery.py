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
