import pytest

from evlab import battery as m


def test_pack_scaling_preserves_polarization_time_constant():
    p = m.Pack()
    assert p.r1 * p.c1 == pytest.approx(p.cell_r1_ohm * p.cell_c1_f)
    assert p.capacity_c == 540000
