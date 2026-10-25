import numpy as np
import pytest

from evlab import electrical_analysis as m


def test_voltage_ellipse_matches_independent_steady_voltage():
    from evlab.motor import steady_voltage

    a = m.voltage_ellipse(500, 400)
    i = np.array([-50, 80])
    d = i - a["center_a"]
    assert d @ a["quadratic"] @ d == pytest.approx(np.sum(steady_voltage(i, 500) ** 2))


def test_pack_impedance_matches_dc_and_high_frequency_limits():
    from evlab.battery import Pack

    p = Pack()
    z = m.pack_impedance([0, 1e9])
    assert z[0] == pytest.approx(p.r0 + p.r1)
    assert z[-1] == pytest.approx(p.r0, abs=1e-10)
