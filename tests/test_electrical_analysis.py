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


def test_impedance_fit_recovers_independent_synthetic_spectrum():
    f = np.geomspace(0.001, 100, 100)
    z = 0.05 + 0.03 / (1 + 2j * np.pi * f * 25)
    r = m.fit_impedance(f, z)
    assert r["r0_ohm"] == pytest.approx(0.05, rel=1e-5)
    assert r["r1_ohm"] == pytest.approx(0.03, rel=1e-5)
    assert r["tau_s"] == pytest.approx(25, rel=1e-5)
