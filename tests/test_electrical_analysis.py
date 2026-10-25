import numpy as np
import pytest

from evlab import electrical_analysis as m


def test_voltage_ellipse_matches_independent_steady_voltage():
    from evlab.motor import steady_voltage

    a = m.voltage_ellipse(500, 400)
    i = np.array([-50, 80])
    d = i - a["center_a"]
    assert d @ a["quadratic"] @ d == pytest.approx(np.sum(steady_voltage(i, 500) ** 2))
