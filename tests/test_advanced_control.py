import numpy as np
import pytest

from evlab import advanced_control as m


def test_deadbeat_current_target_is_exact_when_voltage_is_feasible():
    from evlab.motor import exact_step

    r = m.deadbeat_voltage([0, 50], [1, 51], 100, 0.0001, 400)
    assert not r["saturated"]
    assert np.allclose(exact_step([0, 50], r["applied_v"], 100, 0.0001), [1, 51], atol=1e-9)


def test_priority_voltage_retains_selected_axis_before_secondary():
    a = m.priority_voltage([100, 200], 300)
    assert a[0] == 100 and np.linalg.norm(a) == pytest.approx(300 / np.sqrt(3))


def test_rls_recovers_resistance_and_skips_unexcited_samples():
    r = m.ResistanceRLS(0.1)
    assert not r.update(0, 0)
    for i in np.tile([10, -30, 50, -100], 20):
        r.update(i, 0.064 * i)
    assert r.value == pytest.approx(0.064, rel=1e-4)
    assert r.covariance > 0
