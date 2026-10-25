import numpy as np

from evlab import advanced_control as m


def test_deadbeat_current_target_is_exact_when_voltage_is_feasible():
    from evlab.motor import exact_step

    r = m.deadbeat_voltage([0, 50], [1, 51], 100, 0.0001, 400)
    assert not r["saturated"]
    assert np.allclose(exact_step([0, 50], r["applied_v"], 100, 0.0001), [1, 51], atol=1e-9)
