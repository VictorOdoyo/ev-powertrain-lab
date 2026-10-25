import numpy as np

from evlab import benches as m


def test_current_loop_reaches_target_against_independent_exact_plant():
    a = m.current_loop()
    assert np.linalg.norm(a["current_dq_a"][-1] - [0, 80]) < 0.02


def test_charge_tapers_and_respects_predictive_voltage_ceiling():
    a = m.charge(100, 1, 0.98)
    assert a["soc"][-1] > 0.98
    assert np.max(a["voltage_v"]) <= 96 * 4.2 + 1e-7
    assert np.all(a["current_a"] <= 0)
