import numpy as np

from evlab import studies_bench as m


def test_foc_study_uses_resolved_plant_channels():
    r = m.foc()
    assert r["summary"]["final_error_a"] < 0.02


def test_delay_study_propagates_delayed_commands():
    r = m.foc_delay()
    assert np.max(abs(r["arrays"]["ideal_a"] - r["arrays"]["delayed_a"])) > 0.1


def test_charging_study_preserves_voltage_ceiling():
    r = m.charging()
    assert max(r["arrays"]["voltage_v"]) <= 403.200001
