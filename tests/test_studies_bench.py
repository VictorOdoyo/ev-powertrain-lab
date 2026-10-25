import numpy as np
import pytest

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


def test_field_weakening_study_marks_infeasible_speed():
    r = m.field_weakening()
    a = r["arrays"]
    assert a["valid"][0] and not a["valid"][-1]
    assert max(a["voltage_v"]) <= 350 / np.sqrt(3) + 1e-5


def test_mtpa_study_compares_independent_solvers():
    r = m.mtpa_curve()
    assert r["summary"]["maximum_independent_error_a"] < 1e-4


def test_pulse_study_relaxes_after_current_removal():
    r = m.battery_pulse()
    a = r["arrays"]
    assert a["polarization_v"][-1] < max(a["polarization_v"])
    assert a["soc"][-1] == pytest.approx(0.7 - 3000 / 540000)


def test_thermal_study_conserves_heat():
    r = m.thermal_network()
    assert r["summary"]["maximum_energy_drift_j"] < 1e-5


def test_modulation_study_matches_inverse_transform():
    r = m.modulation()
    assert r["summary"]["maximum_average_error_v"] < 1e-10
