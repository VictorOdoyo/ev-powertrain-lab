import numpy as np

from evlab import studies_drive as m


def test_urban_study_reports_achieved_energy_and_masked_startup():
    r = m.urban()
    assert r["summary"]["distance_m"] > 100
    assert r["summary"]["maximum_bus_residual_v"] < 1e-4
    assert not r["arrays"]["valid"][0]


def test_highway_study_has_finite_energy():
    r = m.highway()
    assert r["summary"]["net_battery_wh"] > 0


def test_hill_study_preserves_grade():
    r = m.hill()
    assert np.max(r["arrays"]["grade"]) > 0.06


def test_friction_replay_rejects_motor_regeneration():
    r = m.friction_only()
    assert np.min(r["arrays"]["motor_torque_nm"]) >= -1e-7
    assert np.max(r["arrays"]["friction_n"]) > 100


def test_full_pack_never_overcharges():
    r = m.full_pack()
    assert np.max(r["arrays"]["soc"]) <= 1


def test_hot_pack_blocks_traction():
    r = m.hot_pack()
    assert r["summary"]["distance_m"] == 0
    assert np.all(r["arrays"]["battery_current_a"] == 0)


def test_observer_study_records_sensor_bias():
    r = m.biased()
    assert r["parameters"]["replay"]["current_bias_a"] == 8
