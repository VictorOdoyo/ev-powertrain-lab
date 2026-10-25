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
