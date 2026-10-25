import pytest

from evlab import studies_extended as m


def test_impedance_study_identifies_declared_pack():
    r = m.impedance()
    assert r["summary"]["tau_s"] == pytest.approx(30, rel=1e-4)


def test_precharge_study_closes_accumulated_energy():
    r = m.precharge()
    assert abs(r["summary"]["energy_defect_j"]) < 1e-9
