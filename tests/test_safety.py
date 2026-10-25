import pytest

from evlab import safety as m


def test_contactor_latches_after_dwell_and_rejects_loaded_reset():
    r = m.Contactor()
    assert r.update(400, 0.005)
    assert not r.update(400, 0.005)
    assert not r.update(0, 1)
    with pytest.raises(ValueError):
        r.reset(100)
    r.reset(0)
    assert r.closed


def test_precharge_energy_closes_and_link_approaches_bus():
    a = m.precharge(400, 0, 100, 0.002, 1)
    assert 390 < a["link_v"] < 400
    assert abs(a["defect_j"]) < 1e-10
