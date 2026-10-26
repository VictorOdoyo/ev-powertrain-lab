from evlab import studies_cells as m


def test_balanced():
    r = m.balanced()
    assert abs(r["summary"]["max_energy_residual_j"]) < 1e-4
    assert r["summary"]["final_soc_spread"] <= 0.081


def test_unbalanced():
    r = m.unbalanced()
    assert abs(r["summary"]["max_energy_residual_j"]) < 1e-4
    assert r["summary"]["final_soc_spread"] <= 0.081


def test_noisy():
    r = m.noisy()
    assert abs(r["summary"]["max_energy_residual_j"]) < 1e-4
    assert r["summary"]["final_soc_spread"] <= 0.081


def test_weak():
    r = m.weak()
    assert r["summary"]["boundary"]
    assert r["summary"]["elapsed_s"] < 1200
    assert abs(r["summary"]["energy_residual_j"]) < 1e-3


def test_charge():
    r = m.charge()
    assert r["summary"]["boundary"]
    assert r["summary"]["elapsed_s"] < 1200
    assert abs(r["summary"]["energy_residual_j"]) < 1e-3


def test_capacity():
    assert abs(m.capacity()["summary"]["capacity_ah"] - 1.65) < 0.02


def test_pulse():
    assert abs(m.pulse()["summary"]["resistance_ohm"] - 0.035) < 0.001
