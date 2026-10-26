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
