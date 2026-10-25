from evlab import studies_bench as m


def test_foc_study_uses_resolved_plant_channels():
    r = m.foc()
    assert r["summary"]["final_error_a"] < 0.02
