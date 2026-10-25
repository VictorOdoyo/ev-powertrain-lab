from evlab import vehicle as m


def test_reflected_mass_accounts_for_motor_gearing_squared():
    p = m.Vehicle()
    assert p.effective_mass > p.mass_kg
    assert m.Vehicle(motor_inertia=0, wheel_inertia=0).effective_mass == 1800


def test_tailwind_and_downhill_force_signs_are_preserved():
    assert m.road_forces(10, wind_m_s=20)["aero_n"] < 0
    assert m.road_forces(10, grade=-0.1)["grade_n"] < 0
    assert m.road_forces(0)["rolling_n"] == 0
