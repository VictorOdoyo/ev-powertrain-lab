from evlab import vehicle as m


def test_reflected_mass_accounts_for_motor_gearing_squared():
    p = m.Vehicle()
    assert p.effective_mass > p.mass_kg
    assert m.Vehicle(motor_inertia=0, wheel_inertia=0).effective_mass == 1800
