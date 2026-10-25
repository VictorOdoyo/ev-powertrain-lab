import pytest

from evlab import vehicle as m


def test_reflected_mass_accounts_for_motor_gearing_squared():
    p = m.Vehicle()
    assert p.effective_mass > p.mass_kg
    assert m.Vehicle(motor_inertia=0, wheel_inertia=0).effective_mass == 1800


def test_tailwind_and_downhill_force_signs_are_preserved():
    assert m.road_forces(10, wind_m_s=20)["aero_n"] < 0
    assert m.road_forces(10, grade=-0.1)["grade_n"] < 0
    assert m.road_forces(0)["rolling_n"] == 0


def test_regen_gear_loss_reduces_recovered_shaft_power():
    p = m.Vehicle()
    force = -1000.0
    speed = 20.0
    shaft = m.motor_torque(force, speed, p) * speed * p.gear / p.wheel_radius_m
    assert shaft == pytest.approx(force * speed * p.gear_efficiency)


def test_bidirectional_driveline_roundtrip():
    for force in (-1000, 0, 1000):
        assert m.wheel_force(m.motor_torque(force, 20), 20) == pytest.approx(force)
