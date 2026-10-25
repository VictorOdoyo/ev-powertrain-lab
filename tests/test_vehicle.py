import numpy as np
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


def test_constant_speed_cycle_force_is_road_load():
    t = np.linspace(0, 10, 101)
    v = np.ones(101) * 20
    assert np.allclose(m.required_force(t, v), m.road_forces(20)["total_n"])


def test_friction_brake_carries_full_demand_when_charge_is_blocked():
    result = m.blend_brakes(3000, 0)
    assert result["friction_n"] == 3000 and result["unmet_n"] == 0
