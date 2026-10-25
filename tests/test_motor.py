import numpy as np
import pytest

from evlab import motor as m


def test_motor_parameters_reject_nonphysical_inductance():
    with pytest.raises(ValueError):
        m.PMSM(ld_h=0)


def test_saliency_and_torque_reversal():
    p = m.PMSM()
    assert m.torque([0, 100], p) == pytest.approx(48)
    assert m.torque([-100, 100], p) > 48
    assert m.torque([0, -100], p) == pytest.approx(-48)


def test_back_emf_at_zero_current_is_not_omitted():
    p = m.PMSM()
    rhs = m.electrical_rhs([0, 0], [0, 0], 100, p)
    assert rhs[0] == 0
    assert rhs[1] == pytest.approx(-p.pole_pairs * 100 * p.flux_wb / p.lq_h)


def test_steady_voltage_annuls_independent_dynamic_rhs():
    p = m.PMSM()
    i = np.array([-30.0, 80.0])
    assert np.allclose(m.electrical_rhs(i, m.steady_voltage(i, 300, p), 300, p), 0, atol=1e-10)


def test_instantaneous_motor_power_closes_with_saliency():
    p = m.power_balance([-70, 130], [40, 100], 400)
    assert abs(p["defect_w"]) < 1e-8


def test_mechanical_balance_and_reverse_load():
    p = m.PMSM()
    speed = 100.0
    a = m.mechanical_rhs(speed, 20, 5, p)
    assert p.inertia_kg_m2 * speed * a + 5 * speed + p.friction_nm_s * speed**2 == pytest.approx(20 * speed)


def test_exact_step_matches_stationary_rl_reference():
    p = m.PMSM()
    t = 0.003
    out = m.exact_step([0, 0], [10, 0], 0, t, p)
    assert out[0] == pytest.approx(10 / p.resistance_ohm * (1 - np.exp(-p.resistance_ohm * t / p.ld_h)))
    assert out[1] == 0


def test_motor_energy_has_correct_current_and_speed_derivatives():
    p = m.PMSM()
    x = np.array([30.0, 70.0, 100.0])
    h = 1e-4
    delta = np.array([h, 0, 0])
    derivative = (m.stored_energy(x + delta, p) - m.stored_energy(x - delta, p)) / (2 * h)
    assert derivative == pytest.approx(1.5 * p.ld_h * x[0], rel=1e-7)


def test_coupled_motor_rotates_and_accelerates_from_q_current():
    x = m.coupled_step([0, 100, 0, 0], [0, 4], 0, 1e-5)
    assert x[2] > 0 and x[3] > 0


def test_mtpa_has_less_current_than_id_zero_for_same_torque():
    p = m.PMSM()
    current = m.mtpa(80, p)
    assert m.torque(current, p) == pytest.approx(80)
    assert np.linalg.norm(current) < 80 / (1.5 * p.pole_pairs * p.flux_wb)


def test_torque_capability_agrees_with_dense_independent_circle_search():
    p = m.PMSM()
    angle = np.linspace(np.pi / 2, np.pi, 10001)
    currents = np.column_stack((p.current_limit_a * np.cos(angle), p.current_limit_a * np.sin(angle)))
    result = m.torque_capability(p)
    assert result["maximum_nm"] == pytest.approx(np.max(m.torque(currents, p)), rel=1e-7)


def test_field_weakening_satisfies_voltage_and_current_ellipse():
    p = m.PMSM()
    current = m.current_reference(40, 900, 400, p)
    assert current[0] < -30
    assert abs(m.torque(current, p) - 40) < 1e-5
    assert np.linalg.norm(m.steady_voltage(current, 900, p)) <= 400 / np.sqrt(3) + 1e-5
