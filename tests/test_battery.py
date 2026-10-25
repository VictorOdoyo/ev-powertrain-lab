import numpy as np
import pytest

from evlab import battery as m


def test_pack_scaling_preserves_polarization_time_constant():
    p = m.Pack()
    assert p.r1 * p.c1 == pytest.approx(p.cell_r1_ohm * p.cell_c1_f)
    assert p.capacity_c == 540000


def test_ocv_is_monotone_and_not_a_manufacturer_calibration():
    z = np.linspace(0, 1, 101)
    assert np.all(np.diff(m.ocv(z)) > 0)
    assert m.ocv(0.5, 35) - m.ocv(0.5, 25) == pytest.approx(-0.192)


def test_ocv_jacobian_matches_central_difference():
    z = 0.45
    h = 1e-6
    assert m.ocv_jacobian(z) == pytest.approx((m.ocv(z + h) - m.ocv(z - h)) / (2 * h), rel=1e-8)


def test_battery_discharge_reduces_and_charge_raises_terminal_voltage():
    assert m.terminal_voltage(0.7, 0, 100) < m.ocv(0.7)
    assert m.terminal_voltage(0.7, 0, -100) > m.ocv(0.7)


def test_held_battery_step_matches_coulomb_and_rc_solutions():
    p = m.Pack()
    x = m.step([0.8, 0], 100, 30, p)
    assert x[0] == pytest.approx(0.8 - 3000 / p.capacity_c)
    assert x[1] == pytest.approx(p.r1 * 100 * (1 - np.exp(-1)))
    with pytest.raises(ValueError):
        m.step([0, 0], 1, 1, p)


def test_battery_resistive_heat_is_nonnegative_and_reversible_term_changes_sign():
    a = m.heat(0.5, 1, 10)
    b = m.heat(0.5, 1, -10)
    assert a["ohmic_w"] == b["ohmic_w"]
    assert a["reversible_w"] == -b["reversible_w"]


def test_power_inversion_is_stable_near_zero_and_supports_regeneration():
    for power in (-20000, 1e-8, 20000):
        i = m.current_for_power(power, 0.8)
        assert i * m.terminal_voltage(0.8, 0, i) == pytest.approx(power, rel=1e-10)
    with pytest.raises(ValueError):
        m.current_for_power(1e8, 0.5)


def test_current_limits_remove_regen_at_full_soc():
    limits = m.current_limits([1, 0], 1)
    assert limits["charge_a"] == 0
    assert limits["discharge_a"] > 0


def test_weakest_cell_is_not_masked_by_pack_average():
    p = m.Pack()
    z = np.full(p.series, 0.8)
    z[0] = 0
    result = m.weakest_cell_limits(z, 600, p)
    assert not result["voltage_ok"]


def test_energy_integral_has_ocv_coulomb_derivative():
    p = m.Pack()
    h = 1e-5
    derivative = (m.available_energy_j(0.2, 0.7 + h, p) - m.available_energy_j(0.2, 0.7 - h, p)) / (2 * h)
    assert derivative == pytest.approx(p.capacity_c * m.ocv(0.7, model=p), rel=1e-8)


def test_aging_model_counts_equal_charge_and_discharge_throughput():
    assert m.aging_increment(100, 25, 3600)["equivalent_cycles"] == pytest.approx(1 / 3)
    assert m.aging_increment(-100, 25, 3600) == m.aging_increment(100, 25, 3600)


def test_charge_inefficiency_is_zero_on_discharge_and_closes_coulomb_loss_power():
    assert m.charge_inefficiency_heat(100, 0.5) == 0
    assert m.charge_inefficiency_heat(-100, 0.5) == pytest.approx(0.5 * m.ocv(0.5))


@pytest.mark.parametrize("temperature", [-273.15, -300.0])
def test_battery_thermodynamics_reject_absolute_zero(temperature):
    with pytest.raises(ValueError):
        m.ocv(0.5, temperature)
    with pytest.raises(ValueError):
        m.aging_increment(1, temperature, 1)
