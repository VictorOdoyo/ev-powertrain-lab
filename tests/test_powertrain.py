import pytest

from evlab import powertrain as m


def test_motor_point_retains_power_balance_and_thermal_shutdown():
    result = m.motor_point(80, 300, 400)
    assert result["dc_w"] == pytest.approx(
        result["torque_nm"] * 300 + result["copper_w"] + result["inverter_w"]
    )
    assert m.motor_point(80, 300, 400, 120)["torque_nm"] == 0


def test_powertrain_bus_solution_satisfies_terminal_power_and_blocks_full_soc_charge():

    result = m.operating_point(80, 300, [0.8, 0], 0.2)
    assert result["battery_current_a"] * result["bus_voltage_v"] == pytest.approx(result["battery_power_w"])
    full = m.operating_point(-50, 300, [1, 0], 0.2)
    assert full["battery_current_a"] >= -1e-8


def test_vehicle_step_achieves_less_force_when_drive_torque_is_infeasible():
    result = m.vehicle_step(0, 20000, [0.8, 0], 0.1)
    assert 0 < result["wheel_force_n"] < 20000
    assert result["speed_m_s"] > 0
    assert abs(result["mechanical_defect_j"]) < 1
