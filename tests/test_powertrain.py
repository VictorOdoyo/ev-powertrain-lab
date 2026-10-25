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
