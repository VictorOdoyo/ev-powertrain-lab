import pytest

from evlab import powertrain as m


def test_motor_point_retains_power_balance_and_thermal_shutdown():
    result = m.motor_point(80, 300, 400)
    assert result["dc_w"] == pytest.approx(
        result["torque_nm"] * 300 + result["copper_w"] + result["inverter_w"]
    )
    assert m.motor_point(80, 300, 400, 120)["torque_nm"] == 0
