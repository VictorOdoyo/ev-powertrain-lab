import numpy as np


def motor_point(torque_request_nm, omega_m, dc_voltage_v, temperature_c=25.0, model=None):
    from .inverter import losses
    from .motor import PMSM, at_temperature, current_reference, torque, torque_capability
    from .thermal import derate

    p = at_temperature(temperature_c, PMSM() if model is None else model)
    from dataclasses import replace

    fraction = float(derate(temperature_c))
    if fraction < 1e-6:
        return {
            "torque_nm": 0.0,
            "current_dq_a": np.zeros(2),
            "copper_w": 0.0,
            "inverter_w": 0.0,
            "ac_w": 0.0,
            "dc_w": 0.0,
            "limited": True,
        }
    p = replace(p, current_limit_a=p.current_limit_a * fraction)
    maximum = torque_capability(p)["maximum_nm"]
    command = float(np.clip(torque_request_nm, -maximum, maximum))
    for _ in range(80):
        try:
            current = current_reference(command, omega_m, dc_voltage_v, p)
            break
        except ValueError:
            command *= 0.85
    else:
        return {
            "torque_nm": 0.0,
            "current_dq_a": np.zeros(2),
            "copper_w": 0.0,
            "inverter_w": 0.0,
            "ac_w": 0.0,
            "dc_w": 0.0,
            "limited": True,
        }
    delivered = float(torque(current, p))
    copper = 1.5 * p.resistance_ohm * float(current @ current)
    ac = delivered * omega_m + copper
    electronic = float(losses(current, dc_voltage_v, temperature_c=temperature_c)["total_w"])
    return {
        "torque_nm": delivered,
        "current_dq_a": current,
        "copper_w": copper,
        "inverter_w": electronic,
        "ac_w": ac,
        "dc_w": ac + electronic,
        "limited": bool(abs(delivered - torque_request_nm) > 1e-4),
    }


def operating_point(
    torque_request_nm,
    omega_m,
    battery_state,
    dt_s,
    temperature_battery_c=25.0,
    temperature_motor_c=25.0,
    auxiliary_w=300.0,
    battery_model=None,
    motor_model=None,
):
    from .battery import Pack, current_for_power, current_limits, ocv, terminal_voltage
    from .core import positive

    p = Pack() if battery_model is None else battery_model
    z, vp = battery_state
    limits = current_limits(battery_state, dt_s, temperature_battery_c, p)
    bus = float(ocv(z, temperature_battery_c, p) - vp)
    command = float(torque_request_nm)
    auxiliary = positive(auxiliary_w, allow_zero=True)
    for _ in range(100):
        point = motor_point(command, omega_m, bus, temperature_motor_c, motor_model)
        power = point["dc_w"] + auxiliary
        try:
            current = current_for_power(power, z, vp, temperature_battery_c, p)
        except ValueError:
            command *= 0.8
            continue
        if current > limits["discharge_a"] + 1e-8 or current < -limits["charge_a"] + 1e-8:
            command *= 0.8
            continue
        voltage = float(terminal_voltage(z, vp, current, temperature_battery_c, p))
        if voltage <= 0:
            raise RuntimeError("nonpositive battery bus")
        residual = voltage - bus
        if abs(residual) < 1e-5:
            return {
                **point,
                "battery_current_a": current,
                "battery_power_w": power,
                "bus_voltage_v": voltage,
                "bus_residual_v": residual,
            }
        bus = (bus + voltage) / 2
    raise ValueError("power request cannot satisfy battery current and voltage envelope")
