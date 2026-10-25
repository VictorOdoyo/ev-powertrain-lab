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
