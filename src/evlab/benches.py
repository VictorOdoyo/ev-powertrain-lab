import numpy as np

from .core import positive


def current_loop(duration_s=0.08, dt_s=0.00005, speed_rad_s=200, target_a=80, bus_v=400, delay_samples=0):
    from .control import CurrentLoop, Delay
    from .motor import PMSM, exact_step

    positive(duration_s)
    positive(dt_s)
    if dt_s > duration_s / 2:
        raise ValueError("resolved controller timeline required")
    model = PMSM()
    controller = CurrentLoop(model)
    delay = Delay(delay_samples, np.zeros(2))
    time = np.arange(int(duration_s / dt_s) + 1) * dt_s
    current = np.zeros((len(time), 2))
    voltage = np.zeros_like(current)
    reference = np.zeros_like(current)
    for k in range(1, len(time)):
        reference[k, 1] = target_a if time[k] >= 0.005 else 0
        command = controller.update(reference[k], current[k - 1], speed_rad_s, bus_v, dt_s)
        voltage[k] = delay.update(command)
        current[k] = exact_step(current[k - 1], voltage[k], speed_rad_s, dt_s, model)
    return {"time_s": time, "current_dq_a": current, "reference_dq_a": reference, "voltage_dq_v": voltage}


def charge(duration_s=600, dt_s=1, initial_soc=0.9, temperature_c=25):
    from .battery import Pack, heat, step, terminal_voltage
    from .charging import command, grid_power, thermal_current_limit
    from .thermal import lumped_step

    positive(duration_s)
    positive(dt_s)
    p = Pack()
    time = np.arange(int(duration_s / dt_s) + 1) * dt_s
    state = np.array([initial_soc, 0.0])
    temp = float(temperature_c)
    data = {key: np.zeros(len(time)) for key in ("soc", "current_a", "voltage_v", "grid_w", "temperature_c")}
    data["soc"][0] = initial_soc
    data["temperature_c"][0] = temp
    data["voltage_v"][0] = terminal_voltage(state[0], state[1], 0, temp, p)
    for k in range(1, len(time)):
        current = command(state, dt_s, temp, p, maximum_a=thermal_current_limit(temp))
        q = heat(state[0], state[1], current, temp, p)["total_w"]
        state = step(state, current, dt_s, p)
        voltage = terminal_voltage(state[0], state[1], current, temp, p)
        temp = lumped_step(temp, q, 25, 400000, 0.05, dt_s)
        for key, value in zip(
            data, (state[0], current, voltage, grid_power(voltage * current), temp), strict=True
        ):
            data[key][k] = value
    data["time_s"] = time
    return data
