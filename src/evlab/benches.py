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
