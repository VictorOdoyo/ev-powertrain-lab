import numpy as np

from .results import curve, make


def lines(name, parameters, summary, arrays, specs, assumptions=()):
    return make(
        name,
        parameters,
        summary,
        arrays,
        [
            curve(title, x, [(key, label) for key, label in series], xlabel, ylabel)
            for title, x, series, xlabel, ylabel in specs
        ],
        assumptions=assumptions,
    )


def foc(scale=1, seed=42):
    from .benches import current_loop

    a = current_loop(dt_s=0.00005 / scale)
    for key in ("current_dq_a", "reference_dq_a", "voltage_dq_v"):
        a[key + "_d"] = a[key][:, 0]
        a[key + "_q"] = a[key][:, 1]
    return lines(
        "foc-step",
        {"dt_s": 0.00005 / scale, "speed_rad_s": 200, "target_a": 80},
        {"final_error_a": float(np.linalg.norm(a["current_dq_a"][-1] - a["reference_dq_a"][-1]))},
        a,
        [
            (
                "q-axis tracking",
                "time_s",
                [("current_dq_a_q", "Plant"), ("reference_dq_a_q", "Command")],
                "Time (s)",
                "Peak current (A)",
            ),
            ("d-axis decoupling", "time_s", [("current_dq_a_d", "Plant")], "Time (s)", "Peak current (A)"),
            (
                "Applied voltage",
                "time_s",
                [("voltage_dq_v_d", "d"), ("voltage_dq_v_q", "q")],
                "Time (s)",
                "Peak voltage (V)",
            ),
        ],
    )


def foc_delay(scale=1, seed=42):
    from .benches import current_loop

    a = current_loop(dt_s=0.00005 / scale)
    b = current_loop(dt_s=0.00005 / scale, delay_samples=2 * scale)
    data = {"time_s": a["time_s"], "ideal_a": a["current_dq_a"][:, 1], "delayed_a": b["current_dq_a"][:, 1]}
    return lines(
        "foc-delay",
        {"transport_s": 0.0001},
        {"delayed_final_error_a": float(abs(data["delayed_a"][-1] - 80))},
        data,
        [
            (
                "Command transport delay",
                "time_s",
                [("ideal_a", "No delay"), ("delayed_a", "100 us delay")],
                "Time (s)",
                "q current (A)",
            )
        ],
    )


def charging(scale=1, seed=42):
    from .benches import charge

    a = charge(600, 1 / scale, 0.9)
    return lines(
        "cccv-charge",
        {"duration_s": 600, "initial_soc": 0.9, "grid_efficiency": 0.94},
        {"final_soc": float(a["soc"][-1]), "grid_wh": float(np.trapezoid(a["grid_w"], a["time_s"]) / 3600)},
        a,
        [
            ("Charge trajectory", "time_s", [("soc", "SOC")], "Time (s)", "SOC"),
            (
                "Charge taper",
                "time_s",
                [("current_a", "Positive means discharge")],
                "Time (s)",
                "Current (A)",
            ),
            ("Voltage ceiling", "time_s", [("voltage_v", "Pack voltage")], "Time (s)", "Voltage (V)"),
            ("Grid demand", "time_s", [("grid_w", "AC input incl auxiliary")], "Time (s)", "Power (W)"),
        ],
        ["Illustrative OCV and CC-CV setpoints; no manufacturer certification."],
    )


def field_weakening(scale=1, seed=42):
    from .motor import PMSM, current_reference, steady_voltage

    p = PMSM()
    speed = np.linspace(0, 1800, 101 * scale)
    currents = np.zeros((len(speed), 2))
    valid = np.zeros(len(speed), bool)
    voltage = np.zeros(len(speed))
    for k, w in enumerate(speed):
        try:
            currents[k] = current_reference(60, w, 350, p)
            voltage[k] = np.linalg.norm(steady_voltage(currents[k], w, p))
            valid[k] = True
        except ValueError:
            pass
    a = {
        "speed_rad_s": speed,
        "id_a": currents[:, 0],
        "iq_a": currents[:, 1],
        "current_a": np.linalg.norm(currents, axis=1),
        "voltage_v": voltage,
        "valid": valid,
    }
    plots = [
        curve(
            "Voltage-constrained current reference",
            "speed_rad_s",
            [("id_a", "d"), ("iq_a", "q")],
            "Speed (rad/s)",
            "Peak current (A)",
            valid="valid",
        ),
        curve(
            "Feasibility envelope",
            "speed_rad_s",
            [("voltage_v", "Voltage magnitude")],
            "Speed (rad/s)",
            "Peak voltage (V)",
            valid="valid",
        ),
    ]
    return make(
        "field-weakening",
        {"torque_nm": 60, "bus_v": 350},
        {"feasible_fraction": float(np.mean(valid))},
        a,
        plots,
        assumptions=["Unattainable speeds masked rather than plotted as zero current."],
    )


def mtpa_curve(scale=1, seed=42):
    from .motor import PMSM, mtpa, mtpa_stationary

    p = PMSM()
    t = np.linspace(-140, 140, 81 * scale)
    i = np.array([mtpa_stationary(x, p) for x in t])
    ref = np.array([mtpa(x, p) for x in t])
    a = {"torque_nm": t, "id_a": i[:, 0], "iq_a": i[:, 1], "error_a": np.linalg.norm(i - ref, axis=1)}
    return lines(
        "mtpa-reference",
        {"current_limit_a": p.current_limit_a},
        {"maximum_independent_error_a": float(max(a["error_a"]))},
        a,
        [
            (
                "Minimum-current locus",
                "torque_nm",
                [("id_a", "d"), ("iq_a", "q")],
                "Torque (Nm)",
                "Peak current (A)",
            ),
            (
                "Independent optimizer residual",
                "torque_nm",
                [("error_a", "Current vector error")],
                "Torque (Nm)",
                "Error (A)",
            ),
        ],
    )


def battery_pulse(scale=1, seed=42):
    from .battery import Pack, step, terminal_voltage

    p = Pack()
    dt = 0.2 / scale
    t = np.arange(0, 90 + dt / 2, dt)
    state = np.array([0.7, 0.0])
    a = {
        "time_s": t,
        "current_a": np.where((t > 10) & (t <= 40), 100.0, 0.0),
        "soc": np.zeros(len(t)),
        "polarization_v": np.zeros(len(t)),
        "voltage_v": np.zeros(len(t)),
    }
    for k in range(len(t)):
        if k:
            state = step(state, a["current_a"][k], dt, p)
        a["soc"][k], a["polarization_v"][k] = state
        a["voltage_v"][k] = terminal_voltage(*state, a["current_a"][k], model=p)
    return lines(
        "battery-pulse",
        {"r0_ohm": p.r0, "r1_ohm": p.r1, "tau_s": p.r1 * p.c1},
        {"final_soc": float(state[0])},
        a,
        [
            ("Applied load", "time_s", [("current_a", "Pack current")], "Time (s)", "Current (A)"),
            (
                "Polarization and recovery",
                "time_s",
                [("polarization_v", "RC state")],
                "Time (s)",
                "Voltage (V)",
            ),
            ("Terminal response", "time_s", [("voltage_v", "Terminal")], "Time (s)", "Voltage (V)"),
        ],
    )


def thermal_network(scale=1, seed=42):
    from .thermal import network, network_step

    cap = np.array([1000.0, 3000.0, 8000.0])
    mat = network(cap, [(0, 1, 20), (1, 2, 10)], np.zeros(3))
    t = np.linspace(0, 200, 101 * scale)
    temp = np.array([100.0, 25.0, 25.0])
    hist = [temp.copy()]
    for dt in np.diff(t):
        temp = network_step(temp, np.zeros(3), 25, dt, mat)
        hist.append(temp.copy())
    hist = np.array(hist)
    a = {
        "time_s": t,
        "motor_c": hist[:, 0],
        "coolant_c": hist[:, 1],
        "sink_c": hist[:, 2],
        "energy_j": hist @ cap,
    }
    return lines(
        "thermal-network",
        {"capacities_j_k": cap.tolist(), "edges_w_k": [20, 10]},
        {"maximum_energy_drift_j": float(np.max(abs(a["energy_j"] - a["energy_j"][0])))},
        a,
        [
            (
                "Isolated heat redistribution",
                "time_s",
                [("motor_c", "Motor"), ("coolant_c", "Coolant"), ("sink_c", "Sink")],
                "Time (s)",
                "Temperature (deg C)",
            ),
            (
                "Conservation audit",
                "time_s",
                [("energy_j", "Sum C T")],
                "Time (s)",
                "Stored thermal energy (J)",
            ),
        ],
    )


def modulation(scale=1, seed=42):
    from .inverter import svpwm
    from .transforms import inverse_clarke

    angle = np.linspace(0, 2 * np.pi, 121 * scale)
    target = np.column_stack([150 * np.cos(angle), 150 * np.sin(angle)])
    duty = np.array([svpwm(v, 400) for v in target])
    phase = (duty - duty.mean(axis=1)[:, None]) * 400
    ref = inverse_clarke(np.column_stack([target, np.zeros(len(angle))]))
    a = {
        "angle_rad": angle,
        "duty_a": duty[:, 0],
        "duty_b": duty[:, 1],
        "duty_c": duty[:, 2],
        "error_v": np.linalg.norm(phase - ref, axis=1),
    }
    return lines(
        "svpwm-average",
        {"bus_v": 400, "vector_peak_v": 150},
        {"maximum_average_error_v": float(max(a["error_v"]))},
        a,
        [
            (
                "Three-leg modulation",
                "angle_rad",
                [("duty_a", "A"), ("duty_b", "B"), ("duty_c", "C")],
                "Electrical angle (rad)",
                "Duty ratio",
            ),
            (
                "Analytic averaged voltage closure",
                "angle_rad",
                [("error_v", "Phase-vector norm")],
                "Electrical angle (rad)",
                "Residual (V)",
            ),
        ],
    )


def energy_uncertainty(scale=1, seed=42):
    from dataclasses import replace

    from .statistics import correlated_samples, interval, route_energy
    from .vehicle import Vehicle

    samples = correlated_samples([1800, 0.62, 0.01], np.diag([100**2, 0.05**2, 0.001**2]), 300 * scale, seed)
    time = np.linspace(0, 60, 301)
    speed = 20 * np.sin(np.pi * time / 60) ** 2
    wh = np.array(
        [
            route_energy(time, speed, 0, replace(Vehicle(), mass_kg=x[0], drag_area_m2=x[1], rolling=x[2]))[
                "traction_j"
            ]
            / 3600
            for x in samples
        ]
    )
    a = {
        "sample": np.arange(len(wh)),
        "traction_wh": wh,
        "mass_kg": samples[:, 0],
        "drag_area_m2": samples[:, 1],
    }
    return lines(
        "road-load-uncertainty",
        {"seed": seed, "samples": len(wh), "covariance": np.diag([100**2, 0.05**2, 0.001**2]).tolist()},
        {"traction_interval_wh": interval(wh).tolist()},
        a,
        [
            (
                "Conditional road-load ensemble",
                "sample",
                [("traction_wh", "Positive wheel work")],
                "Sample",
                "Energy (Wh)",
            )
        ],
        ["Backward prescribed-speed work envelope, not achieved drive or battery range prediction."],
    )
