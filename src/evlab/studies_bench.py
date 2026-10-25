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
