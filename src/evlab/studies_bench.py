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
