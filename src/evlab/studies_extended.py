import numpy as np

from .results import curve, make


def output(name, p, s, a, title, x, ys, xlabel, ylabel):
    return make(
        name,
        p,
        s,
        a,
        [curve(title, x, ys, xlabel, ylabel)],
        assumptions=["Standalone bench; not inserted into the main forward vehicle replay."],
    )


def impedance(scale=1, seed=42):
    from .electrical_analysis import fit_impedance, pack_impedance

    f = np.geomspace(0.001, 100, 150 * scale)
    z = pack_impedance(f)
    fit = fit_impedance(f, z)
    a = {"frequency_hz": f, "real_ohm": z.real, "negative_imag_ohm": -z.imag, "impedance_ohm": z}
    return output(
        "pack-impedance",
        {"frequency_min_hz": 0.001, "frequency_max_hz": 100},
        fit,
        a,
        "Thevenin impedance spectrum",
        "frequency_hz",
        [("real_ohm", "Real"), ("negative_imag_ohm", "Negative imaginary")],
        "Frequency (Hz)",
        "Impedance (ohm)",
    )


def precharge(scale=1, seed=42):
    from .safety import precharge as advance

    time = np.linspace(0, 1, 201 * scale)
    link = 0.0
    heat = source = 0.0
    volts = [0.0]
    heats = [0.0]
    sources = [0.0]
    for dt in np.diff(time):
        r = advance(400, link, 100, 0.002, dt)
        link = r["link_v"]
        heat += r["resistor_heat_j"]
        source += r["source_j"]
        volts.append(link)
        heats.append(heat)
        sources.append(source)
    a = {
        "time_s": time,
        "link_v": volts,
        "heat_j": heats,
        "source_j": sources,
        "stored_j": 0.5 * 0.002 * np.array(volts) ** 2,
    }
    return make(
        "dc-precharge",
        {"bus_v": 400, "resistance_ohm": 100, "capacitance_f": 0.002},
        {"final_link_v": link, "energy_defect_j": float(source - heat - a["stored_j"][-1])},
        a,
        [
            curve("DC-link charge", "time_s", [("link_v", "Link")], "Time (s)", "Voltage (V)"),
            curve(
                "Energy accounting",
                "time_s",
                [("source_j", "Source"), ("stored_j", "Capacitor"), ("heat_j", "Resistor")],
                "Time (s)",
                "Energy (J)",
            ),
        ],
    )


def tire_envelope(scale=1, seed=42):
    from .chassis import friction_circle

    lateral = np.linspace(0, 4000, 101 * scale)
    longitudinal = np.array([friction_circle(10000, y, 5000, 0.8) for y in lateral])
    a = {
        "lateral_n": lateral,
        "longitudinal_n": longitudinal,
        "circle_error_n2": longitudinal**2 + lateral**2 - 4000**2,
    }
    return output(
        "tire-envelope",
        {"normal_n": 5000, "mu": 0.8},
        {"maximum_circle_residual_n2": float(max(abs(a["circle_error_n2"])))},
        a,
        "Remaining traction under lateral demand",
        "lateral_n",
        [("longitudinal_n", "Longitudinal capacity")],
        "Lateral force (N)",
        "Longitudinal force (N)",
    )
