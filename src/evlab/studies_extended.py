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
