import numpy as np

from .results import curve, make


def balance_run(name, scale, seed, enabled=True, noise_std_v=0):
    from .cellbank import Balancer, Cells, advance, ports
    from .core import stream

    p = Cells()
    x = np.array([[0.72, 0.64, 0.66, 0.65], [0.0] * 4])
    t = np.linspace(0, 1800, 61 * scale)
    states = [x.copy()]
    ledger = [0.0]
    bleed = [0.0]
    b = Balancer(4)
    rng = stream(seed, "balance-voltage")
    cumulative = 0.0
    cumulative_bleed = 0.0
    for dt in np.diff(t):
        measured = ports(x, 0, model=p)["voltage_v"] + rng.normal(0, noise_std_v, 4)
        shunt = b.update(measured) * 0.2 if enabled else np.zeros(4)
        r = advance(x, 0, dt, shunt, p)
        x = r["state"]
        states.append(x.copy())
        cumulative += r["energy_residual_j"]
        cumulative_bleed += r["bleed_j"]
        ledger.append(cumulative)
        bleed.append(cumulative_bleed)
    states = np.array(states)
    arrays = {"time_s": t, "bleed_j": np.array(bleed), "energy_residual_j": np.array(ledger)}
    for k in range(4):
        arrays["cell_" + str(k)] = states[:, 0, k]
    return make(
        name,
        {
            "duration_s": 1800,
            "shunt_s": 0.2 if enabled else 0,
            "noise_std_v": noise_std_v,
            "capacity_ah": p.capacity_ah,
        },
        {
            "initial_soc_spread": 0.08,
            "final_soc_spread": float(np.ptp(x[0])),
            "bleed_heat_j": cumulative_bleed,
            "max_energy_residual_j": float(max(abs(np.array(ledger)))),
        },
        arrays,
        [
            curve(
                "Measured-voltage passive balancing",
                "time_s",
                [(f"cell_{k}", f"Cell {k}") for k in range(4)],
                "Time (s)",
                "SOC fraction",
            ),
            curve(
                "Dissipated bleed energy",
                "time_s",
                [("bleed_j", "Shunt resistors")],
                "Time (s)",
                "Energy (J)",
            ),
            curve(
                "Chemical and RC energy closure",
                "time_s",
                [("energy_residual_j", "Cumulative residual")],
                "Time (s)",
                "Residual (J)",
                ylim=(-1e-4, 1e-4),
            ),
        ],
        assumptions=[
            "Standalone four-cell string, synthetic OCV and constant 25 C. Unit coulombic efficiency for explicit energy closure.",
            "Voltage is sampled with shunts momentarily off; acquisition duration is idealized as zero.",
        ],
    )


def balanced(scale=1, seed=42):
    return balance_run("cell-passive-balancing", scale, seed)


def unbalanced(scale=1, seed=42):
    return balance_run("cell-unbalanced-reference", scale, seed, enabled=False)


def noisy(scale=1, seed=42):
    return balance_run("cell-noisy-balancing", scale, seed, noise_std_v=0.003)


def weak(scale=1, seed=42):
    from .cellbank import Cells, advance

    p = Cells(capacity_ah=(2.0, 2.1, 1.0, 2.0))
    initial = np.array([[0.4] * 4, [0.0] * 4])
    r = advance(initial, 4, 1200, model=p)
    return make(
        "weakest-cell-soc-cutoff",
        {"current_a": 4, "capacity_ah": p.capacity_ah, "requested_s": 1200},
        {"elapsed_s": r["elapsed_s"], "boundary": r["boundary"], "energy_residual_j": r["energy_residual_j"]},
        {"cell": np.arange(4), "initial_soc": initial[0], "final_soc": r["state"][0]},
        [
            curve(
                "First-cell SOC termination",
                "cell",
                [("initial_soc", "Initial"), ("final_soc", "Terminated state")],
                "Cell index",
                "SOC fraction",
                ylim=(-0.02, 1.02),
            )
        ],
        assumptions=["SOC boundary study, not a complete voltage/thermal charge-protection algorithm."],
    )


def charge(scale=1, seed=42):
    from .cellbank import Cells, advance

    p = Cells(capacity_ah=(2.0, 2.1, 1.9, 2.0))
    initial = np.array([[0.99] * 4, [0.0] * 4])
    r = advance(initial, -3, 1200, model=p)
    return make(
        "cell-charge-soc-boundary",
        {"current_a": -3, "capacity_ah": p.capacity_ah, "requested_s": 1200},
        {"elapsed_s": r["elapsed_s"], "boundary": r["boundary"], "energy_residual_j": r["energy_residual_j"]},
        {"cell": np.arange(4), "initial_soc": initial[0], "final_soc": r["state"][0]},
        [
            curve(
                "First-cell SOC termination",
                "cell",
                [("initial_soc", "Initial"), ("final_soc", "Terminated state")],
                "Cell index",
                "SOC fraction",
                ylim=(-0.02, 1.02),
            )
        ],
        assumptions=["SOC boundary study, not a complete voltage/thermal charge-protection algorithm."],
    )
