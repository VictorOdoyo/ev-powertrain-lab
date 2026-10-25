from dataclasses import asdict

import numpy as np

from .results import curve, make
from .simulation import Config, run


def study(name, config):
    from .battery import Pack
    from .motor import PMSM
    from .vehicle import Vehicle

    a = run(config)
    t = a["time_s"]
    a["speed_error_m_s"] = a["reference_m_s"] - a["speed_m_s"]
    a["soc_error"] = a["estimated_soc"] - a["soc"]
    a["valid"] = t > 0
    summary = {
        "distance_m": float(a["distance_m"][-1]),
        "net_battery_wh": float(np.trapezoid(a["battery_power_w"], t) / 3600),
        "final_soc": float(a["soc"][-1]),
        "speed_rmse_m_s": float(np.sqrt(np.mean(a["speed_error_m_s"] ** 2))),
        "limited_fraction": float(np.mean(a["limited"])),
        "maximum_bus_residual_v": float(np.max(abs(a["bus_residual_v"]))),
    }
    plots = [
        curve(
            "Reference and achieved motion",
            "time_s",
            [("reference_m_s", "Requested"), ("speed_m_s", "Achieved")],
            "Time (s)",
            "Speed (m/s)",
        ),
        curve(
            "Battery state and observer",
            "time_s",
            [("soc", "Physical SOC"), ("estimated_soc", "Voltage/current EKF")],
            "Time (s)",
            "SOC",
        ),
        curve(
            "Battery power and losses",
            "time_s",
            [
                ("battery_power_w", "Battery"),
                ("motor_copper_w", "Motor copper"),
                ("inverter_loss_w", "Inverter"),
            ],
            "Time (s)",
            "Power (W)",
            valid="valid",
        ),
        curve(
            "Coupled thermal states",
            "time_s",
            [("battery_c", "Battery"), ("motor_c", "Motor")],
            "Time (s)",
            "Temperature (deg C)",
        ),
    ]
    return make(
        name,
        {
            "replay": config.to_dict(),
            "motor": asdict(PMSM()),
            "pack": asdict(Pack()),
            "vehicle": asdict(Vehicle()),
        },
        summary,
        a,
        plots,
        assumptions=[
            "Synthetic reference; no regulatory cycle certification.",
            "Quasi-steady feasible PMSM points; separate bench resolves current-loop dynamics.",
            "Illustrative OCV, losses, temperature and aging coefficients.",
            "Battery observer is evaluated separately from the physical BMS constraints.",
        ],
    )


def urban(scale=1, seed=42):
    return study("urban-drive", Config(dt_s=0.2 / scale, duration_s=40, seed=seed))


def highway(scale=1, seed=42):
    return study("highway-drive", Config(cycle="highway", dt_s=0.2 / scale, duration_s=40, seed=seed))


def hill(scale=1, seed=42):
    return study("hill-drive", Config(cycle="hill", dt_s=0.2 / scale, duration_s=40, seed=seed))


def friction_only(scale=1, seed=42):
    return study("friction-braking", Config(dt_s=0.2 / scale, duration_s=40, regen=False, seed=seed))


def full_pack(scale=1, seed=42):
    return study("full-pack-braking", Config(dt_s=0.2 / scale, duration_s=40, initial_soc=0.9999, seed=seed))


def hot_pack(scale=1, seed=42):
    return study("hot-pack-shutdown", Config(dt_s=0.2 / scale, duration_s=10, battery_c=65, seed=seed))
