from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Config:
    cycle: str = "urban"
    duration_s: float = 60.0
    dt_s: float = 0.2
    initial_soc: float = 0.8
    observer_soc: float = 0.76
    ambient_c: float = 25.0
    battery_c: float = 25.0
    motor_c: float = 25.0
    mass_scale: float = 1.0
    cooling: bool = True
    regen: bool = True
    current_bias_a: float = 0.0
    voltage_bias_v: float = 0.0
    noise_v: float = 0.1
    seed: int = 42

    def __post_init__(self):
        from .core import integer, positive

        if self.cycle not in ("urban", "highway", "hill"):
            raise ValueError("known cycle required")
        for key in ("duration_s", "dt_s", "mass_scale"):
            positive(getattr(self, key), key)
        if self.dt_s > 1 or self.duration_s < 2 * self.dt_s:
            raise ValueError("resolved drive timing required")
        if not 0 < self.initial_soc < 1 or not 0 < self.observer_soc < 1:
            raise ValueError("interior initial SOC required")
        if not np.all(
            np.isfinite(
                [self.ambient_c, self.battery_c, self.motor_c, self.current_bias_a, self.voltage_bias_v]
            )
        ):
            raise ValueError("finite replay conditions required")
        if min(self.ambient_c, self.battery_c, self.motor_c) <= -273.15:
            raise ValueError("temperature below absolute zero")
        positive(self.noise_v, allow_zero=True)
        integer(self.seed, 0)
        if self.seed >= 2**32 or type(self.cooling) is not bool or type(self.regen) is not bool:
            raise ValueError("boolean switches and 32-bit seed required")

    def to_dict(self):
        return asdict(self)


def run(config):
    from dataclasses import replace

    from .battery import Pack, charge_inefficiency_heat, heat, step, terminal_voltage
    from .control import PI
    from .core import stream
    from .cycles import synthetic
    from .estimation import BatteryEKF
    from .powertrain import vehicle_step
    from .thermal import Thermostat, coolant, lumped_step
    from .vehicle import Vehicle, road_forces

    cycle = synthetic(config.cycle, config.dt_s, config.duration_s)
    time = cycle.time_s
    n = len(time)
    vehicle = replace(Vehicle(), mass_kg=Vehicle().mass_kg * config.mass_scale)
    pack = Pack()
    observer = BatteryEKF(config.observer_soc, pack)
    controller = PI(vehicle.effective_mass, vehicle.effective_mass * 0.3, 15000, 0.5)
    fan = Thermostat()
    noise = stream(config.seed, "battery-voltage").normal(0, config.noise_v, n)
    keys = (
        "speed_m_s",
        "soc",
        "estimated_soc",
        "battery_current_a",
        "battery_voltage_v",
        "battery_power_w",
        "motor_torque_nm",
        "motor_copper_w",
        "inverter_loss_w",
        "battery_c",
        "motor_c",
        "friction_n",
        "distance_m",
        "bus_residual_v",
        "mechanical_defect_j",
        "pump_w",
        "innovation_v",
        "limited",
    )
    arrays = {key: np.zeros(n) for key in keys}
    arrays["current_dq_a"] = np.zeros((n, 2))
    arrays["estimator_accepted"] = np.zeros(n, bool)
    battery = np.array([config.initial_soc, 0.0])
    speed = 0.0
    tb = config.battery_c
    tm = config.motor_c
    distance = 0.0
    arrays["soc"][0] = battery[0]
    arrays["estimated_soc"][0] = observer.state[0]
    arrays["battery_c"][0] = tb
    arrays["motor_c"][0] = tm
    for k in range(1, n):
        dt = time[k] - time[k - 1]
        flowing = fan.update(time[k], max(tb, tm)) if config.cooling else False
        cooling = coolant(1.0 if flowing else 0.0)
        auxiliary = 300.0 + cooling["pump_w"]
        road = float(road_forces(speed, cycle.grade[k], model=vehicle)["total_n"])
        requested = controller.update(cycle.speed_m_s[k] - speed, dt, road)
        if not config.regen and requested < 0:
            point = vehicle_step(speed, 0, battery, dt, cycle.grade[k], tb, tm, auxiliary, vehicle, pack)
            desired = min(15000.0, -requested)
            next_speed = max(
                0.0, speed + (point["wheel_force_n"] - desired - road) * dt / vehicle.effective_mass
            )
            point["speed_m_s"] = next_speed
            point["friction_n"] = desired
            point["wheel_force_n"] -= desired
            point["distance_m"] = (speed + next_speed) * dt / 2
            midpoint = (speed + next_speed) / 2
            actual_road = float(road_forces(midpoint, cycle.grade[k], model=vehicle)["total_n"])
            point["mechanical_defect_j"] = (
                0.5 * vehicle.effective_mass * (next_speed**2 - speed**2)
                - (point["wheel_force_n"] - actual_road) * midpoint * dt
            )
        else:
            point = vehicle_step(
                speed, requested, battery, dt, cycle.grade[k], tb, tm, auxiliary, vehicle, pack
            )
        controller.integral += dt * ((point["wheel_force_n"] - requested) / controller.tracking)
        current = point["battery_current_a"]
        old_battery = battery.copy()
        battery = step(battery, current, dt, pack)
        terminal = float(terminal_voltage(battery[0], battery[1], current, tb, pack))
        estimate = observer.update(
            terminal + config.voltage_bias_v + noise[k], current + config.current_bias_a, dt, tb
        )
        battery_heat = heat(old_battery[0], old_battery[1], current, tb, pack)[
            "total_w"
        ] + charge_inefficiency_heat(current, old_battery[0], tb, pack)
        tb = lumped_step(
            tb, battery_heat, config.ambient_c, 400000, 1 / (10 + cooling["conductance_w_k"]), dt
        )
        tm = lumped_step(
            tm, point["copper_w"], config.ambient_c, 12000, 1 / (8 + cooling["conductance_w_k"]), dt
        )
        speed = point["speed_m_s"]
        distance += point["distance_m"]
        values = {
            "speed_m_s": speed,
            "soc": battery[0],
            "estimated_soc": observer.state[0],
            "battery_current_a": current,
            "battery_voltage_v": terminal,
            "battery_power_w": point["battery_power_w"],
            "motor_torque_nm": point["torque_nm"],
            "motor_copper_w": point["copper_w"],
            "inverter_loss_w": point["inverter_w"],
            "battery_c": tb,
            "motor_c": tm,
            "friction_n": point["friction_n"],
            "distance_m": distance,
            "bus_residual_v": point["bus_residual_v"],
            "mechanical_defect_j": point["mechanical_defect_j"],
            "pump_w": cooling["pump_w"],
            "innovation_v": estimate["innovation_v"],
            "limited": float(point["limited"]),
        }
        for key, value in values.items():
            arrays[key][k] = value
        arrays["current_dq_a"][k] = point["current_dq_a"]
        arrays["estimator_accepted"][k] = estimate["accepted"]
    arrays["time_s"] = time
    arrays["reference_m_s"] = cycle.speed_m_s
    arrays["grade"] = cycle.grade
    return arrays
