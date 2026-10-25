"""Run standalone component benches from an installed evlab environment."""

from evlab.advanced_control import deadbeat_voltage
from evlab.battery import Pack, step, terminal_voltage
from evlab.motor import exact_step


def main():
    p = Pack()
    state = step([0.8, 0], 100, 10, p)
    print("Battery SOC, polarization, terminal V:", state, terminal_voltage(*state, 100, model=p))
    command = deadbeat_voltage([0, 50], [1, 51], 100, 0.0001, 400)
    print("Next d-q current:", exact_step([0, 50], command["applied_v"], 100, 0.0001))


if __name__ == "__main__":
    main()
