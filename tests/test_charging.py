import numpy as np

from evlab import charging as m


def test_cccv_command_respects_predicted_terminal_voltage_and_soc():
    from evlab.battery import Pack, step, terminal_voltage

    p = Pack()
    state = np.array([0.96, -2.0])
    current = m.command(state, 10, model=p)
    future = step(state, current, 10, p)
    assert current <= 0
    assert terminal_voltage(future[0], future[1], current, model=p) <= p.series * 4.2 + 1e-8
