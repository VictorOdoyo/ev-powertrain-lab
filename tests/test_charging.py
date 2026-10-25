import numpy as np
import pytest

from evlab import charging as m


def test_cccv_command_respects_predicted_terminal_voltage_and_soc():
    from evlab.battery import Pack, step, terminal_voltage

    p = Pack()
    state = np.array([0.96, -2.0])
    current = m.command(state, 10, model=p)
    future = step(state, current, 10, p)
    assert current <= 0
    assert terminal_voltage(future[0], future[1], current, model=p) <= p.series * 4.2 + 1e-8


def test_grid_charging_power_includes_losses_and_auxiliaries():
    assert m.grid_power(-9400) == pytest.approx(10100)


def test_charger_blocks_subzero_and_hot_cells_and_derates_cold_pack():
    assert m.thermal_current_limit(-5) == 0
    assert m.thermal_current_limit(55) == 0
    assert m.thermal_current_limit(7.5) == 100
