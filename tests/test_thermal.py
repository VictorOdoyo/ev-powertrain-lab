import numpy as np
import pytest

from evlab import thermal as m


def test_lumped_temperature_matches_step_response():
    assert m.lumped_step(25, 100, 25, 1000, 0.1, 100) == pytest.approx(25 + 10 * (1 - np.exp(-1)))


def test_internal_thermal_exchange_conserves_heat():
    net = m.network([100, 200], [(0, 1, 3)], [0, 0])
    assert np.allclose(np.sum(net["conductance"], axis=0), 0)


def test_coupled_thermal_step_conserves_energy_without_ambient():
    net = m.network([100, 200], [(0, 1, 3)], [0, 0])
    old = np.array([100.0, 0.0])
    new = m.network_step(old, [0, 0], 25, 10, net)
    assert new @ net["capacity"] == pytest.approx(old @ net["capacity"])


def test_thermal_derating_has_bounded_linear_transition():
    assert np.allclose(m.derate([60, 80, 95, 110, 120]), [1, 1, 0.5, 0, 0])


def test_pump_affinity_is_not_confused_with_heat_transfer_scaling():
    half = m.coolant(0.5)
    assert half["pump_w"] == pytest.approx(37.5)
    assert half["conductance_w_k"] > 50
