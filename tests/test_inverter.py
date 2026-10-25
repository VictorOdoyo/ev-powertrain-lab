import numpy as np
import pytest

from evlab import inverter as m


def test_vector_limit_preserves_direction_and_linear_modulation_bound():
    command = np.array([300.0, 400.0])
    result = m.limit_vector(command, 400)
    assert np.linalg.norm(result) == pytest.approx(0.98 * 400 / np.sqrt(3))
    assert result[0] / result[1] == pytest.approx(0.75)


def test_svpwm_reconstructs_command_and_preserves_duty_bounds():
    from evlab.transforms import clarke

    duty = m.svpwm([100, 50], 400)
    phase = 400 * (duty - duty.mean())
    assert np.allclose(clarke(phase)[:2], [100, 50])
    assert np.all((duty >= 0) & (duty <= 1))


def test_switching_average_matches_commanded_duty():
    t = (np.arange(10000) + 0.5) / 1e8
    duty = np.array([0.2, 0.5, 0.8])
    assert np.allclose(np.mean(m.switching(t, duty), axis=0), duty, atol=2e-4)
