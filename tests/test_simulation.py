import numpy as np
import pytest

from evlab import simulation as m


def test_drive_config_rejects_unphysical_and_ambiguous_conditions():
    with pytest.raises(ValueError):
        m.Config(initial_soc=1.2)
    with pytest.raises(ValueError):
        m.Config(cooling=1)
    with pytest.raises(ValueError):
        m.Config(ambient_c=-300)


def test_forward_drive_replay_reports_achieved_speed_state_and_energy():
    result = m.run(m.Config(duration_s=8, dt_s=0.2))
    assert result["distance_m"][-1] > 0
    assert result["soc"][-1] < 0.8
    assert np.max(abs(result["bus_residual_v"])) < 1e-4
    assert np.all(result["speed_m_s"] >= 0)
