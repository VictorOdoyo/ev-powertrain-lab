import pytest

from evlab import simulation as m


def test_drive_config_rejects_unphysical_and_ambiguous_conditions():
    with pytest.raises(ValueError):
        m.Config(initial_soc=1.2)
    with pytest.raises(ValueError):
        m.Config(cooling=1)
    with pytest.raises(ValueError):
        m.Config(ambient_c=-300)
