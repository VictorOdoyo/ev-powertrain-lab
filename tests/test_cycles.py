import pytest

from evlab import cycles as m


def test_cycle_rejects_unordered_time_and_negative_speed():
    with pytest.raises(ValueError):
        m.Cycle([0, 1], [-1, 2])
    with pytest.raises(ValueError):
        m.Cycle([0, 0], [1, 2])
