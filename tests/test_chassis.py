import pytest

from evlab import chassis as m


def test_friction_circle_uses_remaining_vector_capacity():
    assert m.friction_circle(10000, 3000, 5000, 1) == pytest.approx(4000)
