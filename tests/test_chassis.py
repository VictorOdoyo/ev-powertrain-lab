import pytest

from evlab import chassis as m


def test_friction_circle_uses_remaining_vector_capacity():
    assert m.friction_circle(10000, 3000, 5000, 1) == pytest.approx(4000)


def test_axle_load_transfer_preserves_total_normal_force():
    a = m.axle_loads(1800, 3)
    b = m.axle_loads(1800, 0)
    assert a["rear_n"] > b["rear_n"]
    assert a["front_n"] + a["rear_n"] == pytest.approx(1800 * 9.80665)


def test_tire_force_is_odd_and_saturates_at_friction_capacity():
    assert m.tire_force(0, 4000) == 0
    assert m.tire_force(100, 4000) == pytest.approx(3200)
    assert m.tire_force(-0.1, 4000) == -m.tire_force(0.1, 4000)
