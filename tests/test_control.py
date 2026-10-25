import pytest

from evlab import control as m


def test_pi_unwinds_after_sustained_saturation():
    controller = m.PI(1, 10, 2, 0.01)
    for _ in range(1000):
        controller.update(10, 0.001)
    outputs = [controller.update(0, 0.001) for _ in range(1000)]
    assert abs(outputs[-1]) < 2.001
    assert abs(controller.integral) < 2.001


def test_current_gain_tuning_preserves_axis_time_constants():
    from evlab.motor import PMSM

    p = PMSM()
    gains = m.current_gains(p)
    assert gains["kp"][1] / gains["kp"][0] == pytest.approx(p.lq_h / p.ld_h)
