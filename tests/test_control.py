import numpy as np
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


def test_field_oriented_current_loop_tracks_with_actual_motor_updates():
    from evlab.motor import PMSM, exact_step

    p = PMSM()
    controller = m.CurrentLoop(p)
    current = np.zeros(2)
    dt = 50e-6
    for _ in range(800):
        voltage = controller.update([0, 80], current, 200, 400, dt)
        current = exact_step(current, voltage, 200, dt, p)
    assert np.linalg.norm(current - [0, 80]) < 0.01


def test_command_delay_has_exact_sample_latency_and_no_aliasing():
    delay = m.Delay(2, [0, 0])
    assert np.allclose(delay.update([1, 2]), [0, 0])
    assert np.allclose(delay.update([3, 4]), [0, 0])
    assert np.allclose(delay.update([5, 6]), [1, 2])


def test_adc_error_is_bounded_and_clipping_is_explicit():
    value = np.array([1.0, 3.0, 1000.0])
    result = m.quantize_current(value)
    assert np.max(abs(result["value"][:2] - value[:2])) <= result["lsb_a"] / 2 + 1e-12
    assert result["clipped"][-1]


def test_torque_slew_is_time_based_not_sample_count_based():
    assert m.slew(0, 100, 1000, 0.01) == 10
    assert m.slew(0, -100, 1000, 0.01) == -10
