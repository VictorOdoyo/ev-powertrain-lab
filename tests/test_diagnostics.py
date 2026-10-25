import numpy as np
import pytest

from evlab import diagnostics as m


def test_identification_recovers_excited_motor_without_assuming_known_flux():
    from evlab.motor import PMSM, steady_voltage

    p = PMSM()
    t = np.linspace(0, 1, 501)
    current = np.column_stack((30 * np.sin(31 * t), 80 + 20 * np.cos(43 * t)))
    derivative = np.column_stack((930 * np.cos(31 * t), -860 * np.sin(43 * t)))
    speed = 250 + 50 * np.sin(6 * t)
    voltage = steady_voltage(current, speed, p) + derivative * np.array([p.ld_h, p.lq_h])
    result = m.identify_motor(current, derivative, voltage, speed)
    assert result["resistance_ohm"] == pytest.approx(p.resistance_ohm)
    assert result["ld_h"] == pytest.approx(p.ld_h)
    assert result["lq_h"] == pytest.approx(p.lq_h)
    assert result["flux_wb"] == pytest.approx(p.flux_wb)


def test_current_sum_detects_one_biased_sensor_without_fault_truth():
    healthy = np.array([100, -40, -60.0])
    assert not m.current_sum(healthy)["alarm"]
    assert m.current_sum(healthy + np.array([10, 0, 0]))["alarm"]


def test_innovation_monitor_requires_persistent_measurement_inconsistency():
    monitor = m.InnovationMonitor(dwell_s=0.3)
    assert not monitor.update(20, 0.1)
    assert not monitor.update(20, 0.1)
    assert monitor.update(20, 0.1)
    assert not monitor.update(0, 0.1)


def test_tracking_residual_does_not_confuse_transport_delay_with_gain_error():
    target = np.arange(100, dtype=float)
    delayed = np.r_[0, 0, target[:-2]]
    assert m.tracking_residual(target, delayed, 2)["maximum"] == 0
