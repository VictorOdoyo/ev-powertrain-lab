import numpy as np
import pytest

from evlab import estimation as m


def test_covariance_prediction_preserves_positive_semidefiniteness():
    result = m.predict([0.8, 0], np.diag([0.01, 1.0]), 100, 1)
    assert np.linalg.eigvalsh(result["covariance"]).min() >= 0
    assert result["transition"][1, 1] < 1


def test_ekf_correction_uses_joseph_form_and_rejects_outlier():
    from evlab.battery import terminal_voltage

    prediction = m.predict([0.6, 0], np.diag([0.01, 0.1]), 0, 1)
    measured = float(terminal_voltage(0.7, 0, 0))
    update = m.correct(prediction, measured, 0)
    assert update["accepted"] and abs(update["state"][0] - 0.7) < 0.03
    assert np.linalg.eigvalsh(update["covariance"]).min() >= 0
    assert not m.correct(prediction, 10000, 0)["accepted"]


def test_battery_estimator_converges_from_soc_error_using_only_voltage_and_current():
    from evlab.battery import step, terminal_voltage

    observer = m.BatteryEKF(0.65)
    truth = np.array([0.7, 0.0])
    for k in range(300):
        current = 80 if (k // 50) % 2 else 0
        truth = step(truth, current, 1)
        observer.update(float(terminal_voltage(truth[0], truth[1], current)), current, 1)
    assert abs(observer.state[0] - truth[0]) < 0.003


def test_pulse_resistance_recovers_pack_r0_before_polarization_changes():
    from evlab.battery import Pack, terminal_voltage

    p = Pack()
    before = float(terminal_voltage(0.8, 0, 0, model=p))
    after = float(terminal_voltage(0.8, 0, 100, model=p))
    assert m.pulse_resistance(before, after, 0, 100) == pytest.approx(p.r0)


def test_relaxation_fit_estimates_unknown_offset_amplitude_and_tau():
    t = np.linspace(5, 150, 100)
    fit = m.fit_relaxation(t, 380 - 2 * np.exp(-(t - 5) / 30))
    assert fit["tau_s"] == pytest.approx(30, rel=1e-6)
    assert fit["residual_v"] < 1e-8
