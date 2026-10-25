import numpy as np

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
