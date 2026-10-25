import numpy as np

from evlab import estimation as m


def test_covariance_prediction_preserves_positive_semidefiniteness():
    result = m.predict([0.8, 0], np.diag([0.01, 1.0]), 100, 1)
    assert np.linalg.eigvalsh(result["covariance"]).min() >= 0
    assert result["transition"][1, 1] < 1
