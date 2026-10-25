import numpy as np
import pytest

from evlab import statistics as m


def test_correlated_samples_preserve_covariance_and_repeat_seed():
    a = m.correlated_samples([0, 0], [[1, 0.8], [0.8, 1]], 10000)
    assert np.corrcoef(a.T)[0, 1] == pytest.approx(0.8, abs=0.02)
    assert np.array_equal(a, m.correlated_samples([0, 0], [[1, 0.8], [0.8, 1]], 10000))
