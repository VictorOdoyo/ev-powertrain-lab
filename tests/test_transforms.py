import numpy as np

from evlab import transforms as m


def test_clarke_balanced_and_common_mode():
    assert np.allclose(m.clarke([1, -0.5, -0.5]), [1, 0, 0])
    assert np.allclose(m.clarke([2, 2, 2]), [0, 0, 2])
