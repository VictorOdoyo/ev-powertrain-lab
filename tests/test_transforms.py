import numpy as np

from evlab import transforms as m


def test_clarke_balanced_and_common_mode():
    assert np.allclose(m.clarke([1, -0.5, -0.5]), [1, 0, 0])
    assert np.allclose(m.clarke([2, 2, 2]), [0, 0, 2])


def test_inverse_clarke_roundtrip_unbalanced():
    v = np.random.default_rng(3).normal(size=(50, 3))
    assert np.allclose(m.inverse_clarke(m.clarke(v)), v)


def test_park_aligns_rotor_flux_and_preserves_norm():
    angle = 0.7
    assert np.allclose(m.park([np.cos(angle), np.sin(angle)], angle), [1, 0])
