import numpy as np
import pytest

from evlab import core as m


def test_contracts_streams_and_nonuniform_energy_integrals():
    with pytest.raises(ValueError):
        m.finite([np.nan])
    with pytest.raises(ValueError):
        m.positive(True)
    with pytest.raises(ValueError):
        m.integer(1.5)
    assert np.array_equal(m.stream(42, "battery").normal(size=8), m.stream(42, "battery").normal(size=8))
    assert m.integrate([0, 0.1, 0.4], [0, 0.1, 0.4]) == pytest.approx(0.08)
