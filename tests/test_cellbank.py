import pytest

from evlab import cellbank as m


def test_heterogeneous_parameters():
    p = m.Cells()
    assert p.count == 4
    with pytest.raises(ValueError):
        m.Cells(capacity_ah=(2, 3), r0_ohm=(0.02,))
    assert m.validate_state([[0.5] * 4, [0] * 4]).shape == (2, 4)
