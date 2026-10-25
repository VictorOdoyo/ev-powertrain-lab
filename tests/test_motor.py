import pytest

from evlab import motor as m


def test_motor_parameters_reject_nonphysical_inductance():
    with pytest.raises(ValueError):
        m.PMSM(ld_h=0)
