import pytest

from evlab import motor as m


def test_motor_parameters_reject_nonphysical_inductance():
    with pytest.raises(ValueError):
        m.PMSM(ld_h=0)


def test_saliency_and_torque_reversal():
    p = m.PMSM()
    assert m.torque([0, 100], p) == pytest.approx(48)
    assert m.torque([-100, 100], p) > 48
    assert m.torque([0, -100], p) == pytest.approx(-48)
