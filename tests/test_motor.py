import numpy as np
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


def test_back_emf_at_zero_current_is_not_omitted():
    p = m.PMSM()
    rhs = m.electrical_rhs([0, 0], [0, 0], 100, p)
    assert rhs[0] == 0
    assert rhs[1] == pytest.approx(-p.pole_pairs * 100 * p.flux_wb / p.lq_h)


def test_steady_voltage_annuls_independent_dynamic_rhs():
    p = m.PMSM()
    i = np.array([-30.0, 80.0])
    assert np.allclose(m.electrical_rhs(i, m.steady_voltage(i, 300, p), 300, p), 0, atol=1e-10)
