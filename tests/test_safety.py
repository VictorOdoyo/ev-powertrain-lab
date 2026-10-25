import pytest

from evlab import safety as m


def test_contactor_latches_after_dwell_and_rejects_loaded_reset():
    r = m.Contactor()
    assert r.update(400, 0.005)
    assert not r.update(400, 0.005)
    assert not r.update(0, 1)
    with pytest.raises(ValueError):
        r.reset(100)
    r.reset(0)
    assert r.closed
