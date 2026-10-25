import numpy as np

from evlab import benches as m


def test_current_loop_reaches_target_against_independent_exact_plant():
    a = m.current_loop()
    assert np.linalg.norm(a["current_dq_a"][-1] - [0, 80]) < 0.02
