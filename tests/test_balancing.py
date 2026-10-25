import numpy as np
import pytest

from evlab import balancing as m


def test_balancing_bleeds_high_groups_and_accounts_for_heat():
    result = m.bleed([0.5, 0.6, 0.7], [3.6, 3.7, 3.8], 1)
    assert np.array_equal(result["bleed_a"], [0, 1, 1])
    assert result["heat_w"] == pytest.approx(7.5)
