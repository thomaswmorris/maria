import pytest

import maria
from maria.plan import patterns


@pytest.mark.parametrize("scan_pattern", patterns.index)
def test_pattern(scan_pattern):
    plan = maria.Plan(scan_pattern=scan_pattern)
    print(plan)

    plan.plot()
