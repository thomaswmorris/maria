import pytest

import maria
from maria.plan import PATTERNS


@pytest.mark.parametrize("scan_pattern", PATTERNS.index)
def test_pattern(scan_pattern):
    plan = maria.Plan(scan_pattern=scan_pattern)
    print(plan)
