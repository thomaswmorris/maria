import pytest

import maria


@pytest.mark.parametrize(
    "scan_pattern", ["stare", "daisy", "raster", "grid", "back_and_forth"]
)
def test_pattern(scan_pattern):
    plan = maria.Plan(scan_pattern=scan_pattern)
    print(plan)
