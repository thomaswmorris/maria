import pytest

import maria


@pytest.mark.parametrize("plan_name", maria.all_plans)
def test_get_plan(plan_name):
    plan = maria.get_plan(plan_name)
