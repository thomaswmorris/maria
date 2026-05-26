from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from ..io import DEFAULT_TIME_FORMAT
from ..units import Quantity
from .plan import Plan


class PlanList:
    def __init__(self, plans: list[Plan]):
        if isinstance(plans, PlanList):
            plans = plans.plans
        elif not isinstance(plans, list):
            raise ValueError("'plans' must be a list of Plans")

        self.plans = plans

    def summary(self):
        entries = []
        for p in self.plans:
            c = p.center()
            entry = {
                "start_time": p.start_time.format(DEFAULT_TIME_FORMAT),
                "duration": p.duration,
                # "site": "" if not p.naive else p.site,
                f"target({p.frame.phi_name},{p.frame.theta_name})": c,
            }

            if not p.naive and not p.frame == "az/el":
                entry["center(az,el)"] = str(p.center(frame="az/el"))

            entries.append(entry)

        s = pd.DataFrame(entries)
        s.index.name = "chunk"

        return s

    @property
    def duration(self):
        return Quantity(sum([t.s for t in self.summary().duration]), "s")

    def __repr__(self):
        summary = self.summary()
        return f"""PlanList({len(self.plans)} plans, {self.duration}):
{str(summary)}"""

    def __getitem__(self, index):
        if type(index) is int:
            return self.plans[index]
        elif type(index) is slice:
            return type(self)(self.plans[index])
        else:
            raise ValueError(
                f"Invalid index {index}. A bandList must be indexed by either an integer or a string.",
            )

    def plan_groups(self, max_break: float = 60):
        plan_groups = []
        last_plan_end = -np.inf
        for plan_index, p in enumerate(self.plans):
            if p.start_time.timestamp() - last_plan_end < max_break:
                plan_groups[-1].append(plan_index)
            else:
                plan_groups.append([plan_index])
            last_plan_end = p.end_time.timestamp()

        return plan_groups

    def group_plans(self):
        grouped_plans = []
        for group in self.plan_groups():
            merged_plan = copy.deepcopy(self.plans[group[0]])
            for plan_index in group[1:]:
                merged_plan += self.plans[plan_index]
            grouped_plans.append(merged_plan)

        return PlanList(grouped_plans)
