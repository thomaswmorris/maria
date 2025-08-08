import logging
from collections.abc import Mapping

import arrow
import numpy as np
import scipy as sp

from ..coords import Coordinates
from ..errors import NoSuitablePlansError
from ..site import Site, get_site
from ..units import Quantity
from ..utils import get_day_hour, grouper
from .plan import Plan
from .plan_list import PlanList

CONSTRAINT_KEYS = ["az", "el", "hour"]
SIDEREAL_DAY = Quantity(86164.0905, "s")

logger = logging.getLogger("maria")


class Planner:
    def __init__(
        self,
        target,
        site: Site,
        constraints: Mapping,
        max_lookahead: float = 2 * 365 * 86400.0,
        start_time: float = None,
    ):
        self.target = target

        if isinstance(site, Site):
            self.site = site
        elif isinstance(site, str):
            self.site = get_site(site_name=site)
        else:
            raise ValueError(
                "'site' must be either a Site object or a string.",
            )

        self.constraints = {}
        if not isinstance(constraints, Mapping):
            raise TypeError("'constraints' must be a dict or a mapping")
        for key, value in constraints.items():
            if key in ["az", "el"]:
                self.constraints[key] = (Quantity(value[0], "deg"), Quantity(value[1], "deg"))
            elif key == "hour":
                self.constraints[key] = value
            else:
                raise ValueError(f"constraint keys must be one of {CONSTRAINT_KEYS}")

        self.start_time = arrow.get(start_time or arrow.get().timestamp(), tzinfo=self.site.timezone)
        self.max_lookahead = max_lookahead

    @property
    def start_day(self):
        return arrow.get(year=self.start_time.year, month=self.start_time.month, day=self.start_time.day)

    @staticmethod
    def apply_constraint(x, c1, c2):
        if c1 < c2:
            return (x >= c1) & (x <= c2)
        if c1 > c2:
            return (x >= c1) | (x <= c2)

    def generate_obs_intervals(self, total_duration: float, max_chunk_duration: float, test_points_per_chunk: int = 16):
        max_chunk_duration = max_chunk_duration or np.inf

        sidereal_day_samples = np.linspace(0, 1, 256)
        sample_t = self.start_time.timestamp() + sidereal_day_samples * SIDEREAL_DAY.s
        self.c = Coordinates(
            phi=self.target.center[0].radians,
            theta=self.target.center[1].radians,
            t=sample_t,
            frame=self.target.frame,
            earth_location=self.site.earth_location,
        )

        t_test = np.arange(0, self.max_lookahead, max_chunk_duration / test_points_per_chunk)
        sidereal_day = (t_test % SIDEREAL_DAY.s) / SIDEREAL_DAY.s

        self.day_hour_test = (get_day_hour(self.start_time) + (t_test % 86400) / 3600) % 24
        self.az_test = Quantity(sp.interpolate.interp1d(sidereal_day_samples, np.unwrap(self.c.az))(sidereal_day), "rad")
        self.el_test = Quantity(sp.interpolate.interp1d(sidereal_day_samples, self.c.el)(sidereal_day), "rad")

        mask = np.ones_like(t_test, dtype=bool)

        if "el" in self.constraints:
            mask &= self.apply_constraint(self.el_test, *self.constraints["el"])
        if "az" in self.constraints:
            mask &= self.apply_constraint(self.az_test, *self.constraints["az"])
        if "hour" in self.constraints:
            mask &= self.apply_constraint(self.day_hour_test, *self.constraints["hour"])

        logger.debug(f"{1e2 * mask.sum() / len(mask):.02f}% of times satisfy the observing constraints")

        chunks = []
        duration = 0
        for start, end in grouper(
            mask, min_length=test_points_per_chunk, max_length=test_points_per_chunk + 1, overlap=True
        ):
            this_chunk_duration = np.min([max_chunk_duration, t_test[end] - t_test[start], total_duration - duration])

            chunk_start = self.start_time.shift(seconds=t_test[start])
            chunks.append({"start_time": chunk_start, "duration": this_chunk_duration})
            duration += this_chunk_duration

            if duration >= total_duration:
                break

        return chunks

    def generate_plans(
        self, total_duration: float, max_chunk_duration: float = 600, scan_options: Mapping = {}, **plan_kwargs
    ):
        scan_options["radius"] = scan_options.get("radius", self.target.width.deg / 2)

        chunks = self.generate_obs_intervals(total_duration=total_duration, max_chunk_duration=max_chunk_duration)
        total_duration_of_chunks = sum([chunk["duration"] for chunk in chunks])

        if not chunks:
            raise NoSuitablePlansError(
                constraints=self.constraints,
                max_chunk_duration=max_chunk_duration,
                tmin=self.start_time,
                tmax=self.start_time.shift(seconds=self.max_lookahead),
            )

        if total_duration_of_chunks < total_duration:
            logger.warning(
                f"Found only {total_duration_of_chunks} seconds of observing time between {self.start_time}"
                f" and {self.start_time.shift(seconds=self.max_lookahead)}; consider increasing the "
                "'max_lookahead' parameter"
            )

        plans = [
            Plan.generate(
                scan_center=self.target.center,
                frame=self.target.frame,
                site=self.site,
                scan_options=scan_options,
                **plan_kwargs,
                **chunk,
            )
            for chunk in chunks
        ]

        return PlanList(plans)

    def generate_plan(self, total_duration: float, **plan_kwargs):
        return self.generate_plans(total_duration=total_duration, max_chunk_duration=total_duration, **plan_kwargs)[0]
