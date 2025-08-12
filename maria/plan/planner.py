import logging
from collections.abc import Mapping

import arrow
import astropy as ap
import numpy as np
import scipy as sp

from ..coords import Coordinates
from ..errors import NoSuitablePlansError, PointingError
from ..site import Site, get_site
from ..units import Quantity
from ..utils import get_day_hour, great_circle_distance, grouper
from .plan import Plan
from .plan_list import PlanList

CONSTRAINT_KEYS = ["az", "el", "hour", "min_sun_distance"]
SIDEREAL_DAY_SECONDS = Quantity(86164.0905, "s")
YEAR_SECONDS = Quantity(31_556_926, "s")


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
            elif key == "min_sun_distance":
                self.constraints[key] = Quantity(value, "deg")
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

        self.delta_t_test = np.arange(0, self.max_lookahead, max_chunk_duration / test_points_per_chunk)
        self.mask = np.ones_like(self.delta_t_test, dtype=bool)

        if any([key in self.constraints for key in ["az", "el"]]):
            sidereal_day_seconds_test = self.delta_t_test % SIDEREAL_DAY_SECONDS.s
            sidereal_day_seconds_samples = np.linspace(0, SIDEREAL_DAY_SECONDS.s, 256)
            target_sample_t = self.start_time.timestamp() + sidereal_day_seconds_samples
            self.c_target = Coordinates(
                phi=self.target.center[0].radians,
                theta=self.target.center[1].radians,
                t=target_sample_t,
                frame=self.target.frame,
                earth_location=self.site.earth_location,
            )

            self.target_az_test = Quantity(
                sp.interpolate.interp1d(sidereal_day_seconds_samples, np.unwrap(self.c_target.az))(sidereal_day_seconds_test)
                % (2 * np.pi),
                "rad",
            )
            self.target_el_test = Quantity(
                sp.interpolate.interp1d(sidereal_day_seconds_samples, self.c_target.el)(sidereal_day_seconds_test), "rad"
            )

            if "el" in self.constraints:
                self.mask &= self.target_el_test >= self.constraints["el"][0]
                if not self.mask.any():
                    raise PointingError(
                        f"target is never above an elevation of {self.constraints['el'][0]} "
                        f"(el_max = {self.target_el_test.max()})"
                    )
                self.mask &= self.target_el_test <= self.constraints["el"][1]
                if not self.mask.any():
                    raise PointingError(
                        f"target is never below an elevation of {self.constraints['el'][1]} "
                        f"(el_min = {self.target_el_test.min()})"
                    )

            if "az" in self.constraints:
                self.mask &= self.apply_constraint(self.target_az_test, *self.constraints["az"])

        if "min_sun_distance" in self.constraints:
            year_fraction_test = (self.delta_t_test % YEAR_SECONDS.s) / YEAR_SECONDS.s
            year_fraction_samples = np.linspace(0, 1, 256)
            sun_sample_t = self.start_time.timestamp() + year_fraction_samples * YEAR_SECONDS.s

            sun = ap.coordinates.get_sun(ap.time.Time(sun_sample_t, format="unix"))
            sun_distance_samples = great_circle_distance(
                self.target.center[0].rad, self.target.center[1].rad, sun.ra.rad, sun.dec.rad
            )

            self.sun_distance_test = Quantity(
                sp.interpolate.interp1d(year_fraction_samples, sun_distance_samples)(year_fraction_test), "rad"
            )

            self.mask &= self.sun_distance_test >= self.constraints["min_sun_distance"]

        if "hour" in self.constraints:
            day_hour_test = (get_day_hour(self.start_time) + (self.delta_t_test % 86400) / 3600) % 24
            self.mask &= self.apply_constraint(day_hour_test, *self.constraints["hour"])

        logger.debug(f"{1e2 * self.mask.sum() / len(self.mask):.02f}% of times satisfy the observing constraints")

        chunks = []
        duration = 0
        for start, end in grouper(
            self.mask, min_length=test_points_per_chunk, max_length=test_points_per_chunk + 1, overlap=True
        ):
            this_chunk_duration = np.min(
                [max_chunk_duration, self.delta_t_test[end] - self.delta_t_test[start], total_duration - duration]
            )

            debug_str_parts = [f"start = {arrow.get(self.start_time.timestamp() + self.delta_t_test[start])}"]
            if "az" in self.constraints:
                chunk_az = self.target_az_test[start:end]
                debug_str_parts.append(f"az = {chunk_az.mean()} ± {chunk_az.std()}")
            if "el" in self.constraints:
                chunk_el = self.target_el_test[start:end]
                debug_str_parts.append(f"el = {chunk_el.mean()} ± {chunk_el.std()}")
            logger.debug(f"found good chunk ({', '.join(debug_str_parts)})")

            chunk_start = self.start_time.shift(seconds=self.delta_t_test[start])
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
