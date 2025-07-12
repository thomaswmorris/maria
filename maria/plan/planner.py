import arrow
import numpy as np

from ..coords import Coordinates
from ..plan import Plan
from ..site import Site, get_site
from ..units import Quantity
from ..utils.signal import grouper


class Planner:
    def __init__(
        self,
        target,
        site: Site,
        az_bounds: tuple = (0.0, 360.0),
        el_bounds: tuple = (30.0, 90.0),
        hour_bounds: tuple = (0.0, 24.0),
        max_lookahead: float = 86400.0,
        start_time: float = arrow.get(),
    ):
        if isinstance(site, Site):
            self.site = site
        elif isinstance(site, str):
            self.site = get_site(site_name=site)
        else:
            raise ValueError(
                "'site' must be either a Site object or a string.",
            )

        self.target = target
        self.bounds = {
            "az": Quantity(az_bounds, "deg"),
            "el": Quantity(el_bounds, "deg"),
            "hour": Quantity(hour_bounds, "hour"),
        }
        self.start_time = arrow.get(start_time)
        self.max_lookahead = max_lookahead

    def test_az_el(self, az, el):
        return (
            (az > self.bounds["az"][0].rad)
            & (az < self.bounds["az"][1].rad)
            & (el > self.bounds["el"][0].rad)
            & (el < self.bounds["el"][1].rad)
        )

    def generate_obs_intervals(self, total_duration: float, chunk_duration: float = None):
        chunk_duration = chunk_duration or total_duration

        horizon_resolution = 600.0

        self.t = t = self.start_time.timestamp() + np.arange(0, self.max_lookahead, horizon_resolution)

        self.c = Coordinates(
            phi=self.target.center[0].radians,
            theta=self.target.center[1].radians,
            t=t,
            frame=self.target.frame,
            earth_location=self.site.earth_location,
        )

        chunks = []

        c_mask = self.test_az_el(self.c.az, self.c.el)

        duration = 0
        for start, end in grouper(c_mask, chunk_duration / horizon_resolution):
            chunk_start_time = t[start]

            while (chunk_start_time < t[end]) and (duration < total_duration):
                chunk_duration = np.min([total_duration - duration, chunk_duration, t[end] - t[start]])
                chunks.append({"start_time": arrow.get(chunk_start_time), "duration": chunk_duration})
                chunk_start_time += chunk_duration
                duration += chunk_duration

            if duration >= total_duration:
                break

        return chunks

    def generate_plans(self, total_duration: float, chunk_duration: float = None, **plan_kwargs):
        chunks = self.generate_obs_intervals(total_duration=total_duration, chunk_duration=chunk_duration)

        return [
            Plan(
                scan_center=self.target.center,
                frame=self.target.frame,
                latitude=self.site.latitude,
                longitude=self.site.longitude,
                altitude=self.site.altitude,
                **chunk,
                **plan_kwargs,
            )
            for chunk in chunks
        ]

    def generate_plan(self, total_duration: float, **plan_kwargs):
        return self.generate_plans(total_duration=total_duration, chunk_duration=None, **plan_kwargs)[0]
