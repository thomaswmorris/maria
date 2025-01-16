from __future__ import annotations

import arrow
import logging
import os

import numpy as np
import time as ttime

from ..coords import Coordinates
from ..instrument import Instrument, get_instrument
from ..utils import read_yaml
from ..io import humanize_time
from ..plan import Plan, get_plan
from ..site import Site, get_site
from ..tod import TOD

here, this_filename = os.path.split(__file__)

logger = logging.getLogger("maria")


class InvalidSimulationParameterError(Exception):
    def __init__(self, invalid_keys):
        super().__init__(
            f"The parameters {invalid_keys} are not valid simulation parameters!",
        )


master_params = read_yaml(f"{here}/params.yml")


def parse_sim_kwargs(kwargs, master_kwargs, strict=False):
    parsed_kwargs = {k: {} for k in master_kwargs.keys()}
    invalid_kwargs = {}

    for k, v in kwargs.items():
        parsed = False
        for sub_type, sub_kwargs in master_kwargs.items():
            if k in sub_kwargs.keys():
                parsed_kwargs[sub_type][k] = v
                parsed = True
        if not parsed:
            invalid_kwargs[k] = v

    if len(invalid_kwargs) > 0:
        if strict:
            raise InvalidSimulationParameterError(
                invalid_keys=list(invalid_kwargs.keys()),
            )

    return parsed_kwargs


class BaseSimulation:
    """
    The base class for a simulation. This is an ingredient in every simulation.
    """

    def __init__(
        self,
        instrument: Instrument | str,
        plan: Plan | str,
        site: Site | str,
        progress_bars: bool = True,
        dtype=np.float32,
        **kwargs,
    ):

        start_init_s = ttime.monotonic()

        if hasattr(self, "boresight"):
            return

        self.dtype = dtype
        self.disable_progress_bars = not progress_bars
        parsed_sim_kwargs = parse_sim_kwargs(kwargs, master_params)

        if isinstance(instrument, Instrument):
            self.instrument = instrument
        else:
            self.instrument = get_instrument(
                name=instrument,
                **parsed_sim_kwargs["instrument"],
            )

        logger.debug(
            f"Initialized instrument in {humanize_time(ttime.monotonic() - start_init_s)}."
        )
        instrument_init_s = ttime.monotonic()

        if isinstance(plan, Plan):
            self.plan = plan
        else:
            self.plan = get_plan(plan_name=plan, **parsed_sim_kwargs["plan"])

        logger.debug(
            f"Initialized plan in {humanize_time(ttime.monotonic() - instrument_init_s)}."
        )
        plan_init_s = ttime.monotonic()

        if isinstance(site, Site):
            self.site = site
        elif isinstance(site, str):
            self.site = get_site(site_name=site, **parsed_sim_kwargs["site"])
        else:
            raise ValueError(
                "The passed site must be either a Site object or a string.",
            )

        logger.debug(
            f"Initialized site in {humanize_time(ttime.monotonic() - plan_init_s)}."
        )
        site_init_s = ttime.monotonic()

        self.boresight = Coordinates(
            t=self.plan.time,
            phi=self.plan.phi,
            theta=self.plan.theta,
            earth_location=self.site.earth_location,
            frame=self.plan.frame,
        )

        logger.debug(
            f"Initialized boresight in {humanize_time(ttime.monotonic() - site_init_s)}."
        )
        boresight_init_s = ttime.monotonic()

        self.data = {}

        if self.plan.max_vel_deg > self.instrument.vel_limit:
            raise ValueError(
                (
                    f"The maximum velocity of the boresight ({self.plan.max_vel_deg:.01f} deg/s) exceeds "
                    f"the maximum velocity of the instrument ({self.instrument.vel_limit:.01f} deg/s)."
                ),
            )

        if self.plan.max_acc_deg > self.instrument.acc_limit:
            raise ValueError(
                (
                    f"The maximum acceleration of the boresight ({self.plan.max_acc_deg:.01f} deg/s^2) exceeds "
                    f"the maximum acceleration of the instrument ({self.instrument.acc_limit:.01f} deg/s^2)."
                ),
            )

        # this can be expensive sometimes
        self.coords = self.boresight.broadcast(
            self.instrument.dets.offsets,
            frame="az_el",
        )

        logger.debug(
            f"Initialized coordinates in {humanize_time(ttime.monotonic() - boresight_init_s)}."
        )

        logger.debug(
            f"Initialized generic simulation in {humanize_time(ttime.monotonic() - start_init_s)}."
        )

    def _run(self):
        raise NotImplementedError()

    def run(self):
        self.data = {}

        # Simulate all the junk
        self._run()

        metadata = {
            "sim_time": arrow.now(),
            "altitude": float(self.site.altitude),
            "region": self.site.region,
        }

        if hasattr(self, "atmosphere"):
            metadata["pwv"] = float(np.round(self.atmosphere.weather.pwv, 3))

        tod = TOD(
            data=self.data,
            dets=self.instrument.dets,
            coords=self.coords,
            units="pW",
            metadata=metadata,
        )

        return tod

    def plot_counts(self, x_bins=100, y_bins=100):
        self.plan.plot_counts(instrument=self.instrument, x_bins=x_bins, y_bins=y_bins)
