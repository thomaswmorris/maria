import logging
import os
from typing import Union

import numpy as np

from ..coords import Coordinates
from ..instrument import Instrument, get_instrument
from ..io import read_yaml
from ..plan import Plan, get_plan
from ..site import Site, get_site
from ..tod import TOD

here, this_filename = os.path.split(__file__)

logger = logging.getLogger("maria")


class InvalidSimulationParameterError(Exception):
    def __init__(self, invalid_keys):
        super().__init__(
            f"The parameters {invalid_keys} are not valid simulation parameters!"
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
                invalid_keys=list(invalid_kwargs.keys())
            )

    return parsed_kwargs


class BaseSimulation:
    """
    The base class for a simulation. This is an ingredient in every simulation.
    """

    def __init__(
        self,
        instrument: Union[Instrument, str],
        plan: Union[Plan, str],
        site: Union[Site, str],
        verbose=False,
        dtype=np.float32,
        **kwargs,
    ):
        if hasattr(self, "boresight"):
            return

        self.dtype = dtype
        self.verbose = verbose

        parsed_sim_kwargs = parse_sim_kwargs(kwargs, master_params)

        if isinstance(instrument, Instrument):
            self.instrument = instrument
        else:
            self.instrument = get_instrument(
                name=instrument, **parsed_sim_kwargs["instrument"]
            )

        logger.debug("Constructed instrument.")

        if isinstance(plan, Plan):
            self.plan = plan
        else:
            self.plan = get_plan(plan_name=plan, **parsed_sim_kwargs["plan"])

        logger.debug("Constructed plan.")

        if isinstance(site, Site):
            self.site = site
        elif isinstance(site, str):
            self.site = get_site(site_name=site, **parsed_sim_kwargs["site"])
        else:
            raise ValueError(
                "The passed site must be either a Site object or a string."
            )

        logger.debug("Constructed site.")

        self.data = {}

        self.boresight = Coordinates(
            time=self.plan.time,
            phi=self.plan.phi,
            theta=self.plan.theta,
            earth_location=self.site.earth_location,
            frame=self.plan.frame,
        )

        logger.debug("Constructed boresight.")

        if self.plan.max_vel > np.radians(self.instrument.vel_limit):
            raise ValueError(
                (
                    f"The maximum velocity of the boresight ({np.degrees(self.plan.max_vel):.01f} deg/s) exceeds "
                    f"the maximum velocity of the instrument ({self.instrument.vel_limit:.01f} deg/s)."
                ),
            )

        if self.plan.max_acc > np.radians(self.instrument.acc_limit):
            raise ValueError(
                (
                    f"The maximum acceleration of the boresight ({np.degrees(self.plan.max_acc):.01f} deg/s^2) exceeds "
                    f"the maximum acceleration of the instrument ({self.instrument.acc_limit:.01f} deg/s^2)."
                ),
            )

        # this can be expensive sometimes
        self.coords = self.boresight.broadcast(
            self.instrument.dets.offsets, frame="az_el"
        )

        logger.debug("Constructed offsets.")

    def _run(self):
        raise NotImplementedError()

    def run(self):
        self.data = {}

        # Simulate all the junk
        self._run()

        tod_data = {}

        for k, data in self.data.items():
            # scaling floats doesn't make them more accurate, unless they're huge or tiny
            offset = data.mean(axis=-1)
            # scale = data.std(axis=-1)[..., None]
            tod_data[k] = {
                "data": (data - offset[..., None]).astype(self.dtype),
                "offset": offset,
            }

        tod = TOD(
            data=tod_data,
            dets=self.instrument.dets,
            coords=self.coords,
            units="pW",
        )

        return tod
