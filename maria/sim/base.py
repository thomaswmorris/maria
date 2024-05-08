import os

import dask.array as da
import numpy as np

from .. import utils
from ..coords import Coordinates, dx_dy_to_phi_theta
from ..instrument import Instrument, get_instrument
from ..plan import Plan, get_plan
from ..site import Site, get_site
from ..tod import TOD

here, this_filename = os.path.split(__file__)


class InvalidSimulationParameterError(Exception):
    def __init__(self, invalid_keys):
        super().__init__(
            f"The parameters {invalid_keys} are not valid simulation parameters!"
        )


master_params = utils.io.read_yaml(f"{here}/params.yml")


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
        instrument: Instrument or str = "default",
        plan: Plan or str = "stare",
        site: Site or str = "default",
        verbose=False,
        **kwargs,
    ):
        if hasattr(self, "boresight"):
            return

        self.verbose = verbose

        parsed_sim_kwargs = parse_sim_kwargs(kwargs, master_params)

        if isinstance(instrument, Instrument):
            self.instrument = instrument
        else:
            self.instrument = get_instrument(
                instrument_name=instrument, **parsed_sim_kwargs["instrument"]
            )

        if isinstance(plan, Plan):
            self.plan = plan
        else:
            self.plan = get_plan(scan_pattern=plan, **parsed_sim_kwargs["plan"])

        if isinstance(site, Site):
            self.site = site
        elif isinstance(site, str):
            self.site = get_site(site_name=site, **parsed_sim_kwargs["site"])
        else:
            raise ValueError(
                "The passed site must be either a Site object or a string."
            )

        self.data = {}
        self.calibration = np.ones((self.instrument.dets.n, self.plan.n_time))

        self.boresight = Coordinates(
            self.plan.time - self.plan.time.min(),
            self.plan.phi,
            self.plan.theta,
            location=self.site.earth_location,
            frame=self.plan.pointing_frame,
            time_offset=self.plan.time.min(),
        )

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

        det_az, det_el = dx_dy_to_phi_theta(
            *self.instrument.offsets.T[..., None], self.boresight.az, self.boresight.el
        )

        self.coords = Coordinates(
            self.boresight.time,
            det_az,
            det_el,
            location=self.site.earth_location,
            frame="az_el",
            time_offset=self.boresight.time_offset,
        )

    def _run(self):
        raise NotImplementedError()

    def run(self, dtype=np.float32):
        self.data = {}

        # Simulate all the junk
        self._run()

        tod = TOD(
            components={
                k: da.from_array(v.astype(dtype)) for k, v in self.data.items()
            },
            dets=self.instrument.dets.df,
            coords=self.coords,
        )

        return tod
