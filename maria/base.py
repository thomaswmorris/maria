import os

from . import utils
from .array import Array, get_array
from .coords import Coordinates, dx_dy_to_phi_theta
from .pointing import Pointing, get_pointing
from .site import Site, get_site
from .tod import TOD

here, this_filename = os.path.split(__file__)


class InvalidSimulationParameterError(Exception):
    def __init__(self, invalid_keys):
        super().__init__(
            f"The parameters {invalid_keys} are not valid simulation parameters!"
        )


master_params = utils.io.read_yaml(f"{here}/configs/params.yml")


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
        array: Array or str = "default",
        pointing: Pointing or str = "stare",
        site: Site or str = "default",
        verbose=False,
        **kwargs,
    ):
        if hasattr(self, "boresight"):
            return

        self.verbose = verbose

        self.data = {}

        parsed_sim_kwargs = parse_sim_kwargs(kwargs, master_params)

        if type(array) is Array:
            self.array = array
        else:
            self.array = get_array(array_name=array, **parsed_sim_kwargs["array"])

        if type(pointing) is Pointing:
            self.pointing = pointing
        else:
            self.pointing = get_pointing(
                scan_pattern=pointing, **parsed_sim_kwargs["pointing"]
            )

        if type(site) is Site:
            self.site = site
        else:
            self.site = get_site(site_name=site, **parsed_sim_kwargs["site"])

        self.boresight = Coordinates(
            self.pointing.time,
            self.pointing.phi,
            self.pointing.theta,
            location=self.site.earth_location,
            frame=self.pointing.pointing_frame,
        )

        det_az, det_el = dx_dy_to_phi_theta(
            *self.array.offsets.T[..., None], self.boresight.az, self.boresight.el
        )

        self.det_coords = Coordinates(
            self.boresight.time,
            det_az,
            det_el,
            location=self.site.earth_location,
            frame="az_el",
        )

    def _run(self):
        raise NotImplementedError()

    def run(self):
        self._run()

        tod = TOD(
            data=self.data,
            dets=self.array.dets,
            boresight=self.boresight,
            coords=self.det_coords,
        )

        return tod
