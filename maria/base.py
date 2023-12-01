import os

from astropy.io import fits

from . import utils
from .array import Array, all_array_params, get_array
from .coordinator import Coordinator
from .pointing import POINTING_PARAMS, Pointing, get_pointing
from .site import SITE_PARAMS, Site, get_site
from .tod import TOD

here, this_filename = os.path.split(__file__)


class InvalidSimulationParameterError(Exception):
    def __init__(self, invalid_keys):
        super().__init__(
            f"The parameters {invalid_keys} are not valid simulation parameters!"
        )


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


master_params = utils.io.read_yaml(f"{here}/configs/params.yml")


class BaseSimulation:
    """
    The base class for a simulation. This is an ingredient in every simulation.
    """

    def __init__(self, array, pointing, site, verbose=False, **kwargs):
        # who does each kwarg belong to?

        self.verbose = verbose

        parsed_sim_kwargs = parse_sim_kwargs(kwargs, master_params)

        self.array = (
            array
            if isinstance(array, Array)
            else get_array(array, **parsed_sim_kwargs["array"])
        )
        self.pointing = (
            pointing
            if isinstance(pointing, Pointing)
            else get_pointing(pointing, **parsed_sim_kwargs["pointing"])
        )
        self.site = (
            site
            if isinstance(site, Site)
            else get_site(site, **parsed_sim_kwargs["site"])
        )

        self.coordinator = Coordinator(lat=self.site.latitude, lon=self.site.longitude)

        if self.pointing.pointing_frame == "az_el":
            self.pointing.ra, self.pointing.dec = self.coordinator.transform(
                self.pointing.time,
                self.pointing.az,
                self.pointing.el,
                in_frame="az_el",
                out_frame="ra_dec",
            )

        if self.pointing.pointing_frame == "ra_dec":
            self.pointing.az, self.pointing.el = self.coordinator.transform(
                self.pointing.time,
                self.pointing.ra,
                self.pointing.dec,
                in_frame="ra_dec",
                out_frame="az_el",
            )

    @property
    def params(self):
        _params = {"array": {}, "pointing": {}, "site": {}}
        for key in all_array_params:
            _params["array"][key] = getattr(self.array, key)
        for key in POINTING_PARAMS:
            _params["pointing"][key] = getattr(self.pointing, key)
        for key in SITE_PARAMS:
            _params["site"][key] = getattr(self.site, key)
        return _params

    def _run(self):
        raise NotImplementedError()

    def run(self):
        self._run()

        tod = TOD()

        tod.data = self.data  # this should be set in the _run() method
        tod.time = self.pointing.time
        tod.az = self.pointing.az
        tod.el = self.pointing.el
        tod.ra = self.pointing.ra
        tod.dec = self.pointing.dec
        tod.cntr = self.pointing.scan_center
        tod.pntunit = self.pointing.pointing_units

        if hasattr(self, "map_sim"):
            if self.map_sim is not None:
                tod.unit = self.map_sim.input_map.units
                tod.header = self.map_sim.input_map.header
            else:
                tod.unit = "K"
                tod.header = fits.header.Header()

        tod.dets = self.array.dets

        tod.meta = {
            "latitude": self.site.latitude,
            "longitude": self.site.longitude,
            "altitude": self.site.altitude,
        }

        return tod
