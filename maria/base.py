import os

from astropy.io import fits

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

        # self.coordinator = Coordinator(lat=self.site.latitude, lon=self.site.longitude)

        # if self.pointing.pointing_frame == "az_el":
        #     self.pointing.ra, self.pointing.dec = self.coordinator.transform(
        #         self.pointing.time,
        #         self.boresight.az,
        #         self.boresight.el,
        #         in_frame="az_el",
        #         out_frame="ra_dec",
        #     )

        # if self.pointing.pointing_frame == "ra_dec":
        #     self.boresight.az, self.boresight.el = self.coordinator.transform(
        #         self.pointing.time,
        #         self.pointing.ra,
        #         self.pointing.dec,
        #         in_frame="ra_dec",
        #         out_frame="az_el",
        #     )

    # @property
    # def params(self):
    #     _params = {"array": {}, "pointing": {}, "site": {}}
    #     for key in all_array_params:
    #         _params["array"][key] = getattr(self.array, key)
    #     for key in POINTING_PARAMS:
    #         _params["pointing"][key] = getattr(self.pointing, key)
    #     for key in SITE_PARAMS:
    #         _params["site"][key] = getattr(self.site, key)
    #     return _params

    def _run(self):
        raise NotImplementedError()

    def run(self):
        self._run()

        tod = TOD()

        tod._data = self.data

        det_az, det_el = dx_dy_to_phi_theta(
            self.array.dets.offset_x.values[:, None],
            self.array.dets.offset_y.values[:, None],
            self.boresight.az,
            self.boresight.el,
        )

        tod.coords = Coordinates(
            time=self.boresight.time,
            phi=det_az,
            theta=det_el,
            frame="az_el",
            location=self.site.earth_location,
        )

        tod.boresight = self.boresight
        tod.dets = self.array.dets

        tod.time = self.boresight.time
        tod.cntr = self.pointing.scan_center
        # tod.pntunit = self.pointing.pointing_units

        if self.map_file:
            tod.unit = self.input_map.units
            tod.header = self.input_map.header
        else:
            tod.unit = "K"
            tod.header = fits.header.Header()

        tod.meta = {
            "latitude": self.site.latitude,
            "longitude": self.site.longitude,
            "altitude": self.site.altitude,
        }

        return tod
