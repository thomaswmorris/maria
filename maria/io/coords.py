import logging
import os

from ..units import Quantity

logger = logging.getLogger("maria")
here, this_filename = os.path.split(__file__)


def repr_lat_lon(lat, lon):
    lat_repr = Quantity(abs(lat), "deg").repr_angle("dms") + (" N" if lat > 0 else " S")
    lon_repr = Quantity(abs(lon), "deg").repr_angle("dms") + (" E" if lon > 0 else " W")
    return lat_repr, lon_repr


def repr_phi_theta(phi: float, theta: float, frame: str, join: bool = False):
    qphi = Quantity(phi, "rad")
    qtheta = Quantity(theta, "rad")
    if frame == "az/el":
        res = {"az": qphi.repr_angle("deg"), "el": qtheta.repr_angle("deg")}
    elif frame == "ra/dec":
        res = {"ra": qphi.repr_angle("hms"), "dec": qtheta.repr_angle("dms")}
    elif frame == "galactic":
        res = {"l": qphi.repr_angle("deg"), "b": qtheta.repr_angle("deg")}
    else:
        raise ValueError(f"Invalid frame '{frame}'")

    if join:
        res = (f"{key}: {value}" for key, value in res.items())

    return res
