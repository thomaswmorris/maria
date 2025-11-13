import logging
import os

from ..units import Quantity

logger = logging.getLogger("maria")
here, this_filename = os.path.split(__file__)


def repr_lat_lon(lat, lon):
    lat_repr = Quantity(abs(lat), "deg").repr("dms") + (" N" if lat > 0 else " S")
    lon_repr = Quantity(abs(lon), "deg").repr("dms") + (" E" if lon > 0 else " W")
    return lat_repr, lon_repr


def repr_phi_theta(phi: float, theta: float, frame: str, join: bool = False):
    qphi = Quantity(phi, "rad")
    qtheta = Quantity(theta, "rad")
    if frame == "az/el":
        res = {"az": qphi.repr("deg"), "el": qtheta.repr("deg")}
    elif frame == "ra/dec":
        res = {"ra": qphi.repr("hms"), "dec": qtheta.repr("dms")}
    elif frame == "galactic":
        res = {"l": qphi.repr("deg"), "b": qtheta.repr("deg")}
    else:
        raise ValueError(f"Invalid frame '{frame}'")

    if join:
        res = (f"{key}: {value}" for key, value in res.items())

    return res
