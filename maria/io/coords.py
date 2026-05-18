import logging
import os

from ..units import Quantity

logger = logging.getLogger("maria")
here, this_filename = os.path.split(__file__)


def repr_lat_lon(lat, lon):
    lat_repr = Quantity(abs(lat), "deg").dms + (" N" if lat > 0 else " S")
    lon_repr = Quantity(abs(lon), "deg").dms + (" E" if lon > 0 else " W")
    return lat_repr, lon_repr


def repr_phi_theta(phi: float, theta: float, frame: str, join: bool = False):
    qphi = Quantity(phi, "rad")
    qtheta = Quantity(theta, "rad")
    if frame == "az/el":
        res = {"az": qphi.deg, "el": qtheta.deg}
    elif frame == "ra/dec":
        res = {"ra": qphi.hms, "dec": qtheta.dms}
    elif frame == "galactic":
        res = {"glon": qphi.deg, "glat": qtheta.deg}
    else:
        raise ValueError(f"Invalid frame '{frame}'")

    if join:
        res = (f"{key}: {value}" for key, value in res.items())

    return res
