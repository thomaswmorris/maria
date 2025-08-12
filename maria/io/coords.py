import logging
import os

from ..units import Quantity

logger = logging.getLogger("maria")
here, this_filename = os.path.split(__file__)


def repr_lat_lon(lat, lon):
    lat_repr = Quantity(abs(lat), "deg").dms + (" N" if lat > 0 else " S")
    lon_repr = Quantity(abs(lon), "deg").dms + (" E" if lon > 0 else " W")
    return lat_repr, lon_repr


def repr_phi_theta(phi, theta, frame, join=None):
    qphi = Quantity(phi, "rad")
    qtheta = Quantity(theta, "rad")
    if frame == "az/el":
        res = (f"az: {qphi.deg:.02f}°", f"el: {qtheta.deg:.02f}°")
    elif frame == "ra/dec":
        res = (f"ra:  {qphi.hms}", f"dec: {qtheta.dms}")
    elif frame == "galactic":
        res = (f"l: {qphi.deg:.02f}°", f"b: {qtheta.deg:.02f}°")
    else:
        raise ValueError(f"Invalid frame '{frame}'")

    if join is not None:
        res = join.join(res)

    return res
