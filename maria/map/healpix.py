import healpy as hp

from .base import Map


class HEALPixMap(Map):
    """
    A HEALPix Map.
    """

    def __init__(
        self,
        data: float,
        nu: float = None,
        t: float = None,
        weight: float = None,
        width: float = None,
        resolution: float = None,
        center: tuple[float, float] = (0.0, 0.0),
        frame: str = "ra_dec",
        degrees: bool = True,
        units: str = "K_RJ",
    ):
        ...

        self.nside = hp.pixelfunc.npix2nside(len(data))
        self.data = data
