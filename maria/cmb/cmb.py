from ..map import HEALPixMap


class CMB(HEALPixMap):
    def __init__(
        self,
        data: float,
        weight: float = None,
        stokes: float = None,
        frame: str = "galactic",
        nu: float = None,
        units: str = "K_CMB",
    ):
        super().__init__(data=data, weight=weight, stokes=stokes, nu=nu, z=1100.0, units=units, frame=frame)
