from dataclasses import dataclass

import numpy as np


@dataclass
class Band:
    name: str
    center: float
    width: float
    type: str = 'flat'

    @classmethod
    def from_passband(cls, name, nu, pb, pb_err=None):
        npb = pb / np.sum(pb)

        center = np.round(np.sum(npb * nu), 3)
        width = np.round(nu[npb > 1e-2].ptp(), 3)

        band = cls(name=name, center=center, width=width, type='custom')

        band.nu = nu
        band.pb = npb

        return band

    def passband(self, nu):
        """
        Passband response as a function of nu (in GHz)
        """

        _nu = np.atleast_1d(nu)

        if self.type == 'flat':
            pb = (
                (_nu > self.center - 0.5 * self.width)
                & (_nu < self.center + 0.5 * self.width)
            ).astype(float)
            return pb / (pb.sum() + 1e-16)

        elif self.type == 'custom':
            return np.interp(_nu, self.nu, self.pb)
