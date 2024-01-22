from dataclasses import dataclass

import numpy as np


@dataclass
class Band:
    name: str
    center: float
    width: float
    type: str = "flat"

    @classmethod
    def from_passband(cls, name, nu, pb, pb_err=None):
        center = np.round(np.sum(pb * nu), 3)
        width = np.round(
            nu[pb > pb.max() / np.e**2].ptp(), 3
        )  # width is the two-sigma interval

        band = cls(name=name, center=center, width=width, type="custom")

        band._nu = nu
        band._pb = pb

        return band

    @property
    def nu_min(self):
        if self.type == "flat":
            return self.center - 0.5 * self.width
        if self.type == "gaussian":
            return self.center - self.width
        if self.type == "custom":
            return self._nu[self._pb > 1e-2 * self._pb.max()].min()

    @property
    def nu_max(self):
        if self.type == "flat":
            return self.center + 0.5 * self.width
        if self.type == "gaussian":
            return self.center + self.width
        if self.type == "custom":
            return self._nu[self._pb > 1e-2 * self._pb.max()].max()

    def passband(self, nu):
        """
        Passband response as a function of nu (in GHz). These integrate to one.
        """

        _nu = np.atleast_1d(nu)

        if self.type == "gaussian":
            band_sigma = self.width / 4
            return np.exp(-0.5 * np.square((_nu - self.center) / band_sigma))

        if self.type == "flat":
            return np.where(
                (_nu > self.nu_min) & (_nu < self.nu_max), 1 / self.width, 0
            )

        elif self.type == "custom":
            return np.interp(_nu, self._nu, self._pb)
