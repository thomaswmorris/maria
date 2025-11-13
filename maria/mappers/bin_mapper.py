from __future__ import annotations

import logging
import os
import time as ttime
from collections.abc import Sequence

import numpy as np
from tqdm import tqdm

from ..coords import infer_center_width_height
from ..io import humanize_time, repr_phi_theta
from ..tod import TOD
from ..units import Quantity
from .base import BaseProjectionMapper

np.seterr(invalid="ignore")

here, this_filename = os.path.split(__file__)
logger = logging.getLogger("maria")


class BinMapper(BaseProjectionMapper):
    def __init__(
        self,
        tods: Sequence[TOD],
        center: tuple[float, float] = None,
        stokes: str = None,
        width: float = None,
        height: float = None,
        resolution: float = None,
        frame: str = "ra/dec",
        units: str = "K_RJ",
        degrees: bool = True,
        calibrate: bool = False,
        min_time: float = None,
        max_time: float = None,
        timestep: float = None,
        tod_preprocessing: dict = {},
        map_postprocessing: dict = {},
    ):
        if not tods:
            raise ValueError("You must pass at least one TOD to the mapper!")

        center = (Quantity(center, "deg" if degrees else "rad")) if center is not None else None
        width = (Quantity(width, "deg" if degrees else "rad")) if width is not None else None
        height = (Quantity(height, "deg" if degrees else "rad")) if height is not None else width
        resolution = (Quantity(resolution, "deg" if degrees else "rad")) if resolution is not None else None

        infer_center, infer_width, infer_height = infer_center_width_height(
            coords_list=[tod.coords for tod in tods], center=center, frame=frame, square=True
        )

        logger.debug(
            f"Inferred center={Quantity(infer_center, 'rad')}, width={Quantity(infer_width, 'rad')}, \
                     width={Quantity(infer_height, 'rad')} for map."
        )

        if center is None:
            center = Quantity(infer_center, "rad")
            logger.info(
                f"Inferring center {repr_phi_theta(phi=center[0].rad, theta=center[1].rad, frame=frame)} for mapper."
            )

        if width is None:
            width = Quantity(infer_width, "rad")
            logger.info(f"Inferring width {width} for mapper.")

        if height is None:
            height = Quantity(infer_height, "rad")
            logger.info(f"Inferring height {height} for mapper.")

        if resolution is None:
            resolution = Quantity(width / 100, "rad")
            logger.info(f"Inferring resolution {resolution} for mapper.")

        if stokes is None:
            stokes = "IQUV" if any([tod.dets.polarized for tod in tods]) else "I"
            logger.info(f"Inferring stokes parameters '{stokes}' for mapper.")

        min_time = min_time or min([tod.coords.t.min() for tod in tods])
        max_time = max_time or max([tod.coords.t.max() for tod in tods])

        super().__init__(
            stokes=stokes,
            center=Quantity(center, "rad"),
            width=Quantity(width, "rad"),
            height=Quantity(height, "rad"),
            resolution=Quantity(resolution, "rad"),
            frame=frame,
            calibrate=calibrate,
            tods=tods,
            units=units,
            tod_preprocessing=tod_preprocessing,
            map_postprocessing=map_postprocessing,
            min_time=min_time,
            max_time=max_time,
            timestep=timestep,
        )

    def initialize_mapper(self):
        return

    def _run(self):
        """
        The actual mapper for the BinMapper.
        """

        self.map_data = {}

        map_sum = np.zeros((self.n_stokes, len(self.map.nu), len(self.map.t) * self.n_y * self.n_x))
        map_wgt = np.zeros((self.n_stokes, len(self.map.nu), len(self.map.t) * self.n_y * self.n_x))

        nu_list = []

        for band_index, band in enumerate(self.bands):
            band_start_s = ttime.monotonic()

            nu_list.append(band.center)

            pbar = tqdm(
                enumerate(self.tods),
                total=len(self.tods),
                desc=f"Mapping band {band.name}",
                postfix={"tod": f"1/{len(self.tods)}", "stokes": "I"},
            )

            for tod_index, tod in pbar:
                band_tod = tod.subset(band=band.name)

                if not tod.shape[0] > 0:
                    continue

                pmat = self.map.pointing_matrix(coords=band_tod.coords)
                D = band_tod.signal.compute()
                W = band_tod.weight.compute()

                stokes_weight = band_tod.dets.stokes_weight()

                for stokes_index, stokes in enumerate(self.stokes):
                    pbar.set_postfix(tod=f"{tod_index + 1}/{len(self.tods)}", stokes=stokes)

                    s = stokes_weight[:, "IQUV".index(stokes)][..., None]

                    map_sum[stokes_index, band_index] += (np.sign(s) * W * D).ravel() @ pmat
                    map_wgt[stokes_index, band_index] += (np.abs(s) * W).ravel() @ pmat

                del band_tod

            logger.info(f"Ran mapper for band {band.name} in {humanize_time(ttime.monotonic() - band_start_s)}.")

        return {
            "sum": map_sum.reshape(self.n_stokes, self.n_bands, len(self.map.t), self.n_y, self.n_x),
            "wgt": map_wgt.reshape(self.n_stokes, self.n_bands, len(self.map.t), self.n_y, self.n_x),
            "stokes": self.stokes,
            "nu": nu_list,
        }
