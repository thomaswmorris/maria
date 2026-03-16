from collections.abc import Sequence

import numpy as np
import scipy as sp
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm, trange

from ..map import ProjectionMap
from ..tod import TOD
from ..utils import decompose
from .base import BaseProjectionMapper
from .bin_mapper import BinMapper


def remove_ends(data, ntip=16):
    start_values = data[..., :ntip].mean(axis=-1)
    end_values = data[..., -ntip:].mean(axis=-1)

    return data - np.linspace(start_values, end_values, data.shape[-1]).T


class MaximumLikelihoodSolution(torch.nn.Module):
    def __init__(self, data: torch.Tensor, mask: torch.Tensor, dtype: type = torch.float):
        super(MaximumLikelihoodSolution, self).__init__()

        data_std = data[mask].detach().std()
        data_std = data_std if data_std > 0 else torch.tensor(1.0)
        self.log_scale = torch.nn.Parameter(data_std.log())
        self.unscaled_x = torch.nn.Parameter(data[mask] / data_std)
        self.mask = mask

    def x(self):
        x = torch.zeros_like(self.mask, dtype=torch.float)
        x[self.mask] = self.log_scale.exp() * self.unscaled_x
        return x


class MaximumLikelihoodSolution(torch.nn.Module):
    def __init__(self, data: torch.Tensor, mask: torch.Tensor = None, dtype: type = torch.float):
        super(MaximumLikelihoodSolution, self).__init__()

        mask = mask if mask is not None else torch.ones_like(data, dtype=torch.bool)

        data_std = data[mask].detach().std()
        data_std = data_std if data_std > 0 else torch.tensor(1.0)
        self.log_scale = torch.nn.Parameter(data_std.log())
        self.unscaled_x = torch.nn.Parameter(data[mask] / data_std)
        self.mask = mask

    def x(self):
        x = torch.zeros_like(self.mask, dtype=torch.float)
        x[self.mask] = self.log_scale.exp() * self.unscaled_x
        return x


class MaximumLikelihoodMapper(BaseProjectionMapper):
    def __init__(
        self,
        tods: Sequence[TOD],
        k: int = 3,
        center: tuple[float, float] = None,
        stokes: str = None,
        width: float = None,
        height: float = None,
        resolution: float = None,
        frame: str = "ra/dec",
        units: str = "K_RJ",
        degrees: bool = True,
        min_time: float = None,
        max_time: float = None,
        timestep: float = None,
        tod_preprocessing: dict = {
            # "remove_modes": {"modes_to_remove": 1},
            "remove_spline": {"knot_spacing": 60, "remove_el_gradient": True},
        },
        map_postprocessing: dict = {},
        progress_bars: bool = True,
    ):
        super().__init__(
            tods=tods,
            stokes=stokes,
            center=center,
            width=width,
            height=height,
            resolution=resolution,
            frame=frame,
            units=units,
            tod_preprocessing=tod_preprocessing,
            map_postprocessing=map_postprocessing,
            min_time=min_time,
            max_time=max_time,
            timestep=timestep,
            degrees=degrees,
            progress_bars=progress_bars,
        )

        # if center is None:
        #     center = np.degrees(get_center_phi_theta(*np.stack([tod.coords.center(frame="ra/dec") for tod in tods]).T))

        self.map_init_sigma = 0

        self.iteration = 0

        self.k = k

        self.binner = BinMapper(
            center=self.center,
            stokes=self.stokes,
            width=self.width,
            height=self.height,
            resolution=self.resolution,
            frame=self.frame,
            tods=self.tods,
            units=self.units,
            min_time=min_time,
            max_time=max_time,
            timestep=timestep,
            degrees=False,
            # tod_preprocessing=tod_preprocessing,
            # map_postprocessing=map_postprocessing,
            progress_bars=False,
        )

        # # self.update_deprojection()

        self.sol = MaximumLikelihoodSolution(data=torch.zeros(self.map_shape))

        self.P_list = []

        self.hits = torch.zeros(np.prod(self.map_shape))

        for tod in self.tods:
            values, pixel_index, sample_index = self.map.compute_pointing_matrix_sparse_indices(
                coords=tod.coords, dets=tod.dets
            )
            indices = torch.stack(
                [torch.tensor(sample_index, dtype=torch.long), torch.tensor(pixel_index, dtype=torch.long)], dim=0
            )
            P = torch.sparse_coo_tensor(
                indices=indices, values=torch.tensor(values, dtype=torch.float), size=(tod.coords.size, self.map_size)
            )

            self.P_list.append(P)
            self.hits += P.sum(axis=0).to_dense()

        naive_sol = self.naive_map.ravel()
        init_sol = naive_sol * (self.hits / self.hits.max()).sqrt()

        self.mask = ~init_sol.isnan()
        self.sol = MaximumLikelihoodSolution(data=init_sol.float(), mask=self.mask)

    def update_noise_model(self):
        self.tod_list = []

        pbar = tqdm(enumerate(self.tods), desc="Updating noise model")

        for tod_index, tod in pbar:
            pbar.set_postfix(tod=f"{tod_index + 1}/{len(self.tods)}")

            t = {"tod": tod}
            t["nd"], t["nt"] = tod.shape

            d = sp.signal.detrend(tod.signal.compute())

            t["d"] = torch.tensor(d, dtype=torch.float)

            t["w"] = torch.tensor(sp.signal.windows.tukey(M=t["d"].shape[-1], alpha=0.5), dtype=torch.float)
            t["f"] = np.fft.fftfreq(n=t["d"].shape[-1], d=1 / t["tod"].sample_rate)
            t["abs_f"] = np.abs(t["f"])
            t["P"] = self.P_list[tod_index]

            # project the fitted map out of the data for noise modeling
            t["d_no_map"] = t["d"] - (t["P"] @ self.sol.x().detach()).reshape(t["d"].shape).detach()
            t["wd"] = t["w"] * t["d_no_map"]
            t["fwd"] = torch.fft.fft(t["wd"])

            if self.k > 0:
                t["a"], t["b"] = [torch.tensor(_) for _ in decompose(t["fwd"].numpy(), k=self.k, norm="var")]
                t["U"] = t["a"].T.unsqueeze(-1)
                t["noise"] = t["fwd"] - t["a"] @ t["b"]

            else:
                t["noise"] = t["fwd"]

            t["raw_ps"] = t["noise"].square().abs()
            # ps = sp.ndimage.gaussian_filter(t["raw_ps"], sigma=2, axes=-1)
            # t["A_inv"] = torch.tensor(1 / ps)
            t["A_inv"] = 1 / t["raw_ps"].mean(dim=0).unsqueeze(0)

            if self.k > 0:
                t["Q"] = t["a"].T.unsqueeze(-1)
                t["C_QAQ_inv"] = torch.linalg.inv(
                    torch.eye(self.k) - ((t["A_inv"] * t["Q"]).sum(axis=-1) @ t["Q"]).squeeze(-1)
                )

            t["PNd"] = t["P"].T @ self.apply_inverse_noise_covariance(t["d"], t)

            self.tod_list.append(t)

    def apply_inverse_noise_covariance(self, d, t):
        fwd = torch.fft.fft(t["w"] * d)
        Nfwd = t["A_inv"] * fwd

        if self.k > 0:
            y = t["U"] * t["A_inv"] * fwd
            y = torch.tensordot(t["C_QAQ_inv"], y, dims=((-1,), (0,)))
            Nfwd -= t["A_inv"] * (y * t["U"]).sum(dim=0)

        return torch.fft.ifft(Nfwd).real.ravel()

    @property
    def naive_map(self):
        # return 1e-16 * np.random.standard_normal(self.map_shape).ravel()
        if not hasattr(self, "_naive_map"):
            self._naive_map = self.binner.run()
        return torch.tensor(self._naive_map.data.compute().ravel())

        # return sp.ndimage.median_filter(self._naive_map.data.compute(), size=3, axes=(-2, -1)).ravel()

    def forward(self, t):
        return t["P"].T @ self.apply_inverse_noise_covariance((t["P"] @ self.sol.x()).reshape(t["d"].shape), t)

    def loss(self):
        return sum([(self.forward(t) - t["PNd"]).square().sum() for t in self.tod_list])

    @property
    def map(self):
        return ProjectionMap(
            data=self.sol.x().detach().numpy().reshape(self.map_shape),
            weight=self.hits.numpy().reshape(self.map_shape),
            stokes=self.stokes,
            t=self.t,
            nu=self.nu,
            resolution=self.resolution,
            center=self.center,
            degrees=False,
            frame=self.frame.name,
            units=self.tod_units,
            beam=self.beam,
        ).to(self.map_units)

    def fit(self, epochs: int = 4, steps_per_epoch: int = 64, lr: float = 1e-1, plot: bool = False):
        if not hasattr(self, "optimizer") or lr != getattr(self, "lr", None):
            self.lr = lr
            self.optimizer = torch.optim.Adam(self.sol.parameters(), lr=lr)

        self.sol.train()

        for epoch in range(epochs):
            self.update_noise_model()

            pbar = trange(
                steps_per_epoch,
                bar_format="{l_bar}{bar:16}{r_bar}{bar:-10b}",
                desc=f"Fitting map (epoch {epoch + 1}/{epochs})",
                ncols=250,
            )
            pbar.set_postfix(loss=None)

            try:
                for step in pbar:
                    loss = self.loss()
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    pbar.set_postfix(loss=f"{loss.item():.03e}")

            except KeyboardInterrupt:
                break

            if plot:
                self.map.plot(cmap="cmb")
                plt.show()
