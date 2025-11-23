try:
    import torch

    torch.set_num_threads(1)
except ImportError:
    raise ImportError("To do maximum likelihood mapping in maria, please install PyTorch (pip install torch)")

from collections.abc import Sequence

import numpy as np
import scipy as sp
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


class MaximumLikelihoodMapper(BaseProjectionMapper):
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
        min_time: float = None,
        max_time: float = None,
        timestep: float = None,
        tod_preprocessing: dict = {},
        map_postprocessing: dict = {},
        k=0,
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
            tod_preprocessing={
                "remove_modes": {"modes_to_remove": [0, 1, 2]},
                "remove_spline": {"knot_spacing": 5, "remove_el_gradient": True},
            },
            map_postprocessing={},
            progress_bars=False,
        )

        # # self.update_deprojection()

        naive_sol = self.naive_map.ravel()

        init_sol = naive_sol
        # init_sol = 1e-5 * np.nanstd(naive_sol) * np.random.standard_normal(size=naive_sol.shape)

        self.mask = ~torch.tensor(init_sol).isnan()
        self.sol = MaximumLikelihoodSolution(data=torch.tensor(init_sol).float(), mask=self.mask)

        self.update_noise_model()

    def update_noise_model(self):
        sol_sum = 0
        sol_wgt = 0

        self.tod_list = []

        for tod_index, tod in tqdm(enumerate(self.tods), desc="Computing noise model"):
            t = {"tod": tod}

            t["k"] = max(self.k, 1)

            t["nd"], t["nt"] = tod.shape

            X = tod.dets.offsets
            rel_X = 2 * (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0)) - 1

            i, j = 0, 0
            mn_list = []
            while len(mn_list) < self.k:
                mn_list.append((i - j, j))
                j += 1
                if j > i:
                    i += 1
                    j = 0

            mode_list = []
            for m, n in mn_list:
                mode = sp.special.legendre(m)(rel_X[:, 0]) * sp.special.legendre(n)(rel_X[:, 1])
                mode_list.append(torch.tensor(mode, dtype=torch.complex64))

            t["a"] = torch.stack(mode_list, dim=1) if mode_list else torch.ones((t["nd"], 1), dtype=torch.complex64)

            ntip = int(0.5 * t["tod"].sample_rate)
            d = tod.signal.compute()
            d = remove_ends(d)
            # d = d - np.linspace(d[:, :ntip].mean(axis=-1), d[:, -ntip:].mean(axis=-1), d.shape[-1]).T

            # a, b = decompose(d, k=1)
            t["d"] = torch.tensor(d, dtype=torch.float)

            t["w"] = torch.tensor(sp.signal.windows.tukey(M=t["d"].shape[-1], alpha=0.1), dtype=torch.float)
            t["f"] = np.fft.fftfreq(n=t["d"].shape[-1], d=1 / t["tod"].sample_rate)
            t["abs_f"] = np.abs(t["f"])

            # def pointing_matrix(self, coords: Coordinates, dets: Array):
            #     values, pixel_index, sample_index = self.compute_pointing_matrix_sparse_indices(coords=coords, dets=dets)
            #     return sp.sparse.csr_array(
            #         (values, (sample_index, pixel_index)), shape=(coords.size, self.data.size)
            #     )

            # compute the pointing tensor (but the torch version)
            # sample_index, pixel_index, n_pixels = self.map.compute_pointing_matrix_sparse_indices(t["tod"].coords)
            values, pixel_index, sample_index = self.map.compute_pointing_matrix_sparse_indices(
                coords=t["tod"].coords, dets=t["tod"].dets
            )
            indices = torch.stack(
                [torch.tensor(sample_index, dtype=torch.long), torch.tensor(pixel_index, dtype=torch.long)], dim=0
            )
            t["P"] = torch.sparse_coo_tensor(
                indices=indices, values=torch.tensor(values, dtype=torch.float), size=(t["tod"].coords.size, self.map_size)
            )

            # project the fitted map out of the data
            t["d_no_map"] = t["d"] - (t["P"] @ self.sol.x()).reshape(t["d"].shape).detach()
            t["fd"] = torch.fft.fft(t["w"] * torch.tensor(remove_ends(t["d"].numpy())))
            # t["a"], t["b"] = [torch.tensor(x, dtype=torch.complex64) for x in decompose(t["fd"].numpy(), k=self.k)]
            # t["a"] = torch.ones((t["d"].shape[0], self.k), dtype=torch.complex64)

            # t["b"] = torch.linalg.inv(t["a"].T @ t["a"]) @ t["a"].T @ t["fd"]
            # if self.k > 0 else torch.zeros((1, t["nt"]), dtype=torch.complex64)
            # t["b"] = torch.tensor(sp.ndimage.gaussian_filter1d(t["b"], sigma=1, axis=-1))

            t["residual_ps"] = (t["fd"]).square().abs().float()

            n_bin = 32
            mid_f = np.geomspace(t["abs_f"][t["abs_f"] > 0].min() * 0.99, t["abs_f"].max() * 1.01, n_bin)
            dlogf = np.exp(np.gradient(np.log(mid_f)).mean())
            bin_f = np.geomspace(mid_f[0] / np.sqrt(dlogf), mid_f[-1] * np.sqrt(dlogf), n_bin + 1)
            bin_y = sp.stats.binned_statistic(t["abs_f"], t["residual_ps"].log(), bins=bin_f).statistic

            use = ~np.isnan(bin_y).any(axis=0)
            t["int_ps"] = torch.tensor(
                sp.interpolate.interp1d(
                    mid_f[use], np.median(bin_y[:, use], axis=0), axis=0, bounds_error=False, fill_value="extrapolate"
                )(t["abs_f"]),
                dtype=torch.float,
            ).exp()  # * torch.ones_like(t["d"])

            # t["V"] = (t["a"].T.unsqueeze(-1) * t["b"].unsqueeze(-2)).reshape(t["k"], -1)
            # t["ViA"] = (t["V"].reshape(t["k"], *t["d"].shape) / t["int_ps"]).reshape(t["k"], -1)
            # t["B"] = torch.linalg.inv(torch.eye(t["k"]) - t["ViA"] @ t["V"].T)

            t["PNd"] = t["P"].T @ self.apply_inverse_noise_covariance(t["d"], t)

            self.tod_list.append(t)

    def apply_inverse_noise_covariance(self, d, t):
        fd = torch.fft.fft(t["w"] * d, axis=-1)  # (n x t)

        Nfd1 = fd / t["int_ps"]

        if self.k > 0:
            Nfd2 = (t["ViA"].T @ (t["B"] @ (t["ViA"] @ fd.ravel()))).reshape(fd.shape)
        else:
            Nfd2 = 0.0

        return torch.fft.ifft(Nfd1 - Nfd2).real.ravel()

    @property
    def naive_map(self):
        if not hasattr(self, "_naive_map"):
            self._naive_map = self.binner.run()
        return self._naive_map.data.compute().ravel()

    def forward(self, t):
        return t["P"].T @ self.apply_inverse_noise_covariance((t["P"] @ self.sol.x()).reshape(t["d"].shape), t)

    def loss(self):
        return sum([(self.forward(t) - t["PNd"]).square().sum() for t in self.tod_list])

    @property
    def map(self):
        return ProjectionMap(
            data=self.sol.x().detach().numpy().reshape(self.map_shape),
            weight=np.ones(self.map_shape),
            stokes=self.stokes,
            t=self.t,
            nu=self.nu,
            resolution=self.resolution,
            center=self.center,
            degrees=False,
            frame=self.frame.name,
            units=self.map_units,
        )

    def fit(self, epochs: int = 4, steps_per_epoch: int = 64, lr: float = 1e-1):
        if not hasattr(self, "optimizer"):
            self.optimizer = torch.optim.Adam(self.sol.parameters(), lr=lr)

        self.sol.train()

        for epoch in range(epochs):
            pbar = trange(steps_per_epoch, bar_format="{l_bar}{bar:16}{r_bar}{bar:-10b}", desc=f"epoch {epoch + 1}/{epochs}")

            try:
                for step in pbar:
                    loss = self.loss()
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    pbar.set_postfix(loss=f"{loss.item():.03e}")

            except KeyboardInterrupt:
                break
