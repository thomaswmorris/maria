import logging
from collections.abc import Sequence

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from tqdm import tqdm, trange

from ..io import DEFAULT_BAR_FORMAT
from ..map import Map, ProjectionMap
from ..tod import TOD
from ..utils import bspline_basis_from_knots, decompose
from .base import BaseProjectionMapper
from .bin_mapper import BinMapper

logger = logging.getLogger("maria")

try:
    import torch

    torch.sparse.check_sparse_tensor_invariants.enable()
    torch.set_num_threads(16)
except ImportError:
    raise ImportError("Could not import torch (which is an optional dependency of maria)")
except Exception as error:
    raise error


class MaximumLikelihoodMapper(BaseProjectionMapper):
    def __init__(
        self,
        tods: Sequence[TOD],
        target: Map = None,
        k: int = 0,
        init: str = "bin",
        prior: bool = False,
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
            "remove_spline": {"knot_spacing": 30, "remove_el_gradient": True},
        },
        map_postprocessing: dict = {},
        progress_bars: bool = True,
        bilinear: bool = False,
        verbose: bool = False,
        noise_model_config: dict = {},
    ):
        super().__init__(
            tods=tods,
            target=target,
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
            bilinear=bilinear,
        )

        # if center is None:
        #     center = np.degrees(get_center_phi_theta(*np.stack([tod.coords.center(frame="ra/dec") for tod in tods]).T))

        self.map_init_sigma = 0
        self.noise_model_config = noise_model_config

        self.iteration = 0

        # TODO: fix detector-detector noise modes
        k = 0

        self.k = k

        self.binner = BinMapper(
            center=self.center,
            stokes=self.stokes,
            width=self.width,
            height=self.height,
            resolution=self.resolution,
            frame=self.frame,
            tods=self.tods,  # already preprocessed
            units=self.units,
            min_time=min_time,
            max_time=max_time,
            timestep=timestep,
            degrees=False,
            progress_bars=False,
            bilinear=False,
        )

        if init not in ["bin", "random"]:
            raise ValueError(f"Invalid init method '{init}'")

        self.apply_prior = prior
        self.init_method = init

        self.verbose = verbose

        self.sol = torch.tensor(np.zeros(self.map_shape).ravel(), requires_grad=True, dtype=torch.float)

        self.P_list = []
        self.PT_list = []

        self.hits = torch.zeros(np.prod(self.map_shape))

        for tod in tqdm(self.tods, "Computing pointing matrices", bar_format=DEFAULT_BAR_FORMAT):
            weights, samples, pixels, n_samples, n_pixels = self.map._stokes_weighted_pointing_matrix_ingredients(
                coords=tod.coords, dets=tod.dets, bilinear=self.bilinear
            )

            indices = torch.stack([torch.tensor(samples, dtype=torch.long), torch.tensor(pixels, dtype=torch.long)], dim=0)
            P = torch.sparse_coo_tensor(
                indices=indices, values=torch.tensor(weights, dtype=torch.float), size=(n_samples, n_pixels)
            ).coalesce()

            PT = P.T

            self.P_list.append(P)
            self.PT_list.append(PT)

            self.hits += P.abs().sum(axis=0, keepdim=True)[0].to_dense()

        self.reset_sol()

    def reset_step_size(self):

        self.reset_sol()
        loss1 = self.loss()
        loss1.backward()
        grad = self.sol.grad.data.clone()

        map_scale = self.sol.detach().square().mean().sqrt()
        self.step_size = 1e-1 * map_scale / grad.square().mean().sqrt()

        for _ in range(10):
            self.sol.data = self.sol.data - self.step_size * grad

            if self.loss() < loss1:
                break

            self.sol.data = self.sol.data + self.step_size * grad
            self.step_size *= 1e-1

    def reset_sol(self):

        M = self.naive_map.reshape(self.map_shape)
        H = self.hits.reshape(self.map_shape)
        M = M.where(H > 0, 0.0)

        numer = torch.tensor(sp.ndimage.gaussian_filter(M * H, sigma=0, axes=(-1, -2)))
        denom = torch.tensor(sp.ndimage.gaussian_filter(H, sigma=0, axes=(-1, -2)))

        WX = sp.signal.windows.tukey(M=H.shape[-1], alpha=0.1)[:, None]
        WY = sp.signal.windows.tukey(M=H.shape[-1], alpha=0.1)
        W = torch.tensor(WX * WY)
        TAPER = denom / denom.max()

        # apodize by hits (doctors hate this one weird trick)
        naive_map = (numer / denom).where(H > 0, 0.0) * TAPER * W

        init_sol = naive_map.where(naive_map.isfinite(), 0).ravel()

        if self.init_method == "random":
            map_var = torch.nansum(init_sol**2 * denom.ravel()) / torch.nansum(denom.ravel())
            white_noise_map = (
                map_var.sqrt() * torch.tensor(np.random.standard_normal(init_sol.shape), dtype=float) * (TAPER * W).ravel()
            )
            init_sol = torch.where(denom.ravel() > 0, white_noise_map, 0.0)

        self.sol = torch.tensor(init_sol.numpy(), requires_grad=True, dtype=torch.float)

    def update_noise_model(self, subtract_map: bool = True):
        self.noise_model = []

        pbar = tqdm(enumerate(self.tods), desc="Updating noise model", total=len(self.tods), bar_format=DEFAULT_BAR_FORMAT)

        for tod_index, tod in pbar:
            pbar.set_postfix(tod=f"{tod_index + 1}/{len(self.tods)}")

            t = {"tod": tod}
            t["nd"], t["nt"] = tod.shape

            d = sp.signal.detrend(tod.signal.compute())

            t["d"] = torch.tensor(d, dtype=torch.float)

            t["w"] = torch.tensor(
                sp.signal.windows.tukey(M=t["d"].shape[-1], alpha=self.noise_model_config.get("window_alpha", 0.1)),
                dtype=torch.float,
            ).unsqueeze(-2)
            t["f"] = np.fft.fftfreq(n=t["d"].shape[-1], d=1 / t["tod"].sample_rate)
            t["abs_f"] = np.abs(t["f"])
            t["P"] = self.P_list[tod_index]
            t["PT"] = self.PT_list[tod_index]

            # project the fitted map out of the data for noise modeling
            # m = torch.tensor(self.map.smooth(sigma=self.map.resolution).data.compute().ravel(), dtype=torch.float)
            m = torch.tensor(self.map.data.compute().ravel(), dtype=torch.float)

            t["d_no_map"] = t["d"] - (t["P"] @ m).reshape(t["d"].shape).detach()
            if subtract_map:
                t["wd"] = t["w"] * t["d_no_map"]
            else:
                t["wd"] = t["w"] * t["d"]

            t["fwd"] = torch.fft.fft(t["wd"])

            if self.k > 0:
                # fwd = torch.fft.fft(t["wd"])
                t["a"], t["b"] = [
                    torch.tensor(_, dtype=torch.complex64) for _ in decompose(t["wd"].numpy(), k=self.k, norm="var")
                ]
                t["noise"] = t["wd"] - t["a"] @ t["b"]
                # fn = torch.fft.fft(noise)

                # t["a"], t["b"] = [torch.tensor(_) for _ in decompose(t["fwd"].numpy(), k=self.k, norm="var")]
                t["U"] = t["a"].T.unsqueeze(-1)
                # t["noise"] = t["fwd"] - t["a"] @ t["b"]

            else:
                t["noise"] = t["wd"]

            t["fn"] = torch.fft.fft(t["noise"])

            t["noise_ps"] = t["fn"].square().abs()
            # t["noise_ps"][..., 0] = 0.0

            log_abs_f = np.log(t["abs_f"][1:])
            log_f_knots = np.linspace(log_abs_f.min() - 1e-6, log_abs_f.max() + 1e-6, 64)

            fb = np.zeros((len(log_f_knots), len(log_abs_f) + 1))
            fb[0, 0] = 1.0
            fb[2:-2, 1:] = bspline_basis_from_knots(log_abs_f, log_f_knots, order=3)
            fb = fb[fb.sum(axis=1) > 0]
            fb = fb / fb.sum(axis=0)
            fb = torch.tensor(fb).float()

            # log_ps = sp.ndimage.gaussian_filter(t["raw_ps"].log(), sigma=4, axes=-1, truncate=1)
            # smooth_ps = (torch.linalg.inv(fb @ fb.T) @ fb @ t["noise_ps"].T).T @ fb
            smooth_ps = torch.tensor(
                sp.ndimage.gaussian_filter(
                    t["noise_ps"].numpy(), sigma=self.noise_model_config.get("ps_sigma", 8), axes=-1, truncate=1
                )
            )
            t["A_inv"] = 1 / smooth_ps

            # t["A_inv"] = 1 / t["raw_ps"].mean(dim=0).unsqueeze(0)

            # f_bins = np.geomspace(t["abs_f"][1] / 1.00001, t["abs_f"].max() * 1.00001, 64)
            # f_vals = sp.stats.binned_statistic(t["abs_f"], t["abs_f"], bins=f_bins, statistic="mean").statistic
            # p_vals = sp.stats.binned_statistic(t["abs_f"], t["raw_ps"], bins=f_bins, statistic="mean").statistic

            # # pad
            # f_vals = np.array([0, *f_vals])
            # p_vals = np.concatenate([np.zeros((t["nd"], 1)), p_vals], axis=1)

            # use_bin = ~np.isnan(p_vals).any(axis=0)
            # t["ps"] = sp.interpolate.interp1d(f_vals[use_bin], p_vals[..., use_bin], fill_value="extrapolate")(t["abs_f"])
            # t["A_inv"] = torch.tensor(1 / np.where(t["abs_f"] > 0, t["ps"], t["raw_ps"]), dtype=torch.complex64)

            if self.k > 0:
                t["Q"] = t["a"].T.unsqueeze(-1)
                t["C_QAQ_inv"] = torch.linalg.inv(
                    torch.eye(self.k) - ((t["A_inv"] * t["Q"]).sum(axis=-1) @ t["Q"]).squeeze(-1)
                )

            # t["preconditioner"] = ((t["A_inv"].ravel() * t["PT"]) @ t["P"]) ** -1

            t["PNd"] = t["PT"] @ self.apply_inverse_noise_covariance(t["d"], t)
            # t["pPNd"] = t["preconditioner"] @ t["PNd"]

            t["ivar"] = t["PT"].abs() @ (t["w"] / t["noise"].var(dim=-1, keepdims=True)).ravel()

            self.noise_model.append(t)

            self.map_var = (self.sol.square().detach() * self.hits).sum() / self.hits.sum()

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

    def forward(self, t):
        return t["PT"] @ self.apply_inverse_noise_covariance((t["P"] @ self.sol).reshape(t["d"].shape), t)

    def ivar(self):
        return sum([t["ivar"] for t in self.noise_model])

    # def map_var(self):
    #     # pixel_weight = self.ivar()
    #     pixel_weight = self.hits > 0
    #     return (self.sol.square() * pixel_weight).sum() / pixel_weight.sum()

    def white_log_prior(self):
        return -0.5 * (self.sol.square() / self.map_var).sum()

    def loss(self):

        # this is the negative marginal log likelihood
        loss = sum([(self.forward(t) - t["PNd"]).square().sum() for t in self.noise_model])

        if self.apply_prior:
            self._log_prior = self.white_log_prior()
            loss -= self._log_prior

        return loss

    def get_map_data(self):
        return self.sol.detach().numpy()

    def get_map_weight(self):
        if hasattr(self, "noise_model"):
            return self.ivar().numpy()
        return self.hits.numpy()

    @property
    def map(self):
        return ProjectionMap(
            data=self.sol.detach().numpy().reshape(self.map_shape),
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

    def fit(self, epochs: int = 4, steps_per_epoch: int = 64, plot: bool = False, plot_kwargs: dict = {}):

        self.grad_hist = {}
        self.step_hist = {}
        self.loss_hist = {}

        for epoch in range(epochs):
            self.update_noise_model()

            if not hasattr(self, "step_size"):
                self.reset_step_size()

            last_loss = self.loss()
            self.sol.grad.data.zero_()
            last_loss.backward()
            last_grad = self.sol.grad.data.clone()
            normed_last_grad = last_grad / last_grad.square().sum().sqrt()

            self.grad_hist[epoch] = [last_grad]
            self.step_hist[epoch] = [self.step_size]
            self.loss_hist[epoch] = [last_loss.item()]

            pbar = trange(
                steps_per_epoch,
                bar_format=DEFAULT_BAR_FORMAT,
                desc=f"Fitting map (epoch {epoch + 1}/{epochs})",
                ncols=250,
            )

            postfix = {}
            pbar.set_postfix(log_prior=None, mll=None, pgrad=None, step=None)

            try:
                for step_index in pbar:
                    # do a step
                    self.sol.data = self.sol.data - self.step_size * last_grad
                    this_loss = self.loss()

                    # if too far, go back and decrease the step size
                    retries = 0
                    while (this_loss > last_loss) & (retries < np.inf):
                        retries += 1

                        self.sol.data = self.sol.data + self.step_size * last_grad
                        self.step_size *= 0.5**retries
                        self.sol.data = self.sol.data - self.step_size * last_grad
                        this_loss = self.loss()

                        postfix["step"] = f"{self.step_size:.03e}"
                        pbar.set_postfix(**postfix)

                    # compute gradient for next step
                    self.sol.grad.data.zero_()
                    this_loss.backward()
                    this_grad = self.sol.grad.data.clone()
                    normed_this_grad = this_grad / this_grad.square().sum().sqrt()

                    pgrad = (normed_this_grad * normed_last_grad).sum()

                    # pgrad_scaling = {
                    #     0.9999: 2.0,
                    #     0.999: 1.5,
                    #     0.99: 1.1,
                    #     0.9: 0.9,
                    #     0.5: 0.8,
                    #     -np.inf: 0.5,
                    # }

                    pgrad_scaling = {
                        0.99: 2.0,
                        0.25: 1.2,
                        0.0: 1.1,
                        -0.25: 0.9,
                        -0.5: 0.8,
                        -np.inf: 0.5,
                    }

                    for thresh, scale in pgrad_scaling.items():
                        if pgrad >= thresh:
                            self.step_size *= scale
                            break

                    postfix = {
                        "mll": f"{-this_loss.item():.03e}",
                        "step": f"{self.step_size:.03e}",
                        "pgrad": f"{pgrad:.03f}",
                    }

                    if self.apply_prior:
                        postfix["log_prior"] = f"{self._log_prior.item():.03e}"

                    self.loss_hist[epoch].append(this_loss.item())
                    self.step_hist[epoch].append(self.step_size)

                    last_loss = this_loss
                    last_grad = this_grad
                    normed_last_grad = normed_this_grad

                    pbar.set_postfix(**postfix)

            except KeyboardInterrupt:
                logger.info("Stopped fitting routine")
                break

            if plot:
                self.map.plot(**plot_kwargs)
                plt.show()
