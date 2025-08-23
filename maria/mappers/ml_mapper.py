# try:
#     import torch
# except ImportError:
#     raise ImportError("To do maximum likelihood mapping in maria, please install PyTorch (pip install torch)")

# import matplotlib.pyplot as plt
# import numpy as np
# import scipy as sp
# import torch
# from tqdm import tqdm, trange

# torch.set_num_threads(1)

# from __future__ import annotations

# import os
# from collections.abc import Sequence
# from time import sleep

# import dask.array as da
# import numpy as np
# import scipy as sp
# from torch import nn
# from tqdm import tqdm

# from maria.utils.signal import fit_bspline

# from ..band import Band, BandList
# from ..map import ProjectedMap
# from ..tod import TOD
# from .base import BaseProjectionMapper


# def remove_ends(data, ntip=16):
#     start_values = data[..., :ntip].mean(axis=-1)
#     end_values = data[..., -ntip:].mean(axis=-1)

#     return data - torch.linspace(start_values, end_values, data.shape[-1]).T


# class MaximumLikelihoodMapper(BaseProjectionMapper, nn.Module):
#     def __init__(
#         self,
#         center: tuple[float, float] = None,
#         stokes: str = "I",
#         width: float = 1,
#         height: float = None,
#         resolution: float = 0.01,
#         frame: str = "ra/dec",
#         units: str = "K_RJ",
#         degrees: bool = True,
#         calibrate: bool = False,
#         tod_preprocessing: dict = {},
#         map_postprocessing: dict = {},
#         tods: Sequence[TOD] = [],
#     ):
#         super(MaximumLikelihoodMapper, self).__init__()

#         height = height or width

#         super().__init__(
#             center=center,
#             stokes=stokes,
#             width=width,
#             height=height,
#             resolution=resolution,
#             frame=frame,
#             degrees=degrees,
#             calibrate=calibrate,
#             tods=tods,
#             units=units,
#             tod_preprocessing=tod_preprocessing,
#             map_postprocessing=map_postprocessing,
#         )

#         self.map = map
#         self.units = units

#         self.tods = [tod.to(self.units) for tod in tods]
#         self.tod_list = {}

#         for tod_index, tod in tqdm(enumerate(self.tods)):
#             t = {"tod": tod}
#             self.tod_list[tod_index] = t
#             ntip = int(0.5 * t["tod"].sample_rate)
#             d = tod.signal.compute()
#             d = d - fit_bspline(d.mean(axis=0), x=tod.time, spacing=10)
#             d = d - np.linspace(d[:, :ntip].mean(axis=-1), d[:, -ntip:].mean(axis=-1), d.shape[-1]).T
#             t["d"] = torch.tensor(d, dtype=torch.float)
#             t["w"] = torch.tensor(sp.signal.windows.hamming(M=t["d"].shape[-1]), dtype=torch.float)
#             t["f"] = np.fft.fftfreq(n=t["d"].shape[-1], d=1 / t["tod"].sample_rate)
#             t["abs_f"] = np.abs(t["f"])

#             idx, cum_npix = self.map._pointing_matrix_sparse_indices(t["tod"].coords)
#             indices = torch.stack([torch.arange(len(idx), dtype=torch.long), torch.tensor(idx, dtype=torch.long)], dim=0)
#             t["P"] = torch.sparse_coo_tensor(
#                 indices=indices, values=torch.ones(len(idx), dtype=torch.float), size=(len(idx), cum_npix)
#             )

#         self.sol_list = []
#         self.iteration = 0
#         self.iter_maps = []

#         self.norm = 1

#         self.initialize_mapper()

#         self.tods = []
#         self.add_tods(tods)

#     def initialize_mapper(self):
#         self.bands = BandList(bands=[])

#         for tod_index, tod in tqdm(enumerate(self.tods)):
#             t = {"tod": tod}
#             self.tod_list[tod_index] = t
#             ntip = int(0.5 * t["tod"].sample_rate)
#             d = d - np.linspace(d[:, :ntip].mean(axis=-1), d[:, -ntip:].mean(axis=-1), d.shape[-1]).T
#             t["d"] = torch.tensor(d, dtype=torch.float)
#             t["w"] = torch.tensor(sp.signal.windows.hamming(M=t["d"].shape[-1]), dtype=torch.float)
#             t["f"] = np.fft.fftfreq(n=t["d"].shape[-1], d=1 / t["tod"].sample_rate)
#             t["abs_f"] = np.abs(t["f"])

#             # compute the pointing tensor
#             idx, cum_npix = self.map._pointing_matrix_sparse_indices(t["tod"].coords)
#             indices = torch.stack([torch.arange(len(idx), dtype=torch.long), torch.tensor(idx, dtype=torch.long)], dim=0)
#             t["P"] = torch.sparse_coo_tensor(
#                 indices=indices, values=torch.ones(len(idx), dtype=torch.float), size=(len(idx), cum_npix)
#             )

#         self.iter_sol = nn.Parameter(data=self.naive_map().ravel().clone())

#         self.map = ProjectedMap(
#             data=np.zeros(self.map_shape),
#             weight=np.ones(self.map_shape),
#             stokes=self.stokes,
#             nu=self.bands.center,
#             resolution=self.resolution,
#             center=self.center,
#             degrees=False,
#             frame=self.frame.name,
#             units=self.units,
#         )

#         self.start_iteration()

#     def solution(self):
#         return sum(self.iter_maps) + self.m().detach()

#     def start_iteration(self):
#         self.iteration += 1
#         self.iter_maps.append(self.iter_sol.detach().reshape(self.map_shape()))
#         self.iter_sol = nn.Parameter(data=torch.zeros(len(self.iter_sol)))

#         for tod_index in self.tod_list:
#             t = self.tod_list[tod_index]

#             # project the fitted map out of the data
#             t["d_no_map"] = t["d"] - (t["P"] @ sum(self.iter_maps).ravel()).reshape(t["d"].shape)

#             t = self.tod_list[tod_index]

#             fd = torch.fft.fft(t["d_no_map"] * t["w"])
#             t["ps"] = fd.square().abs().float()

#             n_bin = 128
#             mid_f = np.geomspace(t["abs_f"][t["abs_f"] > 0].min(), t["abs_f"].max(), n_bin)
#             dlogf = np.exp(np.gradient(np.log(mid_f)).mean())
#             bin_f = np.geomspace(mid_f[0] / np.sqrt(dlogf), mid_f[-1] * np.sqrt(dlogf), n_bin + 1)
#             bin_y = sp.stats.binned_statistic(t["abs_f"], t["ps"].log(), bins=bin_f).statistic

#             use = ~np.isnan(bin_y).any(axis=0)
#             t["int_ps"] = torch.tensor(
#                 sp.interpolate.interp1d(mid_f[use], bin_y[:, use], axis=1, bounds_error=False, fill_value="extrapolate")(
#                     t["abs_f"]
#                 ),
#                 dtype=torch.float,
#             ).exp()

#             t["PNd"] = t["P"].T @ self.apply_inverse_noise_covariance(t["d_no_map"], tod_index=tod_index)

#         # self.sol = nn.Parameter(torch.zeros_like(self.naive_map().ravel()))

#         # self.loss_norm = 1
#         # self.loss_norm = self.loss().item()

#         fs = []
#         for tod_index, t in self.tod_list.items():
#             f = t["P"].T @ self.apply_inverse_noise_covariance(
#                 (t["P"] @ self.naive_map().ravel()).reshape(t["d"].shape), tod_index=tod_index
#             )
#             fs.append(f)

#         self.norm = 1 / torch.tensor([f.std() for f in fs]).mean()

#     def apply_inverse_noise_covariance(self, d, tod_index=0):
#         t = self.tod_list[tod_index]
#         return torch.fft.ifft(torch.fft.fft(t["w"] * d, axis=-1) / t["int_ps"]).real.ravel()

#     def map_weight_list(self, index=None):
#         return [t["P"].sum(axis=0).to_dense().reshape(self.map_shape()) for tod_index, t in self.tod_list.items()]

#     def map_weight(self, index=None):
#         return sum(self.map_weight_list())

#     def naive_map(self):
#         cum_wm = 0
#         cum_w = 0
#         map_weight_list = self.map_weight_list()
#         for tod_index, t in self.tod_list.items():
#             cum_wm += (t["P"].T @ t["d"].ravel()).reshape(self.map_shape())
#             cum_w += map_weight_list[tod_index]
#         return cum_wm / cum_w

#     def m(self):
#         return self.iter_sol.reshape(self.map.dims["y"] + 2, self.map.dims["x"] + 2)

#     def forward(self, tod_index: int = None):
#         t = self.tod_list[tod_index]
#         return t["P"].T @ self.apply_inverse_noise_covariance(
#             (t["P"] @ self.iter_sol).reshape(t["d"].shape), tod_index=tod_index
#         )

#     def forward_list(self):
#         return [self.forward(tod_index) for tod_index in self.tod_list]
#         # if index is None:
#         #     return [t["P"].T @ self.apply_inverse_noise_covariance((t["P"]
#  @ self.sol).reshape(t["d"].shape), tod_index=tod_index)
#         #

#     def loss(self):
#         return sum(
#             [
#                 self.norm**2 * (PNPm - t["PNd"]).square().mean()
#                 for PNPm, t in zip(self.forward_list(), self.tod_list.values())
#             ]
#         )


# output_map = input_map[..., :, :]

# ml_mapper = MLMapper(tods=tods, map=output_map, units="mK_RJ")
# # bpm#.loss()
# self = ml_mapper.to("cpu")
# spectra = {}
# self.loss()
