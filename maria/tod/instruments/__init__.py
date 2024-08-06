import copy
import glob
import os
import pickle
import re
import sys
import time as ttime
from datetime import datetime

import astropy as ap
import matplotlib as mpl
import moby2 as m2
import numpy as np
import pandas as pd
import pylab as plt
import pytz
import scipy as sp
from IPython.display import clear_output
from numpy import linalg as la
from scipy import interpolate, ndimage, optimize, signal, stats

base, this_filename = os.path.split(__file__)
sys.path.append(base)


class NotEnoughDetsError(Exception):
    pass


import act

from . import signal


def taper_window(N, end_prop=5e-2):
    n_taper = int(N * end_prop)
    return np.r_[
        0.5 * (1 - np.cos(np.linspace(0, np.pi, n_taper))),
        np.ones(N - 2 * n_taper),
        0.5 * (1 - np.cos(np.linspace(np.pi, 0, n_taper))),
    ]


def load(
    tod_ids,
    source="moby",
    downsample_rate=4,
    min_dets=16,
    cal_schemes=["local", "bs", "iv"],
    cut_schemes=["depot", "auto"],
    flatfield_schemes=["atm", "dummy"],
    verbose=False,
):
    """
    Flexibly load and concatenate a (band, time) array of TODs.
    TOD ids should have format (obs time).(sync time).(array):f(nominal_freq)
    """

    if source == "moby":
        tod = None
        res = []

        for _tod_ids in tod_ids:
            _tod = None

            for __tod_id in _tod_ids:
                st = ttime.time()

                # load a moby TOD, with some extra attributes
                __mtod, __res = act.load_moby_tod(
                    tod_id=__tod_id,
                    cal_schemes=cal_schemes,
                    cut_schemes=cut_schemes,
                    flatfield_schemes=flatfield_schemes,
                )

                # convert to a todder TOD
                __tod = TOD(
                    source=__mtod,
                    source_type="moby",
                    downsample_rate=downsample_rate,
                    downsample_method="triangle",
                    verbose=verbose,
                )

                res.append(__res)

                if not __tod.data.shape[0] > min_dets:
                    raise NotEnoughDetsError(
                        f"Not enough detectors! (n={__tod.data.shape[0]})"
                    )

                # add to our existing TODs
                if _tod is None:
                    _tod = __tod
                else:
                    _tod.append(__tod, dimension="time")
                del __tod, __mtod

                if verbose:
                    print(f"loaded {__tod_id} in {ttime.time() - st:.01f}s")

            if tod is None:
                tod = _tod
            else:
                tod.append(_tod, dimension="dets")
            del _tod

        tod._update_variable_metadata()
        tod._compute_phase()
        tod.res = res

        return tod


class TOD:
    def __init__(
        self,
        source,
        res=None,
        source_type="moby",
        downsample_rate=1,
        downsample_method="triangle",
        verbose=False,
    ):
        if source_type == "moby":
            if source.cuts is None:
                self.cuts = signal.make_cuts(
                    sp.signal.detrend(source.data), downsample_rate=4
                )
                self.cuts_type = "auto"
            else:
                self.cuts = source.cuts
                self.cuts_type = "act"

            self.data = signal.apply_cuts(
                source.data, source.cuts, tol=4, method="splice"
            )
            self.data = signal.downsample(
                self.data, rate=downsample_rate, method=downsample_method
            )

            self.metadata = pd.DataFrame(index=np.arange(len(source.det_uid)))
            for key in source.info.array_data.keys():
                self.metadata.loc[:, key] = np.array(
                    source.info.array_data[key][source.det_uid]
                )

            self.metadata.loc[:, "cals"] = source.calibration.cal.values
            self.metadata.loc[:, "cals"] *= 1e12 / source.info.mce_filter.gain()
            self.metadata.loc[:, "sky_x"] = source.offsets.loc[
                source.det_uid, "x0"
            ].values
            self.metadata.loc[:, "sky_y"] = source.offsets.loc[
                source.det_uid, "y0"
            ].values
            self.metadata.loc[:, "flatfield"] = source.flatfield.ff.values

            self.hk = np.c_[
                source.ctime,
                signal.unwrap_angle(source.az),
                source.alt,
                source.thermometer,
            ].T
            self.hk = signal.downsample(
                self.hk, rate=downsample_rate, method="triangle"
            )
            self.hk = signal.apply_cuts(
                self.hk, signal.make_cuts(self.hk, downsample_rate=4), method="splice"
            )
            self.time, self.azim, self.elev, self.thrm = self.hk
            self.thrm = self.thrm[None]

            if np.isnan(self.hk).any():
                return

            # cut detectors that are zero or that are calibrated to zero
            self.subset(
                dets=(self.data[:, :2].std(axis=1) > 0)
                & (np.abs(self.metadata.cals.values) > 0),
                verbose=False,
            )

            if not len(self.data) > 0:
                return

            pa, nom_freq = re.sub(
                r".*ar([0-9]):f([0-9])", r"PA\1 \2", source.tod_id
            ).split()
            band = f"{pa}_f{nom_freq}"
            self.metadata.insert(0, "pa", pa)
            self.metadata.insert(1, "band", band)

    def _update_variable_metadata(self):
        self.dt = np.median(np.gradient(self.time))
        self.nd, self.nt = self.data.shape
        self.fs = 1 / self.dt

        centers = np.degrees(np.vstack(self.metadata.loc[:, ["sky_x", "sky_y"]].values))

        max_dist = 1 / 60  # one arcminute

        det_group_indices = []

        group_label = np.zeros(len(centers)).astype(int) - 1

        n_groups = 0
        for i, _center in enumerate(centers):
            if not group_label[i] < 0:
                continue

            inds = np.where(
                np.sqrt(np.square(_center - centers).sum(axis=1)) < max_dist
            )[0]
            det_group_indices.append(inds)
            group_label[inds] = n_groups
            n_groups += 1

        self.metadata.loc[:, "group"] = group_label

    def _compute_phase(self):
        f_w = 4
        f, ps = sp.signal.periodogram(self.azim, fs=self.fs, window="hamming")
        i_f = ps.argmax()
        self.period = np.sum(ps[i_f - f_w : i_f + f_w]) / np.sum(
            f[i_f - f_w : i_f + f_w] * ps[i_f - f_w : i_f + f_w]
        )
        self.phase = np.pi * (
            sp.signal.sawtooth(2 * np.pi * self.time / self.period, width=1) + 1
        )

        self.phase_mids = np.linspace(0, 2 * np.pi, 360 + 1)
        self.dphase = np.gradient(self.phase_mids).mean()
        self.phase_bins = np.append(
            self.phase_mids[0] - self.dphase / 2, self.phase_mids + self.dphase / 2
        )

    def post_process(self, mode="minimal"):
        if mode == "crop":
            exponential = lambda t, a, b, c, d: a * np.exp(-t / b) + c * t + d

            y = signal.downsample(
                sp.signal.detrend(
                    (self.data / self.data.std(axis=1)[:, None]).mean(axis=0)
                ),
                rate=16,
                method="triangle",
            )
            x = np.linspace(0, 1, len(y))

            p0 = [y[0] - y.mean(), 0.01, 0, y.mean()]
            pars, cpars = sp.optimize.curve_fit(
                exponential,
                x,
                y,
                p0,
                bounds=[
                    [-np.inf, 0, -np.inf, -np.inf],
                    [np.inf, 0.05, +np.inf, +np.inf],
                ],
            )
            t_cut = 4 * pars[1] * np.ptp(self.time)  # cuts a maximum of 20 percent
            print(f"cutting {t_cut:.01f} seconds")
            self.subset(samples=[int(t_cut * self.fs), self.nt - 1])

        if mode == "calibrate":
            self.power = sp.signal.detrend(
                self.data
                * (self.metadata.flatfield.values * self.metadata.cals.values)[:, None]
            )
            self.power = signal.highpass(self.power, c=1e-3, fs=self.fs, order=1)

        if mode == "lax_cut":
            keep = self.get_good_dets(
                field="power",
                verbose=True,
                rel_gain_bounds=(1e-2, 1e2),
                rel_residual_bounds=(0, 5e-1),
            )
            self.subset(dets=keep, fields=["data", "power"], verbose=True)

        if mode == "strict_cut":
            keep = self.get_good_dets(
                field="power",
                verbose=True,
                rel_gain_bounds=(0.25, 4),
                rel_residual_bounds=(0.0, 0.25),
            )
            self.subset(dets=keep, fields=["data", "power"], verbose=True)

        if mode == "remove_template":
            self.phase_template = signal.get_phase_template(
                self.power,
                self.phase,
                n_phase_bins=360,
                discriminator=self.metadata.pa.values,
            )
            self.power -= self.phase_template

    def detrend(self, deg, downsample_rate=16):
        pt = np.linspace(0, 1, self.nt)
        self.data -= np.poly1d(
            np.polyfit(
                pt[::downsample_rate],
                self.data[:, ::downsample_rate].mean(axis=0),
                deg=deg,
            )
        )(pt)[None, :]

    def get_good_dets(
        self,
        field,
        verbose=False,
        rel_gain_bounds=(0.5, 2),
        rel_residual_bounds=(0.0, 0.25),
    ):
        downsample_rate = (
            16  # downsample rate, this speeds things up and doesn't affect accuracy
        )

        DATA = getattr(self, field)

        keep = np.ones(DATA.shape[0]).astype(
            bool
        )  # mask of detectors to keep, initially all true

        for band in np.unique(self.metadata.band):  # for all unique passbands ...
            m = self.metadata.band == band  # get a mask of detectors in this passband

            downsampled_data = signal.downsample(DATA[m], rate=downsample_rate)

            data_norm = downsampled_data.std(axis=-1)

            normed_data = (
                downsampled_data / data_norm[:, None]
            )  # grab and normalize the data for this passband

            u, s, v = la.svd(normed_data, full_matrices=False)  # take the SVD

            first_mode = np.outer(
                u[:, 0], s[0] * v[0]
            )  # the dominant mode from the SVD

            rel_residuals = (normed_data - first_mode).std(
                axis=1
            )  # RMS of normalized residuals after removing the first mode

            gains = (
                data_norm * u[:, 0] * np.sqrt(len(s))
            )  # unitful coupling to normalize first mode

            rel_gains = gains / np.median(gains)  # relative gains WRT the band median

            # cut detectors based on these: bad detectors should pop out to the eye when these are plotted
            # rel_residuals, "how much of the data remains after removing the first mode" (usually ~0.1 for well-behaved dets)
            # rel_gains, "the gain of this detector relative to other detectors"

            # why are these what they are?
            # rel_gain_bounds      = (0.5, 2) # lower and upper bounds for acceptable relative gains
            # rel_residual_bounds  = (0.0, 0.25) # lower and upper bounds for acceptable relative residuals

            g = (rel_gains > rel_gain_bounds[0]) & (rel_gains < rel_gain_bounds[1])
            g &= (rel_residuals > rel_residual_bounds[0]) & (
                rel_residuals < rel_residual_bounds[1]
            )
            g &= rel_residuals < 25 * np.percentile(
                rel_residuals, q=50
            )  # cut if it's a lot worse than its band colleagues

            if verbose:
                print(
                    f"{band} : ({(~g).sum():>3} / {len(g)}) are bad ({1e2*(~g).sum()/len(g):>4.01f}%)"
                )

            keep[m] = g

        return keep

    def subset(self, dets=None, samples=None, fields=["data"], verbose=False):
        init_shape = self.data.shape

        if not dets is None:
            if not dets.dtype == bool:
                raise ValueError()

            for attr in fields:
                if attr in dir(self):
                    setattr(self, attr, getattr(self, attr)[dets])
            self.metadata = self.metadata.loc[dets]

            if verbose:
                print(f"dets subset : {init_shape} -> {self.data.shape}")

            if not self.data.shape[0] > 12:
                raise NotEnoughDetsError(
                    f"Not enough detectors! (n={self.data.shape[0]} < 12)"
                )

        if not samples is None:
            if len(samples) == 2:
                samples = (np.arange(self.nt) > samples[0]) & (
                    np.arange(self.nt) < samples[1]
                )

            for attr in [*fields, "thrm"]:
                if attr in dir(self):
                    setattr(self, attr, getattr(self, attr)[:, samples])
            for attr in ["time", "azim", "elev", "phase"]:
                setattr(self, attr, getattr(self, attr)[samples])

            if verbose:
                print(f"time subset : {init_shape} -> {self.data.shape}")

            if not self.data.shape[1] > 0:
                print("all times cut!")
                self.nd, self.nt = self.data.shape
                return

        self._update_variable_metadata()

    def copy(self):
        return copy.deepcopy(self)

    def append(tod1, tod2, fields=["data"], dimension=None, bins=None):
        if dimension == "time":
            new_det_uid = sorted(
                list(set(tod1.metadata.det_uid) & set(tod2.metadata.det_uid))
            )

            tod1_det_list = list(tod1.metadata.det_uid)
            tod2_det_list = list(tod2.metadata.det_uid)

            tod1_det_index = [tod1_det_list.index(det) for det in new_det_uid]
            tod2_det_index = [tod2_det_list.index(det) for det in new_det_uid]

            tod1.data = np.c_[
                tod1.data[tod1_det_index],
                tod2.data[tod2_det_index]
                + (
                    tod1.data[tod1_det_index, -16:].mean(axis=-1)
                    - tod2.data[tod2_det_index, :16].mean(axis=-1)
                )[:, None],
            ]

            tod1.metadata = tod1.metadata.iloc[tod1_det_index]
            for attr in ["time", "azim", "elev", "phase", "thrm"]:
                if attr in dir(tod1):
                    setattr(
                        tod1,
                        attr,
                        np.concatenate(
                            [getattr(tod1, attr), getattr(tod2, attr)], axis=-1
                        ),
                    )

            # ensures unique time samples
            linear = lambda x, a, b: a * x + b
            pars, cpars = sp.optimize.curve_fit(
                linear, np.arange(len(tod1.time)), tod1.time
            )
            tod1.time = linear(np.arange(len(tod1.time)), *pars)

            tod1._update_variable_metadata()

        if dimension == "dets":
            t_min = np.maximum(tod1.time[0], tod2.time[0])
            t_max = np.minimum(tod1.time[-1], tod2.time[-1])

            master_time = np.arange(t_min, t_max, 1 / np.minimum(tod1.fs, tod2.fs))

            tod1_t_index = np.round(
                np.interp(master_time, tod1.time, np.arange(tod1.nt))
            ).astype(int)
            tod2_t_index = np.round(
                np.interp(master_time, tod2.time, np.arange(tod2.nt))
            ).astype(int)

            tod1.metadata = tod1.metadata.append(tod2.metadata)
            tod1.metadata.index = np.arange(len(tod1.metadata))

            for attr in ["azim", "elev"]:
                setattr(
                    tod1, attr, np.interp(master_time, tod1.time, getattr(tod1, attr))
                )

            for attr in [*fields, "thrm"]:
                setattr(
                    tod1,
                    attr,
                    np.vstack(
                        [
                            getattr(tod1, attr)[:, tod1_t_index],
                            getattr(tod2, attr)[:, tod2_t_index],
                        ]
                    ),
                )

            tod1.time = master_time

            tod1._update_variable_metadata()

    def reindex(self, start=0, end=-1, downsample_rate=1):
        self.data = self.data[:, start:end:downsample_rate]
        for attr in ["time", "azim", "elev"]:
            setattr(self, attr, getattr(self, attr)[start:end:downsample_rate])

        self.nd, self.nt = self.data.shape

    def get_bimodes(self):
        noise_data = np.zeros(self.power.shape)

        for pa in np.unique(self.metadata.paba):
            pam = self.metadata.pa == pa
            band_pair_common_modes = []

            for ba in np.unique(self.metadata.paba.loc[pam]):
                bam = self.metadata.paba == paba
                band_data = self.power[pam & bam].mean(axis=0)
                band_pair_common_modes.append(band_data)

            if len(band_pair_common_modes) > 2:
                pa_power_template = np.subtract(*band_pair_common_modes)
                pa_power_template /= pa_power_template.std()

            else:
                pa_power_template = self.power[pam].mean(axis=0)

            power_gains = (pa_power_template * self.power[pam]).sum(axis=1) / np.square(
                pa_power_template
            ).sum()

            pa_noise_template = (
                self.power[pam] - np.outer(power_gains, pa_power_template)
            ).mean(axis=0)
            pa_noise_template /= pa_noise_template.std()
            pa_noise_gains = (pa_noise_template * self.power[pam]).sum(
                axis=1
            ) / np.square(pa_noise_template).sum()
            noise_data[pam] = np.outer(pa_noise_gains, pa_noise_template)

        power_data = self.power - noise_data

        return power_data, noise_data

    def cluster(
        self,
        n=16,
        method="kmeans",
        discriminator=None,
        field="data",
        filter_kind="none",
        flim=[1e-2, 1e1],
        order=3,
        min_spacing=np.radians(0.1),
    ):
        if discriminator is None:
            discriminator = np.repeat("all", self.nd)

        setattr(self, f"c_{field}", np.empty((0, self.nt)))

        self.cx, self.cy, self.ci = [], [], []

        for attr in ["c_x", "c_y", "c_pa", "c_band"]:
            setattr(self, attr, [])
        for d in np.unique(discriminator):
            m = discriminator == d

            print(f"clustering {m.sum()} detectors from {d} into {n} groups")

            from sklearn.cluster import KMeans

            points = np.vstack(
                [self.metadata.sky_x.values[m], self.metadata.sky_y.values[m]]
            ).T
            kmeans = KMeans(n_clusters=n, random_state=0).fit(points)

            if method == "kmeans":
                kmeans = KMeans(n_clusters=n, random_state=0).fit(points)
                self.c_id = kmeans.labels_

            if method == "grid":
                nx, ny = int(np.ptp(self.x[m]) / min_spacing), int(
                    np.ptp(self.y[m]) / min_spacing
                )

                x_bins = np.linspace(
                    self.x[m].min() - 1e-6, self.x[m].max() + 1e-6, nx + 1
                )
                y_bins = np.linspace(
                    self.y[m].min() - 1e-6, self.y[m].max() + 1e-6, ny + 1
                )
                x_id, y_id = np.digitize(self.x[m], bins=x_bins), np.digitize(
                    self.y[m], bins=y_bins
                )
                print(nx, ny, x_bins.min(), x_bins.max())
                self.c_id = nx * x_id + y_id

            cx = np.array(
                [
                    self.metadata.sky_x.values[m][self.c_id == i].mean(axis=0)
                    for i in np.sort(np.unique(self.c_id))
                ]
            )
            cy = np.array(
                [
                    self.metadata.sky_y.values[m][self.c_id == i].mean(axis=0)
                    for i in np.sort(np.unique(self.c_id))
                ]
            )
            self.c_n = np.array(
                [np.sum(self.c_id == i) for i in np.sort(np.unique(self.c_id))]
            )

            masked_data = getattr(self, field)[m]

            setattr(
                self,
                f"c_{field}",
                np.r_[
                    getattr(self, f"c_{field}"),
                    np.concatenate(
                        [
                            masked_data[self.c_id == i].mean(axis=0)[None, :]
                            for i in np.sort(np.unique(self.c_id))
                        ],
                        axis=0,
                    ),
                ],
            )
            # self.pa = np.array([self.pa[self.c_id==i][0] for i in np.sort(np.unique(self.c_id))])
            # self.ba = np.array([self.ba[self.c_id==i][0] for i in np.sort(np.unique(self.c_id))])

            self.c_x.extend(cx), self.c_y.extend(cy)

            for attr in ["pa", "ba", "band"]:
                getattr(self, f"c_{attr}").extend(
                    np.repeat(self.metadata[attr].values[m][0], len(cx))
                )

        for attr in ["c_x", "c_y", "c_pa", "c_ba", "c_band"]:
            setattr(self, attr, np.array(getattr(self, attr)))

        self.n_cluster = len(self.cx)

        if filter_kind == "band-pass":
            setattr(
                self,
                f"c_{field}",
                _bandpass(getattr(self, f"c_{field}"), *flim, self.fs, order=order),
            )
        if filter_kind == "low-pass":
            setattr(
                self,
                f"c_{field}",
                _lowpass(getattr(self, f"c_{field}"), flim, self.fs, order=order),
            )
        if filter_kind == "high-pass":
            setattr(
                self,
                f"c_{field}",
                _highpass(getattr(self, f"c_{field}"), flim, self.fs, order=order),
            )

        self.c_z = self.c_x + 1j * self.c_y
        self.c_nn = np.outer(self.c_n, self.c_n)
        self.c_dx = np.subtract.outer(self.c_x, self.c_x)
        self.c_dy = np.subtract.outer(self.c_y, self.c_y)
        self.c_dz = np.subtract.outer(self.c_z, self.c_z)
        self.c_da, self.c_dr = np.angle(self.c_dz), np.abs(self.c_dz)

    def get_modes(self, dt, fields=["power"], discriminators=[None]):
        master_time = np.arange(self.time[0], self.time[-1], dt)

        modes = {}
        modes["data"] = {}
        modes["meta"] = {}

        modes["meta"]["time"] = master_time
        modes["meta"]["azim"] = sp.interpolate.interp1d(
            self.time, self.azim, kind="quadratic"
        )(master_time)
        modes["meta"]["elev"] = sp.interpolate.interp1d(
            self.time, self.elev, kind="quadratic"
        )(master_time)

        for field, discriminator in zip(fields, discriminators):
            if discriminator is None:
                discriminator = np.repeat("all", self.nd)

            modes["data"][field] = {}

            for d_val in np.unique(discriminator):
                d_mask = discriminator == d_val

                modes["data"][field][d_val] = sp.interpolate.interp1d(
                    self.time,
                    getattr(self, field)[d_mask].mean(axis=0),
                    kind="quadratic",
                )(master_time)

        return modes

    def get_splits(self, min_length=5):
        sv = sp.ndimage.gaussian_filter(np.gradient(self.azim) * self.fs, sigma=1)

        # flag wherever the scan velocity changing direction (can be more general)
        flags = np.r_[0, np.where(np.sign(sv[:-1]) != np.sign(sv[1:]))[0], len(sv) - 1]

        min_samples = min_length * self.fs
        pair_lag_splits = [
            (s, e) for s, e in zip(flags[:-1], flags[1:]) if e - s > min_samples
        ]

        return pair_lag_splits

    def do_pair_lag(
        self,
        splits=None,
        max_lag=5,
        field="c_power",
        method="classic",
        discriminator=None,
    ):
        if splits is None:
            splits = self.get_splits(min_length=5)
        if discriminator is None:
            discriminator = np.repeat("all", self.nd)

        field_data = getattr(self, f"c_{field}").copy()
        nd, nt = field_data.shape

        # TOD is a enact-defined object. (wait no it's not!)
        max_di = int(max_lag / self.dt)
        di_sampler = np.linspace(-max_di, max_di, 2 * max_di + 1)

        nr_us = 1024
        us_di_sampler = np.linspace(-max_di, max_di, 2 * nr_us + 1)

        # use Fourier methods to compute the pair-lag, for each sub-split and each cluster pair
        self.pair_lag = np.zeros((len(splits), len(discriminator), len(discriminator)))

        idets, jdets = np.where(discriminator[:, None] == discriminator[None, :])

        for split, (s, e) in enumerate(splits):
            hp_data = _highpass(field_data[:, s:e], c=1 / max_lag, fs=self.fs, order=3)
            pl_data = hp_data * taper_window(e - s, end_prop=1e-1)[None, :]

            ft_pl_data = np.fft.fft(pl_data, axis=-1)

            for i, j in zip(idets, jdets):
                if not i > j:
                    continue

                if method == "fourier":
                    dijt = np.real(np.fft.ifft(ft_pl_data[i] * np.conj(ft_pl_data[j])))
                    corr = np.r_[dijt[-max_di:], dijt[: max_di + 1]]

                if method == "classic":
                    corr = np.correlate(pl_data[i], pl_data[j], mode="full")[
                        (nt - 1) - max_di : (nt - 1) + max_di + 1
                    ]

                rs_corr = sp.interpolate.interp1d(di_sampler, corr, kind="cubic")(
                    us_di_sampler
                )
                self.pair_lag[split, i, j] = (rs_corr.argmax() - nr_us) * max_di / nr_us

        self.pair_lag *= self.dt

        self.norm_pair_lag = np.zeros(self.pair_lag.shape)
        self.norm_pair_lag = self.pair_lag / (self.c_dr[None] + 1e-20)

        norm_lag_flat = lambda oa, vx, vy: (
            vx * np.cos(oa) + vy * np.sin(oa)
        ) / np.square(np.abs(vx + 1j * vy))

        motion = pd.DataFrame(
            columns=[
                "t",
                "band",
                "duration",
                "azim_v",
                "fit_ang_v_x",
                "fit_ang_v_y",
                "co_r_2",
            ]
        )

        mv = np.radians(10)
        bounds = [[-mv, -mv], [mv, mv]]

        for d_val in np.unique(discriminator):
            d_mask = discriminator == d_val

            for split, (s, e) in enumerate(splits):
                azim_v = np.gradient(self.azim[s:e]).mean()

                vx0, vy0 = -azim_v * self.fs * np.cos(self.elev[s:e].mean()) + 1e-4, 0

                USE = (
                    (self.norm_pair_lag[split] != 0) & d_mask[None, :] & d_mask[:, None]
                )

                if not USE.sum() > 0:
                    continue

                abs_norm_pair_lag_bounds = np.percentile(
                    np.abs(self.norm_pair_lag[split][USE]), q=[5, 95]
                )

                USE &= abs_norm_pair_lag_bounds[0] < np.abs(self.norm_pair_lag[split])
                USE &= abs_norm_pair_lag_bounds[1] > np.abs(self.norm_pair_lag[split])

                try:
                    (fit_avx, fit_avy), cpars = sp.optimize.curve_fit(
                        norm_lag_flat,
                        self.c_da[USE],
                        self.norm_pair_lag[split][USE],
                        p0=[vx0, vy0],
                        bounds=bounds,
                        maxfev=10000,
                    )

                    TSS = np.square(self.norm_pair_lag[split][USE]).sum()
                    RSS = np.square(
                        self.norm_pair_lag[split][USE]
                        - norm_lag_flat(self.c_da[USE], fit_avx, fit_avy)
                    ).sum()
                    co_r_2 = RSS / TSS

                    if not co_r_2 < 0.5:
                        assert False
                    motion.loc[len(motion)] = (
                        self.time[s:e].mean(),
                        (e - s) * self.dt,
                        d_val,
                        azim_v,
                        fit_avx,
                        fit_avy,
                        co_r_2,
                    )

                except:
                    continue

        return motion
