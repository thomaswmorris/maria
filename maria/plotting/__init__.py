from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from matplotlib.patches import Patch

from ..utils.signal import detrend as detrend_signal


def plot_tod(
    tod,
    detrend: bool = True,
    n_freq_bins: int = 256,
    max_dets: int = 100,
    lw: float = 1e0,
    fontsize: float = 8,
):
    fig, axes = plt.subplots(
        ncols=2, nrows=len(tod.fields), sharex="col", figsize=(8, 4), dpi=160
    )
    axes = np.atleast_2d(axes)
    gs = axes[0, 0].get_gridspec()

    plt.subplots_adjust(top=0.99, bottom=0.01, hspace=0, wspace=0.02)

    for ax in axes[:, -1]:
        ax.remove()
    ps_ax = fig.add_subplot(gs[:, -1])

    handles = []
    colors = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    for i, field in enumerate(tod.fields):
        data = tod.get_field(field)

        tod_ax = axes[i, 0]

        for band_name in np.unique(tod.dets.band_name):
            color = next(colors)

            band_mask = tod.dets.band_name == band_name
            signal = data[band_mask]

            if band_mask.sum() > max_dets:
                signal = signal[
                    np.random.choice(np.arange(len(signal)), max_dets, replace=False)
                ]

            detrended_signal = detrend_signal(signal, order=1)
            if detrend:
                signal = detrended_signal

            f, ps = sp.signal.periodogram(detrended_signal, fs=tod.fs, window="tukey")

            f_bins = np.geomspace(f[1], f[-1], n_freq_bins)
            f_mids = np.sqrt(f_bins[1:] * f_bins[:-1])

            binned_ps = sp.stats.binned_statistic(
                f, ps.mean(axis=0), bins=f_bins, statistic="mean"
            )[0]

            use = binned_ps > 0

            ps_ax.plot(
                f_mids[use],
                binned_ps[use],
                lw=lw,
                color=color,
            )
            tod_ax.plot(
                tod.time,
                signal[0],
                lw=lw,
                color=color,
            )

            handles.append(Patch(color=color, label=f"{field} ({band_name})"))

        tod_ax.set_xlim(tod.time.min(), tod.time.max())

        # if tod.units == "K_RJ":
        #     ylabel = f"{field} [K]"
        # elif tod.units == "K_CMB":
        #     ylabel = rf"{field} [K]"
        # elif tod.units == "pW":
        #     ylabel = f"{field} [pW]"

        # tod_ax.set_ylabel(tod.units)

        label = tod_ax.text(
            0.01,
            0.99,
            f"{field} [{tod.units}]",
            fontsize=fontsize,
            ha="left",
            va="top",
            transform=tod_ax.transAxes,
        )
        label.set_bbox(dict(facecolor="white", alpha=0.8))

        # if i + 1 < n_fields:
        # tod_ax.set_xticklabels([])

    tod_ax.set_xlabel("Time [s]", fontsize=fontsize)

    ps_ax.yaxis.tick_right()
    ps_ax.yaxis.set_label_position("right")
    ps_ax.set_xlabel("Frequency [Hz]", fontsize=fontsize)
    ps_ax.set_ylabel(f"[{tod.units}$^2$/Hz]", fontsize=fontsize)
    ps_ax.legend(handles=handles, loc="upper right", fontsize=fontsize)
    ps_ax.loglog()
    ps_ax.set_xlim(f_mids.min(), f_mids.max())
