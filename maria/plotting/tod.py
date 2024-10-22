from itertools import cycle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from matplotlib.animation import FuncAnimation
from matplotlib.collections import EllipseCollection
from matplotlib.patches import Patch

from maria.instrument.beam import compute_angular_fwhm
from maria.units import TOD_UNITS, Angle, parse_tod_units
from maria.utils.signal import detrend as detrend_signal


def tod_plot(
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


def twinkle_plot(tod, rate=2, fps=30, start_index=0, max_frames=100, filename=None):
    fps = np.minimum(fps, tod.fs)
    time_offset = tod.time.min()

    frame_time = np.arange(tod.time.min(), tod.time.max(), rate / fps)[:max_frames]
    frame_index = np.interp(frame_time, tod.time, np.arange(len(tod.time))).astype(int)

    offsets = Angle(np.c_[tod.dets.sky_x, tod.dets.sky_y])
    fwhms = Angle(
        compute_angular_fwhm(fwhm_0=tod.dets.primary_size, nu=tod.dets.band_center)
    )

    bands = sorted(np.unique(tod.dets.band_name))

    nrows = int(np.sqrt(len(bands)))
    ncols = int(np.ceil(len(bands) / nrows))
    subplot_width = np.minimum(6, 10 / nrows)
    subplot_height = np.minimum(6, 10 / ncols)
    subplot_size = np.minimum(subplot_width, subplot_height)
    figure_size = (ncols * subplot_size, nrows * (subplot_size + 1))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        sharex=True,
        sharey=True,
        figsize=figure_size,
        dpi=160,
        constrained_layout=True,
    )

    axes_iterator = iter(np.atleast_1d(axes).ravel())

    subplots = {}

    for band in bands:
        ax = next(axes_iterator)

        band_mask = tod.dets.band_name == band

        ec = EllipseCollection(
            widths=getattr(fwhms, offsets.units)[band_mask],
            heights=getattr(fwhms, offsets.units)[band_mask],
            angles=0,
            units="xy",
            offsets=getattr(offsets, offsets.units)[band_mask],
            transOffset=ax.transData,
            cmap="cmb",
        )

        ax.add_collection(ec)
        ax.scatter(
            *getattr(offsets, offsets.units)[band_mask].T,
            label=band,
            s=0,
            color="k",
        )

        # ax.axis('equal')

        ax.set_xlabel(r"$\Delta \theta_x$ [deg.]")
        ax.set_ylabel(r"$\Delta \theta_y$ [deg.]")

        boresight_info = ax.text(
            0.01, 0.99, "", ha="left", va="top", transform=ax.transAxes
        )

        cbar = fig.colorbar(ec, ax=ax, shrink=0.8, location="bottom")

        base_units = parse_tod_units("pW")["base"]

        cbar.set_label(f'{TOD_UNITS[base_units]["long_name"]} [{tod.units}]')

        subplots[band] = {
            "ax": ax,
            "ec": ec,
            "cbar": cbar,
            "info": boresight_info,
            "time": tod.time[frame_index] - time_offset,
            "az": tod.boresight.az[frame_index].compute(),
            "el": tod.boresight.el[frame_index].compute(),
            "data": tod.signal[band_mask][..., frame_index].compute(),
        }

    for ax in axes_iterator:
        ax.set_axis_off()

    def update(frame, subplots):
        # i = frame_index[frame]

        for band, subplot in subplots.items():
            info = f'time = {time_offset:.00f} + {subplot["time"][frame]:.02f} s\naz = {np.degrees(subplot["az"][frame]):.02f} deg\nel = {np.degrees(subplot["el"][frame]):.02f} deg'  # noqa

            buffer = 4

            norm_start = np.maximum(0, frame - buffer)
            norm_end = np.minimum(len(frame_index) - 1, frame + buffer)

            # if frame < buffer:
            #     norm_start, norm_end = 0, 2 * buffer
            # if frame > len(t) - buffer:
            #     norm_start, norm_end = len(t) - 2 * buffer - 1, len(t) - 1

            norm = mpl.colors.Normalize(
                *np.quantile(subplot["data"][:, norm_start:norm_end], q=[0.01, 0.99])
            )

            subplot["ec"].set_array(subplot["data"][:, frame])
            subplot["ec"].set_norm(norm)
            subplot["cbar"].update_normal(subplot["ec"])
            subplot["info"].set_text(info)  # noqa

        return subplots

    update(0, subplots)

    ani = FuncAnimation(
        fig=fig,
        func=update,
        fargs=(subplots,),
        frames=len(frame_index),
        interval=1e3 / fps,
    )

    if filename:
        ani.save(filename)
