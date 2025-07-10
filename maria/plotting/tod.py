from __future__ import annotations

import logging
import warnings
from itertools import cycle

import arrow
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from matplotlib.animation import FuncAnimation
from matplotlib.collections import EllipseCollection
from matplotlib.patches import Patch

from maria.beam import compute_angular_fwhm
from maria.units import Quantity, parse_units
from maria.utils.signal import detrend as detrend_signal

logger = logging.getLogger("maria")

FIELD_LABELS = {"atmosphere": "atm."}


def plot_tod(
    tod,
    detrend: bool = True,
    n_freq_bins: int = 1024,
    max_dets: int = 1,
    lw: float = 1e0,
    fontsize: float = 10,
):
    fig, axes = plt.subplots(
        ncols=2,
        nrows=len(tod.fields),
        sharex="col",
        figsize=(10, 4),
        dpi=256,
    )
    axes = np.atleast_2d(axes)
    gs = axes[0, 0].get_gridspec()

    plt.subplots_adjust(top=0.99, bottom=0.01, hspace=0, wspace=0.025)

    for ax in axes[:, -1]:
        ax.remove()
    ps_ax = fig.add_subplot(gs[:, -1])

    handles = []
    colors = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    tod_ubands = np.unique(tod.dets.band_name)
    power_spectra = np.zeros((len(tod_ubands), len(tod.fields), n_freq_bins - 1))

    tod_u = parse_units(tod.units)

    for field_index, field in enumerate(tod.fields):
        data = tod.data[field]

        tod_ax = axes[field_index, 0]
        field_label = FIELD_LABELS.get(field, field)

        field_plot_data = {}

        for band_index, band_name in enumerate(tod_ubands):
            band_mask = tod.dets.band_name == band_name

            field_plot_data[band_name] = data[band_mask]

            if band_mask.sum() > max_dets:
                field_plot_data[band_name] = field_plot_data[band_name][
                    np.linspace(0, band_mask.sum() - 1, max_dets).astype(int)
                ]

            if detrend:
                field_plot_data[band_name] = detrend_signal(field_plot_data[band_name], order=1)

        max_abs_signal = Quantity(
            [np.abs(field_plot_data[band_name]).max().compute() for band_name in tod_ubands], units=tod.units
        ).max()
        field_u = max_abs_signal.u
        logger.debug(
            f"inferring units '{max_abs_signal.u['units']}' for field '{field}' from max abs signal '{max_abs_signal}'"
        )

        for band_index, band_name in enumerate(tod_ubands):
            color = next(colors)

            band_mask = tod.dets.band_name == band_name
            f, ps = sp.signal.periodogram(
                detrend_signal(tod.data[field][band_mask].compute(), order=1), fs=tod.fs, window="hann"
            )

            f_bins = np.geomspace(0.999 * f[1], 1.001 * f[-1], n_freq_bins)
            f_mids = np.sqrt(f_bins[1:] * f_bins[:-1])

            binned_ps = sp.stats.binned_statistic(
                f,
                ps.mean(axis=0),
                bins=f_bins,
                statistic="mean",
            )[0]

            power_spectra[band_index, field_index] = binned_ps

            use = binned_ps > 0

            ps_ax.plot(
                f_mids[use],
                binned_ps[use],
                lw=5e-1,
                color=color,
            )
            tod_ax.plot(
                tod.time,
                Quantity(field_plot_data[band_name].T, units=tod.units).to(field_u["units"]),
                lw=5e-1,
                color=color,
                zorder=-band_index,
            )

            handles.append(Patch(color=color, label=f"{field_label} ({band_name}, $n_\\text{{det}} = {band_mask.sum()}$)"))

        tod_ax.set_xlim(tod.time.min(), tod.time.max() + 1e-1)
        tod_ax.set_ylabel(f"{field_label} [${field_u['math_name']}$]", fontsize=fontsize)

        if field_index == 0:
            time_label_ax = tod_ax.twiny()

    tod_ax.set_xlabel("Timestamp [s]", fontsize=fontsize)

    time_label_ticks = np.linspace(tod.time.min(), tod.time.max(), 3)
    time_label_ticks -= time_label_ticks % 60
    time_label_ax.set_xticks(np.unique(time_label_ticks))
    time_label_ax.set_xticklabels([arrow.get(t).strftime("%Y-%m-%d\n%H:%M:%S") for t in time_label_ax.get_xticks()])
    time_label_ax.set_xlim(tod.time.min(), tod.time.max() + 1e-1)

    ps_ax.yaxis.tick_right()
    ps_ax.yaxis.set_label_position("right")
    ps_ax.set_xlabel("Frequency [Hz]", fontsize=fontsize)
    ps_ax.set_ylabel(f"Power spectral density [${tod_u['math_name']}^2$/Hz]", fontsize=fontsize)
    ps_ax.legend(handles=handles, loc="upper right", fontsize=0.8 * fontsize)
    ps_ax.loglog()
    ps_ax.set_xlim(f_mids.min(), f_mids.max())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # I want to see at least the top order of magnitude of all fields
        minimum_dominant_ps = np.nanmin(np.nanmax(power_spectra, axis=1))
        minimum_maximum_ps = np.nanmin(np.nanmax(power_spectra, axis=2))

    ps_ax.set_ylim(1e-1 * np.minimum(minimum_dominant_ps, minimum_maximum_ps))


def twinkle_plot(tod, rate=2, fps=30, start_index=0, max_frames=100, filename=None):
    fps = np.minimum(fps, tod.fs)
    time_offset = tod.time.min()

    frame_time = np.arange(tod.time.min(), tod.time.max(), rate / fps)[:max_frames]
    frame_index = np.interp(frame_time, tod.time, np.arange(len(tod.time))).astype(int)

    offsets = Quantity(np.c_[tod.dets.sky_x, tod.dets.sky_y], "rad")
    fwhms = Quantity(compute_angular_fwhm(fwhm_0=tod.dets.primary_size, nu=tod.dets.band_center), "rad")

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
        dpi=256,
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

        ax.set_xlabel(r"$\Delta \theta_x$ [deg.]")
        ax.set_ylabel(r"$\Delta \theta_y$ [deg.]")

        boresight_info = ax.text(
            0.01,
            0.99,
            "",
            ha="left",
            va="top",
            transform=ax.transAxes,
        )

        band_qdata = Quantity(tod.signal[band_mask][..., frame_index].compute(), units=tod.units)

        cbar = fig.colorbar(ec, ax=ax, shrink=0.8, location="bottom")

        cbar.set_label(f"{band_qdata.q['long_name']} [${band_qdata.u['math_name']}$]")

        subplots[band] = {
            "ax": ax,
            "ec": ec,
            "cbar": cbar,
            "info": boresight_info,
            "time": tod.time[frame_index] - time_offset,
            "az": tod.boresight.az[frame_index],
            "el": tod.boresight.el[frame_index],
            "data": band_qdata.value,
        }

    for ax in axes_iterator:
        ax.set_axis_off()

    def update(frame, subplots):
        # i = frame_index[frame]

        for band, subplot in subplots.items():
            info = f"time = {time_offset:.00f} + {subplot['time'][frame]:.02f} s\naz = {np.degrees(subplot['az'][frame]):.02f} deg\nel = {np.degrees(subplot['el'][frame]):.02f} deg"  # noqa

            buffer = 4

            norm_start = np.maximum(0, frame - buffer)
            norm_end = np.minimum(len(frame_index) - 1, frame + buffer)

            # if frame < buffer:
            #     norm_start, norm_end = 0, 2 * buffer
            # if frame > len(t) - buffer:
            #     norm_start, norm_end = len(t) - 2 * buffer - 1, len(t) - 1

            norm = mpl.colors.Normalize(
                *np.quantile(subplot["data"][:, norm_start:norm_end], q=[0.01, 0.99]),
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
