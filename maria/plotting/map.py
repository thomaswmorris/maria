import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from ..units import MAP_UNITS


def plot_map(
    m,
    nu_index=None,
    time_index=None,
    cmap="cmb",
    rel_vmin=0.001,
    rel_vmax=0.999,
    units="degrees",
    subplot_size=3,
    **kwargs,
):
    nu_index = np.atleast_1d(nu_index or np.arange(len(m.nu)))
    time_index = np.atleast_1d(time_index or np.arange(len(m.time)))

    n_nu = len(nu_index)
    n_time = len(time_index)
    n_maps = n_nu * n_time

    plot_width = np.maximum(6, subplot_size * n_nu)
    plot_height = np.maximum(6, subplot_size * n_time)

    if (n_nu > 1) and (n_time > 1):
        fig, axes = plt.subplots(
            len(time_index),
            len(nu_index),
            figsize=(plot_width, plot_height),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )

        axes = np.atleast_2d(axes)
        flat = False

    else:
        n_rows = int(np.sqrt(n_maps))
        n_cols = int(np.ceil(n_maps / n_rows))
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(6, 6),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )

        axes = np.atleast_1d(axes).ravel()
        flat = True

    axes_generator = iter(axes)

    d = m.data.ravel()
    w = m.weight.ravel()
    subset = np.random.choice(d.size, size=10000)
    vmin, vmax = np.quantile(
        d[subset], weights=w[subset], q=[rel_vmin, rel_vmax], method="inverted_cdf"
    )

    for i_t in time_index:
        for i_nu in nu_index:
            ax = next(axes_generator) if flat else axes[i_nu, i_t]

            nu = m.nu[i_nu]

            header = fits.header.Header()

            header["RESTFRQ"] = nu

            res_degrees = np.degrees(m.resolution)
            center_degrees = np.degrees(m.center)

            header["CDELT1"] = res_degrees  # degree
            header["CDELT2"] = res_degrees  # degree

            header["CRPIX1"] = m.n_x / 2
            header["CRPIX2"] = m.n_y / 2

            header["CTYPE1"] = "RA---SIN"
            header["CUNIT1"] = "deg     "
            header["CTYPE2"] = "DEC--SIN"
            header["CUNIT2"] = "deg     "
            header["RADESYS"] = "FK5     "

            header["CRVAL1"] = center_degrees[0]
            header["CRVAL2"] = center_degrees[1]
            wcs_input = WCS(header, naxis=2)  # noqa F401

            # ax = fig.add_subplot(len(time_index), len(nu_index), i_ax, sharex=True)  # , projection=wcs_input)

            ax.set_title(f"{nu} GHz")

            map_extent_radians = [
                -m.width / 2,
                m.width / 2,
                -m.height / 2,
                m.height / 2,
            ]

            if units == "degrees":
                map_extent = np.degrees(map_extent_radians)
            if units == "arcmin":
                map_extent = 60 * np.degrees(map_extent_radians)
            if units == "arcsec":
                map_extent = 3600 * np.degrees(map_extent_radians)

            ax.imshow(
                m.data[i_t, i_nu].T,
                cmap=cmap,
                interpolation="none",
                extent=map_extent,
                vmin=vmin,
                vmax=vmax,
            )

            ax.set_xlabel(rf"$\Delta\,\theta_x$ [{units}]")
            ax.set_ylabel(rf"$\Delta\,\theta_y$ [{units}]")

        dummy_map = mpl.cm.ScalarMappable(
            mpl.colors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap
        )

        cbar = fig.colorbar(
            dummy_map, ax=axes, shrink=0.75, aspect=16, location="bottom"
        )
        cbar.set_label(f'{MAP_UNITS[m.units]["long_name"]} [{m.units}]')
