from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from ..units import Quantity
from .projection import ProjectionMap


def _extract_2d(m: ProjectionMap, stokes: str = "I", nu_index: int = 0, t_index: int = 0) -> np.ndarray:
    d = m.data
    if "stokes" in m.dims:
        d = d[m.stokes.index(stokes)]
    if "nu" in m.dims:
        d = d[nu_index]
    if "v" in m.dims:
        d = d[0]
    if "t" in m.dims:
        d = d[t_index]
    return np.asarray(d.compute(), dtype=float)


def _resample_to_grid(
    source_map: ProjectionMap,
    target_map: ProjectionMap,
    stokes: str = "I",
    nu_index: int = 0,
    t_index: int = 0,
) -> np.ndarray:
    """Resample source_map onto target_map's pixel grid via bilinear interpolation.

    x_side / y_side are angular offsets from each map's own centre.  When the
    two maps have different centres we must shift the query points so they are
    expressed in source-map-centred coordinates before interpolating.
    """
    f_src = _extract_2d(source_map, stokes, nu_index, t_index)

    # eta is descending (top→bottom); RegularGridInterpolator needs ascending
    y_src = source_map.eta.rad[::-1]
    x_src = source_map.xi.rad

    interp = sp.interpolate.RegularGridInterpolator(
        (y_src, x_src),
        f_src[::-1, :],
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    # Centre offset: how much to add to target offsets to get source offsets.
    # RA offset is multiplied by cos(dec) to convert from angle on sky to the
    # gnomonic x-offset used by both maps.
    mean_dec = 0.5 * (source_map.center[1].rad + target_map.center[1].rad)
    delta_x = (target_map.center[0].rad - source_map.center[0].rad) * np.cos(mean_dec)
    delta_y = target_map.center[1].rad - source_map.center[1].rad

    YY, XX = np.meshgrid(target_map.eta.rad, target_map.xi.rad, indexing="ij")
    pts = np.stack([(YY + delta_y).ravel(), (XX + delta_x).ravel()], axis=-1)
    return interp(pts).reshape(target_map.n_eta, target_map.n_xi)


def _plot_map_panel(
    fig,
    ax,
    m: ProjectionMap,
    stokes: str = "I",
    nu_index: int = 0,
    t_index: int = 0,
    title: str = "",
    cmap: str = "CMRmap",
    vmin: float = None,
    vmax: float = None,
):
    d = _extract_2d(m, stokes, nu_index, t_index)

    # Compute pixel bin edges from centres for pcolormesh
    xi_res = m.xi_res.rad
    eta_res = m.eta_res.rad
    x_bins = np.append(m.xi.rad - xi_res / 2, m.xi.rad[-1] + xi_res / 2)
    y_bins = np.append(m.eta.rad - eta_res / 2, m.eta.rad[-1] + eta_res / 2)

    X = np.r_[x_bins, y_bins]
    hu = Quantity(X, "rad").hu
    x_vals = Quantity(x_bins, "rad").human_value
    y_vals = Quantity(y_bins, "rad").human_value
    ang_unit = hu["units"]

    if vmin is None or vmax is None:
        finite = d[np.isfinite(d)]
        _vmin, _vmax = np.nanquantile(finite, [1e-3, 1 - 1e-3]) if finite.size > 0 else (0.0, 1.0)
        vmin = vmin if vmin is not None else _vmin
        vmax = vmax if vmax is not None else _vmax

    ax.pcolormesh(x_vals, y_vals, d, cmap=cmap, vmin=vmin, vmax=vmax)

    fig.colorbar(
        mpl.cm.ScalarMappable(mpl.colors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap),
        ax=ax,
        shrink=0.75,
        aspect=16,
        location="right",
        pad=0,
        label=rf"${Quantity(d[np.isfinite(d)].mean() if np.isfinite(d).any() else 0, m.units).hu['math_name']}$",
    )

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlabel(rf"$\Delta\theta\;[{ang_unit}]$")
    ax.set_ylabel(rf"$\Delta\theta\;[{ang_unit}]$")


def compute_transfer_function(
    input_map: ProjectionMap,
    output_map: ProjectionMap,
    n_bins: int = 30,
    stokes: str = "I",
    nu_index: int = 0,
    t_index: int = 0,
    window: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the azimuthally-averaged spatial transfer function via cross-correlation.

    Uses T(u) = Re⟨F_in*(u) · F_out(u)⟩ / ⟨|F_in(u)|²⟩ rather than the naive
    power ratio P_out/P_in. Because noise in the output map is uncorrelated with
    the input model, it averages to zero in the cross-spectrum numerator and does
    not bias the estimate.

    A value of 1 indicates perfect signal recovery at that angular scale; values
    below 1 indicate signal suppression (e.g. from filtering or atmospheric removal).

    Parameters
    ----------
    input_map : ProjectionMap
        The input sky map injected into the simulation.
    output_map : ProjectionMap
        The recovered map produced by a mapper.
    n_bins : int
        Number of logarithmically-spaced spatial frequency bins.
    stokes : str
        Stokes parameter to use ("I", "Q", "U", or "V").
    nu_index : int
        Frequency channel index for multi-channel maps.
    t_index : int
        Time index for time-varying maps.
    window : bool
        Apply a 2D Hann window before FFT to reduce spectral leakage.

    Returns
    -------
    u : np.ndarray
        Spatial frequency bin centres in cycles per radian.
    T : np.ndarray
        Transfer function values (dimensionless).
    """
    # Bring both maps to the same units so the cross-spectrum ratio is dimensionless
    if output_map.units != input_map.units:
        output_map = output_map.to(input_map.units)

    # Resample input onto the output grid so both live on exactly the same pixels.
    # Working on the output grid is correct: its resolution and coverage define
    # the spatial frequencies at which the transfer function is meaningful.
    f_in = _resample_to_grid(input_map, output_map, stokes, nu_index, t_index)
    f_out = _extract_2d(output_map, stokes, nu_index, t_index)

    ny, nx = f_out.shape

    # Restrict to pixels observed in both maps; zero elsewhere so the FFT is clean
    valid = np.isfinite(f_in) & np.isfinite(f_out)
    f_in = np.where(valid, f_in, 0.0)
    f_out = np.where(valid, f_out, 0.0)

    # Energy-preserving 2D Hann window to reduce spectral leakage
    if window:
        win = np.outer(np.hanning(ny), np.hanning(nx))
        win /= np.sqrt(np.mean(win**2))
        f_in = f_in * win
        f_out = f_out * win

    # 2D FFTs (shifted so DC is at centre)
    F_in = np.fft.fftshift(np.fft.fft2(f_in))
    F_out = np.fft.fftshift(np.fft.fft2(f_out))

    # Cross-spectrum numerator and auto-power denominator.
    # Noise in F_out is uncorrelated with F_in, so Re(F_in* · N) → 0 when averaged,
    # leaving only the signal contribution in the numerator.
    P_in = np.abs(F_in) ** 2
    C = np.real(np.conj(F_in) * F_out)

    # Radial spatial frequency at each Fourier pixel — defined by the output grid
    dx = output_map.xi_res.rad
    dy = abs(output_map.eta_res.rad)
    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
    KX, KY = np.meshgrid(kx, ky)
    K = np.hypot(KX, KY)

    # Logarithmically-spaced radial bins (skip DC at k=0; Nyquist set by output pixel size)
    k_min = max(1.0 / (nx * dx), 1.0 / (ny * dy))
    k_max = 0.5 * min(1.0 / dx, 1.0 / dy)
    bins = np.geomspace(k_min, k_max, n_bins + 1)
    k_centers = np.sqrt(bins[:-1] * bins[1:])  # geometric midpoints

    # Azimuthally-averaged transfer function via binned cross/auto power ratio
    bin_idx = np.digitize(K.ravel(), bins) - 1
    in_range = (bin_idx >= 0) & (bin_idx < n_bins)

    sum_P_in = np.bincount(bin_idx[in_range], weights=P_in.ravel()[in_range], minlength=n_bins)
    sum_C = np.bincount(bin_idx[in_range], weights=C.ravel()[in_range], minlength=n_bins)

    T = np.where(sum_P_in > 0, sum_C / sum_P_in, np.nan)

    return k_centers, T


class TransferFunction:
    """Result of a spatial transfer function computation.

    Attributes
    ----------
    u : np.ndarray
        Spatial frequency bin centres in cycles per radian.
    T : np.ndarray
        Transfer function values (dimensionless).
    input_map : ProjectionMap or None
        The input sky map used for the computation.
    output_map : ProjectionMap or None
        The recovered map used for the computation.
    """

    def __init__(self, u, T, input_map=None, output_map=None):
        self.u = u
        self.T = T
        self.input_map = input_map
        self.output_map = output_map

    def plot(self, ax=None, x_unit="arcmin", filename=None, stokes="I", nu_index=0, t_index=0):
        """Plot the transfer function, with input/output map panels when available.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes for the transfer function panel. Ignored when maps are stored.
        x_unit : str
            Angular unit for the x-axis: ``"arcsec"``, ``"arcmin"``, or ``"deg"``.
        filename : str, optional
            Save the figure to this path when provided.
        stokes, nu_index, t_index
            Slice selectors forwarded to the map panels.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        return plot_transfer_function(
            self.u,
            self.T,
            input_map=self.input_map,
            output_map=self.output_map,
            stokes=stokes,
            nu_index=nu_index,
            t_index=t_index,
            ax=ax,
            x_unit=x_unit,
            filename=filename,
        )

    def __repr__(self):
        return (
            f"TransferFunction:\n"
            f"  bins: {len(self.u)}\n"
            f"  u: [{self.u.min():.3g}, {self.u.max():.3g}] cycles/rad\n"
            f"  T: [{np.nanmin(self.T):.3f}, {np.nanmax(self.T):.3f}]"
        )


_RAD_TO_DEG = 180.0 / np.pi
_ANGULAR_UNITS = {
    "arcsec": 3600.00 * _RAD_TO_DEG,
    "arcmin": 60.00 * _RAD_TO_DEG,
    "deg": _RAD_TO_DEG,
    "rad": 1.0,
}


def plot_transfer_function(
    u: np.ndarray,
    T: np.ndarray,
    input_map: ProjectionMap = None,
    output_map: ProjectionMap = None,
    add_beam: bool = True,
    stokes: str = "I",
    nu_index: int = 0,
    t_index: int = 0,
    ax=None,
    x_unit: str = "arcmin",
    filename: str = None,
) -> plt.Axes:
    """
    Plot the spatial transfer function, optionally alongside the input and output maps.

    Parameters
    ----------
    u : np.ndarray
        Spatial frequency bin centres in cycles per radian, as returned by
        ``compute_transfer_function``.
    T : np.ndarray
        Transfer function values.
    input_map : ProjectionMap, optional
        When provided, shown as the left panel of a three-panel figure.
    output_map : ProjectionMap, optional
        When provided, shown as the centre panel. Converted to ``input_map``
        units for display when both are given.
    add_beam : bool
        When True, plot the theoretical Gaussian beam curve corresponding to the
        output map's beam FWHM (if available) for comparison.
    stokes : str
        Stokes parameter to extract for the map panels.
    nu_index : int
        Frequency channel index for the map panels.
    t_index : int
        Time index for the map panels.
    ax : matplotlib.axes.Axes, optional
        Axes for the transfer function panel. Ignored when maps are supplied
        (a new figure is always created in that case).
    x_unit : str
        Angular unit for the transfer function x-axis: ``"arcsec"``,
        ``"arcmin"`` (default), or ``"deg"``.
    filename : str, optional
        Save the figure to this path when provided.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The transfer function axes.
    """
    show_maps = input_map is not None or output_map is not None

    if show_maps:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
        ax_in, ax_out, ax_tf = axes

        # Use input map units as the display reference for both panels
        display_units = input_map.units if input_map is not None else output_map.units
        out_for_display = (
            output_map.to(display_units) if (output_map is not None and output_map.units != display_units) else output_map
        )

        # Derive shared colour scale from the input map
        if input_map is not None:
            f_ref = _extract_2d(input_map, stokes, nu_index, t_index)
            finite = f_ref[np.isfinite(f_ref)]
            vmin, vmax = np.nanquantile(finite, [1e-3, 1 - 1e-3]) if finite.size > 0 else (None, None)
        else:
            vmin = vmax = None

        if input_map is not None:
            _plot_map_panel(fig, ax_in, input_map, stokes, nu_index, t_index, title="Input", vmin=vmin, vmax=vmax)
        else:
            ax_in.set_visible(False)

        if out_for_display is not None:
            _plot_map_panel(fig, ax_out, out_for_display, stokes, nu_index, t_index, title="Output", vmin=vmin, vmax=vmax)
        else:
            ax_out.set_visible(False)

        ax = ax_tf

    elif ax is None:
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

    # Transfer function panel
    factor = _ANGULAR_UNITS.get(x_unit, _ANGULAR_UNITS["arcmin"])
    theta = factor / u  # cycles/rad → angular scale in x_unit

    ax.axhline(1.0, color="gray", lw=1.0, ls="--", zorder=0)
    ax.plot(theta, T, color="steelblue", lw=1.5, marker="o", ms=3, label="Measured")

    # Theoretical Gaussian beam curve from the output map's beam attribute
    beam_map = output_map if output_map is not None else input_map

    if add_beam and (beam_map is not None) and hasattr(beam_map, "beam"):
        fwhm_rad = float(np.nanmean(beam_map.beam[..., 0].rad))
        if fwhm_rad > 0:
            u_dense = np.geomspace(u.min(), u.max(), 500)
            B_theory = np.exp(-(np.pi**2) * fwhm_rad**2 * u_dense**2 / (4.0 * np.log(2.0)))
            ax.plot(factor / u_dense, B_theory, color="tomato", lw=1.5, ls="--", label="Beam")

    ax.legend(frameon=False, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel(f"Angular scale [{x_unit}]")
    ax.set_ylabel("Transfer function")
    ax.set_ylim(0, 1.2)
    ax.set_xlim(theta.min(), theta.max())

    if filename is not None:
        ax.get_figure().savefig(filename, dpi=150)

    return ax
