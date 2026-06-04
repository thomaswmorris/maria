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
        d = d[list(m.stokes).index(stokes)]
    if "nu" in m.dims:
        d = d[nu_index]
    if "v" in m.dims:
        d = d[0]
    if "z" in m.dims:
        d = d[0]
    if "t" in m.dims:
        d = d[t_index]
    return np.asarray(d.compute(), dtype=float).reshape(m.dims["eta"], m.dims["xi"])


def compute_transfer_function(
    input_map: ProjectionMap,
    output_map: ProjectionMap,
    n_bins: int = 30,
    stokes: str = "I",
    nu_index: int = 0,
    t_index: int = 0,
    window: str | bool | np.ndarray = "hann",
    taper: float = 0.1,
    pad_factor: int = 1,
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
    window : str, bool, or np.ndarray
        Apodization window applied before the FFT to reduce spectral leakage.
        - ``"tukey"``: separable Tukey window with
          cosine-tapered fraction ``taper`` on each edge; leaves the central
          region at unit weight.
        - ``"hann"`` (default) or ``True``: full Hann window (goes to zero at both edges).
        - ``np.ndarray`` of shape ``(ny, nx)``: custom 2D window applied as-is.
        - ``False`` or ``None``: no windowing.
    taper : float
        Fraction of each axis tapered by the cosine roll-off when
        ``window="tukey"``. Must be in [0, 1]. Default is 0.1.
    pad_factor : int
        Zero-pad each axis to ``pad_factor`` times its original length before
        the FFT. Increases the density of k-space samples (improves large-scale
        / low-wavenumber sampling) without changing the pixel size or Nyquist
        frequency. Must be >= 1. Default is 1 (no padding).

    Returns
    -------
    u : np.ndarray
        Spatial frequency bin centres in cycles per radian.
    T : np.ndarray
        Transfer function values (dimensionless).
    """
    if pad_factor < 1:
        raise ValueError("pad_factor must be >= 1")

    if output_map.units != input_map.units:
        output_map = output_map.to(input_map.units)

    f_in = _extract_2d(input_map.resample(output_map), stokes, nu_index, t_index)
    f_out = _extract_2d(output_map, stokes, nu_index, t_index)

    ny, nx = f_out.shape

    f_in = np.where(np.isfinite(f_in), f_in, 0.00)
    f_out = np.where(np.isfinite(f_out), f_out, 0.00)

    if window is not False and window is not None:
        if isinstance(window, np.ndarray):
            if window.shape != (ny, nx):
                raise ValueError(f"Custom window shape {window.shape} does not match map shape ({ny}, {nx})")
            win = window.astype(float)
        else:
            w_name = window if isinstance(window, str) else "hann"
            if w_name == "hann":
                wx, wy = np.hanning(nx), np.hanning(ny)
            elif w_name == "tukey":
                wx = sp.signal.windows.tukey(nx, alpha=taper)
                wy = sp.signal.windows.tukey(ny, alpha=taper)
            else:
                raise ValueError(f"Unsupported window type: {window}")
            win = np.outer(wy, wx)
        win /= np.nanmax(win)
        
        f_in = f_in * win
        f_out = f_out * win

    ny_pad, nx_pad = ny * pad_factor, nx * pad_factor
    F_in = np.fft.fftshift(np.fft.fft2(f_in, s=(ny_pad, nx_pad)))
    F_out = np.fft.fftshift(np.fft.fft2(f_out, s=(ny_pad, nx_pad)))

    # Cross-spectrum: noise in F_out is uncorrelated with F_in, so Re(F_in*·N)→0
    # when averaged, leaving only the signal contribution in the numerator.
    P_den = np.real(np.conj(F_in) * F_in)
    P_num = np.real(np.conj(F_in) * F_out)

    dx = output_map.xi_res.rad
    dy = abs(output_map.eta_res.rad)
    kx = np.fft.fftshift(np.fft.fftfreq(nx_pad, d=dx))
    ky = np.fft.fftshift(np.fft.fftfreq(ny_pad, d=dy))
    K = np.hypot(*np.meshgrid(kx, ky))

    k_min = max(1.0 / (nx_pad * dx), 1.0 / (ny_pad * dy))
    k_max = 0.5 * min(1.0 / dx, 1.0 / dy)
    bins = np.geomspace(k_min, k_max, n_bins + 1)
    u = np.sqrt(bins[:-1] * bins[1:])

    bin_idx = np.digitize(K.ravel(), bins) - 1
    mask = (bin_idx >= 0) & (bin_idx < n_bins)
    sum_P_den = np.bincount(bin_idx[mask], weights=P_den.ravel()[mask], minlength=n_bins)
    sum_P_num = np.bincount(bin_idx[mask], weights=P_num.ravel()[mask], minlength=n_bins)

    return u, np.where(sum_P_den > 0, sum_P_num / sum_P_den, np.nan)


class TransferFunction:
    """Result of a spatial transfer function computation.

    Attributes
    ----------
    u : np.ndarray
        Spatial frequency bin centres in cycles per radian, shape ``(n_bins,)``.
    T : np.ndarray
        Transfer function values, shape ``(n_nu, n_bins)``.
    nu : Quantity or None
        Frequency axis corresponding to the first dimension of ``T``.
    beam_fwhm : np.ndarray or None
        Per-channel beam FWHM in radians, shape ``(n_nu,)``.
    """

    def __init__(self, u, T, nu=None, beam_fwhm=None, input_map=None, output_map=None):
        self.u = u
        self.T = T
        self.nu = nu
        self.beam_fwhm = beam_fwhm
        self.input_map = input_map
        self.output_map = output_map

    def plot(self, ax=None, x_unit="arcmin", filename=None, add_beam=True, slices=None):
        """Plot the transfer function.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
        x_unit : str
            Angular unit for the x-axis: ``"arcsec"``, ``"arcmin"``, or ``"deg"``.
        filename : str, optional
            Save the figure to this path when provided.
        add_beam : bool
            Overlay the theoretical Gaussian beam curve per channel.
        slices : dict, optional
            Channel selection, e.g. ``dict(nu=[0, 2])``.  ``None`` plots all
            channels, consistent with how ``slices`` is used in map ``.plot()``.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        idx = list(np.atleast_1d(slices["nu"])) if (slices and "nu" in slices) else slice(None)
        return plot_transfer_function(
            self.u,
            self.T[idx],
            nu=self.nu[idx] if self.nu is not None else None,
            beam_fwhm=self.beam_fwhm[idx] if (add_beam and self.beam_fwhm is not None) else None,
            ax=ax,
            x_unit=x_unit,
            filename=filename,
        )

    def __repr__(self):
        n_nu = self.T.shape[0]
        nu_str = f"  nu: {self.nu}\n" if self.nu is not None else ""
        return (
            f"TransferFunction:\n"
            f"  channels: {n_nu}\n"
            f"{nu_str}"
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
    nu=None,
    beam_fwhm=None,
    ax=None,
    x_unit: str = "arcmin",
    filename: str = None,
) -> plt.Axes:
    """
    Plot the spatial transfer function.

    Parameters
    ----------
    u : np.ndarray
        Spatial frequency bin centres in cycles per radian.
    T : np.ndarray
        Transfer function values, shape ``(n_nu, n_bins)``.
    nu : Quantity, optional
        Frequency axis for labelling each curve.
    beam_fwhm : array-like of float, optional
        Per-channel beam FWHM in radians, shape ``(n_nu,)``.  When provided, a
        dashed Gaussian beam curve is overlaid for each channel.
    ax : matplotlib.axes.Axes, optional
    x_unit : str
        Angular unit for the x-axis: ``"arcsec"``, ``"arcmin"`` (default), or ``"deg"``.
    filename : str, optional
        Save the figure to this path when provided.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

    factor = _ANGULAR_UNITS.get(x_unit, _ANGULAR_UNITS["arcmin"])
    theta = factor / u
    n_nu = T.shape[0]
    colors = mpl.colormaps["viridis"](np.linspace(0.2, 0.85, n_nu)) if n_nu > 1 else ["steelblue"]
    u_dense = np.geomspace(u.min(), u.max(), 500) if beam_fwhm is not None else None

    ax.axhline(1.0, color="gray", lw=1.0, ls="--", zorder=0)

    for i, (T_row, color) in enumerate(zip(T, colors)):
        label = str(Quantity(nu[i], "Hz")) if nu is not None else "Measured"
        ax.plot(theta, T_row, color=color, lw=1.5, marker="o", ms=3, label=label)

        if beam_fwhm is not None:
            fwhm = float(beam_fwhm[i])
            if fwhm > 0:
                B = np.exp(-(np.pi**2) * fwhm**2 * u_dense**2 / (4.0 * np.log(2.0)))
                ax.plot(factor / u_dense, B, color=color, lw=1.5, ls="--")

    ax.legend(frameon=False, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel(f"Angular scale [{x_unit}]")
    ax.set_ylabel("Transfer function")
    ax.set_ylim(0, 1.2)
    ax.set_xlim(theta.min(), theta.max())

    if filename is not None:
        ax.get_figure().savefig(filename, dpi=150)

    return ax
