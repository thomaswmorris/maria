from __future__ import annotations

import scipy as sp


def lowpass(data, fc, sample_rate, method="bessel", order=1, axis=-1):
    if method == "bessel":
        return bessel_lowpass(
            data=data,
            fc=fc,
            sample_rate=sample_rate,
            order=order,
            axis=axis,
        )  # noqa
    else:
        raise ValueError(f"Invalid method '{method}'.")


def highpass(data, fc, sample_rate, method="bessel", order=1, axis=-1):
    if method == "bessel":
        return bessel_highpass(
            data=data,
            fc=fc,
            sample_rate=sample_rate,
            order=order,
            axis=axis,
        )  # noqa
    else:
        raise ValueError(f"Invalid method '{method}'.")


def bandpass(data, f_lower, f_upper, sample_rate, method="bessel", order=1, axis=-1):
    kwargs = {
        "sample_rate": sample_rate,
        "order": order,
        "axis": axis,
        "method": method,
    }  # noqa
    if method == "bessel":
        return bessel_highpass(
            bessel_lowpass(data, f_upper, **kwargs),
            f_lower,
            **kwargs,
        )  # noqa
    else:
        raise ValueError(f"Invalid method '{method}'.")


def bessel_lowpass(data, fc, sample_rate, order=1, axis=-1):
    sos = sp.signal.bessel(
        2 * (order + 1),
        2 * fc / sample_rate,
        analog=False,
        btype="low",
        output="sos",
    )  # noqa
    return sp.signal.sosfilt(sos, data, axis=axis)


def bessel_highpass(data, fc, sample_rate, order=1, axis=-1):
    sos = sp.signal.bessel(
        2 * (order + 1),
        2 * fc / sample_rate,
        analog=False,
        btype="high",
        output="sos",
    )  # noqa
    return sp.signal.sosfilt(sos, data, axis=axis)
