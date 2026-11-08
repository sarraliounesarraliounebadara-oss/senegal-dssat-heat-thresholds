"""Single-sine degree days and exposure time (Baskerville & Emin 1969; Schlenker & Roberts 2009).
The diurnal cycle is a sine between Tmin and Tmax. For a threshold b:
  DD(b)  = daily mean of max(T(t) - b, 0)            [degC day]
  TA(b)  = fraction of the day with T(t) > b          [day]
"""
import numpy as np


def degree_days(tmin, tmax, b):
    lo = np.minimum(tmin, tmax)
    hi = np.maximum(tmin, tmax)
    a = 0.5 * (hi + lo)
    w = 0.5 * (hi - lo)
    out = np.where(lo >= b, a - b, 0.0)
    part = (hi > b) & (lo < b)
    with np.errstate(invalid="ignore", divide="ignore"):
        th = np.arcsin(np.clip((b - a) / np.where(w > 0, w, 1), -1, 1))
        dd = ((a - b) * (np.pi / 2 - th) + w * np.cos(th)) / np.pi
    out = np.where(part, dd, out)
    return np.where(np.isnan(tmin) | np.isnan(tmax), np.nan, out)


def time_above(tmin, tmax, b):
    lo = np.minimum(tmin, tmax)
    hi = np.maximum(tmin, tmax)
    a = 0.5 * (hi + lo)
    w = 0.5 * (hi - lo)
    out = np.where(lo >= b, 1.0, 0.0)
    part = (hi > b) & (lo < b)
    with np.errstate(invalid="ignore", divide="ignore"):
        th = np.arcsin(np.clip((b - a) / np.where(w > 0, w, 1), -1, 1))
        ta = (np.pi / 2 - th) / np.pi
    out = np.where(part, ta, out)
    return np.where(np.isnan(tmin) | np.isnan(tmax), np.nan, out)
