"""
06_exposures.py
Growing-season heat and water exposures per department, crop and year, 1986-2016,
from the ANACIM merged CHIRTS Tmax/Tmin and ARC2 daily grids (0.0375 deg).

Season windows (window = [start, start + L_crop - 1]; L = 90 d millet, 100 d groundnut
and maize, 110 d sorghum), three calendars:
  - 'dynamic'   (main): start = sowing rain of the year, i.e. first day from 1 June to
                30 Sep with >= 20 mm over 3 days, first day wet (>= 1 mm); computed per
                pixel, department value = median over the crop mask
  - 'sivakumar' (robustness): Sivakumar (1988) onset, from 1 May: same rain criterion
                plus no dry spell > 7 days (P < 1 mm) in the 30 days starting at onset
  - 'static'    (robustness): median 1986-2016 dynamic start, same date every year
  If no pixel of a department has a start date in a year, the static date is used (flagged).

Exposures per pixel, then crop-weighted (MIRCA2000 harvested fraction x cos(lat))
department means:
  DD_b   degree days above b degC (single sine, b = 7..42), summed over the window
  TA_b   time above b degC (days), b = 20..46
  P_mm, rain_days (>= 1 mm), cdd_max (longest dry spell), n_dry7 / n_dry10 / n_dry15
  (dry spells reaching 7/10/15 days), tmax_mean, tmin_mean

The three merged files have no missing day over 1986-2016; six coastal ARC2 pixels (Dakar, Pikine,
Rufisque, Tivaouane) are always empty and are ignored in the rainfall averages.

Usage: python3 06_exposures.py <data_dir> <outputs_dir> <work_dir>
"""
import os
import sys
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from climate_io import open_climate, read_period, STUDY_PERIOD
from degree_days import degree_days, time_above

data, out, work = sys.argv[1:4]
Y0, Y1 = STUDY_PERIOD
YEARS = list(range(Y0, Y1 + 1))
CROPS = ["GROUNDNUT", "MAIZE", "MILLET", "SORGHUM"]
L = {"MILLET": 90, "GROUNDNUT": 100, "SORGHUM": 110, "MAIZE": 100}
CALS = ["dynamic", "sivakumar", "static"]
DD_B = list(range(7, 43))
TA_B = list(range(20, 47))
PIX_B = list(range(28, 41))          # pixel-level DD kept for maps (dynamic calendar)
CAL_REF = STUDY_PERIOD
JUN1 = 31                            # day index of 1 June (index 0 = 1 May)
SEP30 = 152                          # day index of 30 Sep

tx, tn, pr = open_climate(data)
dept = xr.open_dataset(f"{out}/department_mask_0p0375.nc").dept_id.values
masks = xr.open_dataset(f"{out}/crop_masks_0p0375.nc")
ids = pd.read_csv(f"{out}/department_ids.csv")
inside = ~np.isnan(dept)
npix = int(inside.sum())
dep_pix = dept[inside].astype(int)
lat2d = np.repeat(masks.Lat.values[:, None], masks.sizes["Lon"], axis=1)
coslat = np.cos(np.deg2rad(lat2d))[inside]

W, M = {}, {}
for c in CROPS:
    f = masks[f"frac_{c}"].values[inside]
    m = masks[f"mask_{c}"].values[inside] == 1
    w = np.zeros((len(ids), npix))
    for k, i in enumerate(ids.dept_id):
        sel = m & (dep_pix == i)
        w[k, sel] = f[sel] * coslat[sel]
    W[c], M[c] = w, m


def wmean(wmat, x):
    """Weighted department means of pixel vector x, ignoring NaN pixels."""
    ok = ~np.isnan(x)
    num = wmat @ np.where(ok, x, 0.0)
    den = wmat @ ok.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def sowing_rain(p):
    """First day (index from 1 May) in [1 Jun, 30 Sep] with >= 20 mm in 3 days, day 1 wet."""
    s3 = p[:-2] + p[1:-1] + p[2:]
    ok = (p[:-2] >= 1.0) & (s3 >= 20.0)
    ok[:JUN1] = False
    ok[SEP30 + 1:] = False
    return np.where(ok.any(axis=0), ok.argmax(axis=0), -1)


def sivakumar(p):
    """Sivakumar (1988) onset from 1 May; -1 if none before 30 Sep."""
    nd = p.shape[0]
    dry = p < 1.0
    run = np.zeros(p.shape, dtype=np.int16)
    r = np.zeros(p.shape[1], dtype=np.int16)
    for t in range(nd):
        r = np.where(dry[t], r + 1, 0).astype(np.int16)
        run[t] = r
    s3 = p[:-2] + p[1:-1] + p[2:]
    onset = np.full(p.shape[1], -1, dtype=np.int32)
    todo = ~np.isnan(p).all(axis=0)
    for d in range(0, SEP30 + 1):
        cand = todo & (p[d] >= 1.0) & (s3[d] >= 20.0)
        if not cand.any():
            continue
        mx = np.zeros(p.shape[1], dtype=np.int16)
        for t in range(d, min(d + 30, nd)):
            mx = np.maximum(mx, np.minimum(run[t], t - d + 1))
        ok = cand & (mx <= 7)
        onset[ok] = d
        todo &= ~ok
        if not todo.any():
            break
    return onset


# ---------------- pass 1: pixel start dates ----------------
f_on = f"{work}/pixel_start_dates_{Y0}-{Y1}.npz"
if os.path.exists(f_on):
    z = np.load(f_on); sow, siv = z["sow"], z["siv"]
else:
    sow = np.full((len(YEARS), npix), -1, dtype=np.int32)
    siv = np.full((len(YEARS), npix), -1, dtype=np.int32)
    for k, y in enumerate(YEARS):
        _, p = read_period(pr, f"{y}-05-01", f"{y + 1}-01-31")
        p = p[:, inside]
        sow[k], siv[k] = sowing_rain(p), sivakumar(p)
        print("start dates", y, "sow %.2f siv %.2f" % ((sow[k] >= 0).mean(), (siv[k] >= 0).mean()),
              flush=True)
    np.savez_compressed(f_on, sow=sow, siv=siv)


def dept_median(pix, c):
    o = np.full((len(YEARS), len(ids)), np.nan)
    v = np.zeros((len(YEARS), len(ids)))
    for j, i in enumerate(ids.dept_id):
        sel = M[c] & (dep_pix == i)
        if not sel.any():
            continue
        for k in range(len(YEARS)):
            x = pix[k, sel]
            v[k, j] = (x >= 0).mean()
            if (x >= 0).any():
                o[k, j] = np.median(x[x >= 0])
    return o, v


ref = [k for k, y in enumerate(YEARS) if CAL_REF[0] <= y <= CAL_REF[1]]
START, VALID, FALLBACK = {}, {}, {}
for c in CROPS:
    o_sow, v_sow = dept_median(sow, c)
    o_siv, v_siv = dept_median(siv, c)
    clim = np.nanmedian(o_sow[ref], axis=0)
    for cal, o, v in (("dynamic", o_sow, v_sow), ("sivakumar", o_siv, v_siv),
                      ("static", np.tile(clim, (len(YEARS), 1)), np.ones_like(v_sow))):
        FALLBACK[(c, cal)] = np.isnan(o)
        START[(c, cal)] = np.where(np.isnan(o), clim[None, :], o)
        VALID[(c, cal)] = v

# ---------------- pass 2: exposures ----------------
rows = []
pix_dd = {c: np.full((len(YEARS), len(PIX_B), npix), np.nan, dtype=np.float32) for c in CROPS}
pix_wat = {c: np.full((len(YEARS), 3, npix), np.nan, dtype=np.float32) for c in CROPS}
cols = np.arange(npix)
for k, y in enumerate(YEARS):
    days, p = read_period(pr, f"{y}-05-01", f"{y + 1}-01-31")
    _, a = read_period(tx, f"{y}-05-01", f"{y + 1}-01-31")
    _, b = read_period(tn, f"{y}-05-01", f"{y + 1}-01-31")
    p, a, b = p[:, inside], a[:, inside], b[:, inside]
    nd = len(days)

    win = {}
    for c in CROPS:
        for cal in CALS:
            o = START[(c, cal)][k]
            st = np.full(npix, -1)
            for j, i in enumerate(ids.dept_id):
                if not np.isnan(o[j]):
                    st[dep_pix == i] = int(round(o[j]))
            en = np.where(st >= 0, st + L[c] - 1, -1)
            if np.any(en >= nd):
                raise ValueError(f"Incomplete crop window: {c}, {cal}, {y}")
            win[(c, cal)] = (st, en)

    def wsum(x):
        """Window sums of x (days, npix); NaN if any NaN inside the window."""
        cs = np.vstack([np.zeros((1, npix)), np.cumsum(np.nan_to_num(x), axis=0)])
        cn = np.vstack([np.zeros((1, npix)), np.cumsum(np.isnan(x), axis=0)])
        outw = {}
        for key, (st, en) in win.items():
            ok = st >= 0
            s0, s1 = np.where(ok, st, 0), np.where(ok, en + 1, 0)
            v = cs[s1, cols] - cs[s0, cols]
            bad = (cn[s1, cols] - cn[s0, cols]) > 0
            outw[key] = np.where(ok & ~bad, v, np.nan)
        return outw

    res = {key: {} for key in win}
    for bb in DD_B:
        for key, v in wsum(degree_days(b, a, bb)).items():
            res[key][f"DD_{bb}"] = v
            if key[1] == "dynamic" and bb in PIX_B:
                pix_dd[key[0]][k, PIX_B.index(bb)] = v
    for bb in TA_B:
        for key, v in wsum(time_above(b, a, bb)).items():
            res[key][f"TA_{bb}"] = v
    for var, x in (("P_mm", p), ("rain_days", np.where(np.isnan(p), np.nan, (p >= 1.0) * 1.0)),
                   ("tmax_sum", a), ("tmin_sum", b)):
        for key, v in wsum(x).items():
            res[key][var] = v
    dry = p < 1.0
    for key, (st, en) in win.items():
        n = (en - st + 1).astype(float)
        n[st < 0] = np.nan
        res[key]["window_days"] = n
        res[key]["tmax_mean"] = res[key].pop("tmax_sum") / n
        res[key]["tmin_mean"] = res[key].pop("tmin_sum") / n
        r = np.zeros(npix); cdd = np.zeros(npix)
        n7 = np.zeros(npix); n10 = np.zeros(npix); n15 = np.zeros(npix)
        for t in range(nd):
            r = np.where(dry[t] & (t >= st) & (t <= en), r + 1, 0)
            cdd = np.maximum(cdd, r)
            n7 += r == 7; n10 += r == 10; n15 += r == 15
        for v, x in (("cdd_max", cdd), ("n_dry7", n7), ("n_dry10", n10), ("n_dry15", n15)):
            res[key][v] = np.where((st >= 0) & np.isfinite(res[key]["P_mm"]), x, np.nan)
        if key[1] == "dynamic":
            pix_wat[key[0]][k] = np.vstack([res[key]["P_mm"], res[key]["cdd_max"],
                                            res[key]["tmax_mean"]])

    for (c, cal), d in res.items():
        agg = {v: wmean(W[c], x) for v, x in d.items()}
        for j in range(len(ids)):
            if W[c][j].sum() == 0:
                continue
            row = {"department": ids.department[j], "crop": c, "calendar": cal, "year": y,
                   "start_day_from_1may": START[(c, cal)][k, j],
                   "start_from_fallback": bool(FALLBACK[(c, cal)][k, j]),
                   "frac_pixels_with_start": VALID[(c, cal)][k, j]}
            row.update({v: agg[v][j] for v in agg})
            rows.append(row)
    print("exposures", y, flush=True)

ex = pd.DataFrame(rows)
ex.to_csv(f"{work}/exposures_department_crop.csv", index=False, float_format="%.4f")
np.savez_compressed(f"{work}/pixel_exposures_dynamic.npz", years=np.array(YEARS),
                    thresholds=np.array(PIX_B), inside=inside,
                    **{f"DD_{c}": pix_dd[c] for c in CROPS},
                    **{f"WAT_{c}": pix_wat[c] for c in CROPS})
print(ex.shape)
