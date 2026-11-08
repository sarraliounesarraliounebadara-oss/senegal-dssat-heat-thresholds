"""Exposures for X2 and X3, with the windows and code of 06_exposures.py (dynamic calendar), millet and groundnut.

X2, department level: Hargreaves PET (FAO-56 eq. 52, Ra from eq. 21), climatic water balance D = P - PET summed over
the AFM crop window, and SPEI: a three-parameter log-logistic fitted to D by L-moments (Vicente-Serrano et al. 2010)
for each pixel over 1986-2016, then the crop-weighted department mean (MIRCA fraction x cos(lat), as in 06).
X3, pixel level: every 06 exposure (DD_7..42, TA_20..46, P_mm, rain_days, cdd_max, window_days, tmax_mean,
tmin_mean) plus PET_mm, D_mm and SPEI for the pixels of the DSSAT 1/9 subsample, over their department window.
Check: department means of P_mm and DD_30 recomputed here must equal the AFM exposures file.
usage: python x23_exposures.py   -> x2x3/spei_department_crop.csv, x2x3/pixel_exposures_subsample.csv"""
import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
from scipy import special, stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402
AFM = paths.AFM
from climate_io import open_climate, read_period  # noqa: E402
from degree_days import degree_days, time_above  # noqa: E402

OUT = paths.DSSAT + "x2x3/"
os.makedirs(OUT, exist_ok=True)
YEARS = list(range(1986, 2017))
CROPS = ["MILLET", "GROUNDNUT"]
L = {"MILLET": 90, "GROUNDNUT": 100}
DD_B, TA_B = list(range(7, 43)), list(range(20, 47))

tx, tn, pr = open_climate(paths.DATA)
dept = xr.open_dataset(AFM + "outputs/department_mask_0p0375.nc").dept_id.values
masks = xr.open_dataset(AFM + "outputs/crop_masks_0p0375.nc")
ids = pd.read_csv(AFM + "outputs/department_ids.csv")
inside = ~np.isnan(dept)
npix = int(inside.sum())
dep_pix = dept[inside].astype(int)
lat2d = np.repeat(masks.Lat.values[:, None], masks.sizes["Lon"], axis=1)
coslat = np.cos(np.deg2rad(lat2d))[inside]
latpix = lat2d[inside]

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
    ok = ~np.isnan(x)
    num = wmat @ np.where(ok, x, 0.0)
    den = wmat @ ok.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


# windows: department median of the pixel sowing-rain dates (06 pass 1 cache), static fallback
sow = np.load(AFM + "work/pixel_start_dates_1986-2016.npz")["sow"]
START = {}
for c in CROPS:
    o = np.full((len(YEARS), len(ids)), np.nan)
    for j, i in enumerate(ids.dept_id):
        sel = M[c] & (dep_pix == i)
        if not sel.any():
            continue
        for k in range(len(YEARS)):
            x = sow[k, sel]
            if (x >= 0).any():
                o[k, j] = np.median(x[x >= 0])
    clim = np.nanmedian(o, axis=0)
    START[c] = np.where(np.isnan(o), clim[None, :], o)

# DSSAT 1/9 subsample pixels (same selection as e2_run.py) and their position among the 'inside' cells
pix = pd.read_csv(paths.DSSAT + "pipeline/pixels.csv")
pos = -np.ones(inside.shape, dtype=int)
pos[inside] = np.arange(npix)
sub = {}
for c in CROPS:
    s = pix[(pix.i % 3 == 1) & (pix.j % 3 == 1) & ~pix.rain_missing]
    s = s[masks[f"mask_{c}"].values[s.i, s.j] > 0]
    sub[c] = pd.DataFrame({"pix": s.index.values, "col": pos[s.i.values, s.j.values]})
    assert (sub[c].col >= 0).all()
    assert np.allclose(latpix[sub[c].col], s.lat.values)
cols_sub = np.unique(np.concatenate([sub[c].col.values for c in CROPS]))
csub = {c: np.searchsorted(cols_sub, sub[c].col.values) for c in CROPS}
dname = ids.set_index("dept_id").department


def ra_mm(latdeg, doy):
    """Extraterrestrial radiation (FAO-56 eq. 21) in mm/day of evaporation (x 0.408); doy (days,), lat (npix,)."""
    phi = np.deg2rad(latdeg)[None, :]
    j = doy[:, None].astype(float)
    dr = 1 + 0.033 * np.cos(2 * np.pi * j / 365)
    de = 0.409 * np.sin(2 * np.pi * j / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(de), -1, 1))
    ra = 24 * 60 / np.pi * 0.0820 * dr * (ws * np.sin(phi) * np.sin(de) + np.cos(phi) * np.cos(de) * np.sin(ws))
    return 0.408 * ra


def hargreaves(tmax, tmin, ra):
    return 0.0023 * (0.5 * (tmax + tmin) + 17.8) * np.sqrt(np.clip(tmax - tmin, 0, None)) * ra


def spei_fit(D):
    """SPEI from window water balances D (years, n): generalised log-logistic fitted per column by L-moments
    (unbiased probability-weighted moments; Hosking 1997), as in the SPEI package (Begueria et al. 2014);
    values beyond the fitted bound get the extreme probability 1e-4 or 1 - 1e-4."""
    out = np.full(D.shape, np.nan)
    nfb = 0
    for k in range(D.shape[1]):
        x = D[:, k]
        ok = np.isfinite(x)
        if ok.sum() < 10:
            continue
        xs = np.sort(x[ok]); n = len(xs); i = np.arange(1, n + 1)
        b0 = xs.mean()
        b1 = np.sum((i - 1) / (n - 1) * xs) / n
        b2 = np.sum((i - 1) * (i - 2) / ((n - 1) * (n - 2)) * xs) / n
        l1, l2, l3 = b0, 2 * b1 - b0, 6 * b2 - 6 * b1 + b0
        kap = -l3 / l2
        if abs(kap) < 1e-6:
            alpha, xi = l2, l1
            yv = (x[ok] - xi) / alpha
        else:
            alpha = l2 * np.sin(kap * np.pi) / (kap * np.pi)
            xi = l1 - alpha * (1 / kap - np.pi / np.sin(kap * np.pi))
            arg = 1 - kap * (x[ok] - xi) / alpha
            with np.errstate(invalid="ignore", divide="ignore"):
                yv = -np.log(arg) / kap
            nfb += int((arg <= 0).sum() > 0)
            yv = np.where(arg > 0, yv, np.inf * np.sign(kap))
        cdf = np.clip(1.0 / (1.0 + np.exp(-yv)), 1e-4, 1 - 1e-4)
        out[ok, k] = stats.norm.ppf(cdf)
    return out, nfb


D_all = {c: np.full((len(YEARS), npix), np.nan) for c in CROPS}
PET_all = {c: np.full((len(YEARS), npix), np.nan) for c in CROPS}
chk = []
prow = []
for k, y in enumerate(YEARS):
    days, p = read_period(pr, f"{y}-05-01", f"{y + 1}-01-31")
    _, a = read_period(tx, f"{y}-05-01", f"{y + 1}-01-31")
    _, b = read_period(tn, f"{y}-05-01", f"{y + 1}-01-31")
    p, a, b = p[:, inside], a[:, inside], b[:, inside]
    nd = len(days)
    pet = hargreaves(a, b, ra_mm(latpix, days.dayofyear.values))
    allc = np.arange(npix)
    for c in CROPS:
        st = np.full(npix, -1)
        for j, i in enumerate(ids.dept_id):
            if not np.isnan(START[c][k, j]):
                st[dep_pix == i] = int(round(START[c][k, j]))
        en = st + L[c] - 1
        assert en.max() < nd
        okw = st >= 0

        def wsum(x, cols=allc, st=st, en=en):
            cs = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(np.nan_to_num(x), axis=0)])
            cn = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(np.isnan(x), axis=0)])
            s0, s1 = np.where(st[cols] >= 0, st[cols], 0), np.where(st[cols] >= 0, en[cols] + 1, 0)
            ii = np.arange(x.shape[1])
            v = cs[s1, ii] - cs[s0, ii]
            bad = (cn[s1, ii] - cn[s0, ii]) > 0
            return np.where((st[cols] >= 0) & ~bad, v, np.nan)

        P_ = wsum(p); PET_ = wsum(pet)
        D_all[c][k] = P_ - PET_
        PET_all[c][k] = PET_
        # check against the AFM department file (P_mm and DD_30)
        chk.append(dict(crop=c, year=y, **{f"P_mm_{d}": v for d, v in zip(ids.department, wmean(W[c], P_))},
                        **{f"DD_30_{d}": v for d, v in zip(ids.department, wmean(W[c], wsum(degree_days(b, a, 30))))}))
        # pixel-level exposures for the subsample
        cc = cols_sub[csub[c]]
        aa, bb, pp = a[:, cc], b[:, cc], p[:, cc]
        stc, enc = st[cc], en[cc]
        ws = lambda x: wsum(x, cols=cc)  # noqa: E731
        rec = {}
        for t in DD_B:
            rec[f"DD_{t}"] = ws(degree_days(bb, aa, t))
        for t in TA_B:
            rec[f"TA_{t}"] = ws(time_above(bb, aa, t))
        rec["P_mm"] = ws(pp)
        rec["rain_days"] = ws(np.where(np.isnan(pp), np.nan, (pp >= 1.0) * 1.0))
        n = (enc - stc + 1).astype(float)
        rec["window_days"] = n
        rec["tmax_mean"] = ws(aa) / n
        rec["tmin_mean"] = ws(bb) / n
        dry = pp < 1.0
        r = np.zeros(len(cc)); cdd = np.zeros(len(cc))
        for t in range(nd):
            r = np.where(dry[t] & (t >= stc) & (t <= enc), r + 1, 0)
            cdd = np.maximum(cdd, r)
        rec["cdd_max"] = np.where(np.isfinite(rec["P_mm"]), cdd, np.nan)
        rec["PET_mm"] = PET_[cc]
        rec["D_mm"] = rec["P_mm"] - rec["PET_mm"]
        df = pd.DataFrame(rec)
        df.insert(0, "pix", sub[c].pix.values)
        df.insert(1, "crop", c)
        df.insert(2, "year", y)
        df.insert(3, "dept_name", dname.reindex(dep_pix[cc]).values)
        df.insert(4, "start_day_from_1may", stc.astype(float))
        prow.append(df)
    print("year", y, flush=True)

# SPEI per pixel, then department means
rows, spei_pix = [], {}
for c in CROPS:
    S, nfb = spei_fit(D_all[c])
    spei_pix[c] = S
    print(c, "SPEI: pixels with a value beyond the fitted bound:", nfb, "of", int(np.isfinite(D_all[c]).any(axis=0).sum()), flush=True)
    for k, y in enumerate(YEARS):
        sp, de, pe = wmean(W[c], S[k]), wmean(W[c], D_all[c][k]), wmean(W[c], PET_all[c][k])
        for j in range(len(ids)):
            if W[c][j].sum() == 0:
                continue
            rows.append(dict(department=ids.department[j], crop=c, calendar="dynamic", year=y, spei=sp[j],
                             D_mm=de[j], PET_mm=pe[j]))
dep = pd.DataFrame(rows)
dep.to_csv(OUT + "spei_department_crop.csv", index=False, float_format="%.4f")

px = pd.concat(prow, ignore_index=True)
px["spei"] = np.nan
for c in CROPS:
    for k, y in enumerate(YEARS):
        m_ = (px.crop == c) & (px.year == y)
        px.loc[m_, "spei"] = spei_pix[c][k, cols_sub[csub[c]]]
px.to_csv(OUT + "pixel_exposures_subsample.csv", index=False, float_format="%.4f")

# check
ex = pd.read_csv(AFM + "work/exposures_department_crop.csv").query("calendar == 'dynamic'")
ck = pd.DataFrame(chk)
dif = []
for c in CROPS:
    e = ex[ex.crop == c]
    for _, r in e.iterrows():
        q = ck[(ck.crop == c) & (ck.year == r.year)].iloc[0]
        dif.append((abs(q[f"P_mm_{r.department}"] - r.P_mm), abs(q[f"DD_30_{r.department}"] - r.DD_30)))
dif = np.array(dif)
print("max |diff| vs AFM file: P_mm %.4f, DD_30 %.4f (n=%d)" % (np.nanmax(dif[:, 0]), np.nanmax(dif[:, 1]), len(dif)))
print(dep.groupby("crop")[["spei", "D_mm", "PET_mm"]].describe().round(2).T.to_string())
print(px.shape)
