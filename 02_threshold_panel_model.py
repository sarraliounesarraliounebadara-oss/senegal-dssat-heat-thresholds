"""
07_heat_models.py
Piecewise-linear degree-day yield models (Schlenker & Roberts, 2009), one per crop,
estimated on DAPSA department yields 1986-2016.

  y_it = b1 GDD_it(8, T*) + b2 EDD_it(T*) + g1 P_it + g2 P_it^2 + [water controls] + a_i + e_it

  y      : yield anomaly (% of the department linear trend)
  GDD    : degree days between 8 degC and T* over the crop window
  EDD    : degree days above T* (extreme degree days)
  P      : window precipitation (100 mm)
  water controls (optional): rain_days, cdd_max (longest dry spell), start_day (sowing date)
  a_i    : department fixed effect
All variables are linearly detrended within each department on the same years as the
yields, so estimates use year-to-year variation only (Frisch-Waugh-Lovell).
T* is selected by minimum residual sum of squares over 26..40 degC; confidence intervals
come from a block bootstrap over years (all departments of a year resampled together;
T* re-selected in each replicate). Standard errors are clustered by year.

Usage: python3 07_heat_models.py <outputs_dir> <work_dir> <tag> [key=value ...]
  calendar=dynamic|sivakumar|static   yvar=anom_lin_pct|anom_quad_pct
  controls=rain_days,cdd_max,start_day_from_1may   nogdd=1   drop=y2004,maize0305
  fixT=MAIZE:32   (threshold fixed for crops whose profile is flat, i.e. T* not identified;
                   the profile is still computed over 26-40 degC for display)
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats

out, work, TAG = sys.argv[1], sys.argv[2], sys.argv[3]
opt = dict(a.split("=", 1) for a in sys.argv[4:])
CAL = opt.get("calendar", "dynamic")
YVAR = opt.get("yvar", "anom_lin_pct")
CTRL = [c for c in opt.get("controls", "").split(",") if c]
NOGDD = opt.get("nogdd", "0") == "1"
DROP = [c for c in opt.get("drop", "").split(",") if c]
CROPS = ["MILLET", "GROUNDNUT", "SORGHUM", "MAIZE"]
TSTAR = list(range(26, 41))
NBOOT = int(opt.get("nboot", 1000))
BOOT_BLOCK = int(opt.get("boot_block", 1))
FIXT = {k: int(v) for k, v in (x.split(":") for x in opt.get("fixT", "").split(",") if x)}
rng = np.random.default_rng(2026)

y = pd.read_csv(f"{out}/DAPSA_yields_detrended_1986-2016.csv")
ex = pd.read_csv(f"{work}/exposures_department_crop.csv")
ex = ex[ex.calendar == CAL]
d = y.merge(ex, on=["department", "crop", "year"], how="left")
d = d[d[YVAR].notna()].copy()
if "y2004" in DROP:
    d = d[d.year != 2004]
if "maize0305" in DROP:
    d = d[~((d.crop == "MAIZE") & d.year.between(2003, 2005))]
assert d.DD_8.notna().all(), "missing exposures in the yield period"
d["P100"] = d.P_mm / 100.0
d["P100sq"] = d.P100 ** 2


def detrend_within(df, cols):
    """Residuals of each column on [1, year] within each department."""
    res = pd.DataFrame(index=df.index, columns=cols, dtype=float)
    for _, g in df.groupby("department"):
        t = g.year.values - g.year.mean()
        X = np.column_stack([np.ones(len(g)), t])
        B = np.linalg.lstsq(X, g[cols].values.astype(float), rcond=None)[0]
        res.loc[g.index, cols] = g[cols].values - X @ B
    return res


def ols(X, yv):
    XtX = X.T @ X
    try:
        b = np.linalg.solve(XtX, X.T @ yv)
    except np.linalg.LinAlgError:
        b = np.linalg.lstsq(X, yv, rcond=None)[0]
    return b, yv - X @ b, XtX


def cluster_cov(X, u, XtX, groups, n_fe):
    inv = np.linalg.inv(XtX)
    meat = np.zeros_like(XtX)
    for g in np.unique(groups):
        s = groups == g
        v = X[s].T @ u[s]
        meat += np.outer(v, v)
    # Intercept AND time slope were absorbed separately for each department.
    G, N, K = len(np.unique(groups)), len(u), X.shape[1] + 2 * n_fe
    return inv @ meat @ inv * G / (G - 1) * (N - 1) / (N - K)


rows, prof, binrows = [], [], []
for crop in CROPS:
    g = d[d.crop == crop].copy()
    if g.empty:
        continue
    dd_cols = [f"DD_{b}" for b in range(7, 43)]
    ta_cols = [f"TA_{b}" for b in range(20, 47)]
    Z = detrend_within(g, dd_cols + ta_cols + ["P100", "P100sq"] + CTRL + [YVAR])
    yv = Z[YVAR].values.astype(float)
    years = g.year.values
    n_fe = g.department.nunique()
    water = [Z.P100.values, Z.P100sq.values] + [Z[c].values for c in CTRL]

    def design(ts):
        gdd = Z["DD_8"].values - Z[f"DD_{ts}"].values
        edd = Z[f"DD_{ts}"].values
        cols = ([] if NOGDD else [gdd]) + [edd] + water
        return np.column_stack(cols).astype(float)

    iE = 0 if NOGDD else 1          # index of EDD coefficient

    rss = {}
    for ts in TSTAR:
        b, u, _ = ols(design(ts), yv)
        rss[ts] = float(u @ u)
        prof.append({"crop": crop, "Tstar": ts, "rss": rss[ts],
                     "r2_within": 1 - rss[ts] / float(yv @ yv)})
    ts_best = FIXT.get(crop, min(rss, key=rss.get))
    ts_grid = [ts_best] if crop in FIXT else TSTAR
    X = design(ts_best)
    b, u, XtX = ols(X, yv)
    V = cluster_cov(X, u, XtX, years, n_fe)
    se = np.sqrt(np.diag(V))
    Xw = np.column_stack(water).astype(float)
    bw, uw, _ = ols(Xw, yv)
    r2_w = 1 - float(uw @ uw) / float(yv @ yv)

    def warming_effect(bvec, ts):
        de = g[f"DD_{ts - 1}"].values - g[f"DD_{ts}"].values
        eff = bvec[iE] * de
        if not NOGDD:
            dg = (g.DD_7.values - g[f"DD_{ts - 1}"].values) - (g.DD_8.values - g[f"DD_{ts}"].values)
            eff = eff + bvec[0] * dg
        return eff

    w1 = warming_effect(b, ts_best)
    uy = np.unique(years)
    idx_by_year = {yy: np.where(years == yy)[0] for yy in uy}
    # Re-estimate absorbed department intercepts and time slopes in every draw.
    # Multiplicity weights are exactly equivalent to duplicating sampled years.
    boot_cols = dd_cols + ["P100", "P100sq"] + CTRL + [YVAR]
    values = Z[boot_cols].to_numpy(dtype=float)
    positions = {v: k for k, v in enumerate(boot_cols)}
    group_indices = [np.flatnonzero(g.department.to_numpy() == dep)
                     for dep in g.department.unique()]
    times = years.astype(float) - 2001.0
    year_codes = np.searchsorted(uy, years)
    water_positions = [positions[v] for v in ["P100", "P100sq"] + CTRL]
    draws = []
    bt, bb_edd, bb_gdd, bw1 = [], [], [], []
    for _ in range(NBOOT):
        if BOOT_BLOCK == 1:
            draw = rng.choice(uy, size=len(uy), replace=True)
        else:
            starts = rng.integers(0, len(uy), size=int(np.ceil(len(uy) / BOOT_BLOCK)))
            draw_indices = ((starts[:, None] + np.arange(BOOT_BLOCK)) % len(uy)).ravel()[:len(uy)]
            draw = uy[draw_indices]
        draws.append(draw)
        weights = np.bincount(np.searchsorted(uy, draw), minlength=len(uy))[year_codes].astype(float)
        root_weights = np.sqrt(weights)
        zb = values.copy()
        for indices in group_indices:
            w = weights[indices]
            total = w.sum()
            if total == 0:
                zb[indices] = 0.0
                continue
            t = times[indices]
            tc = t - np.dot(w, t) / total
            mean = w @ values[indices] / total
            denominator = np.dot(w * tc, tc)
            slope = ((w * tc) @ values[indices] / denominator
                     if denominator > 0 else np.zeros(values.shape[1]))
            zb[indices] = values[indices] - mean - tc[:, None] * slope
        boot_y = zb[:, positions[YVAR]] * root_weights
        boot_water = zb[:, water_positions] * root_weights[:, None]
        best, bestrss, bestb = None, np.inf, None
        for ts in ts_grid:
            edd = zb[:, positions[f"DD_{ts}"]] * root_weights
            gdd = (zb[:, positions["DD_8"]] - zb[:, positions[f"DD_{ts}"]]) * root_weights
            boot_x = np.column_stack(([] if NOGDD else [gdd]) + [edd, boot_water])
            bb_, ub, _ = ols(boot_x, boot_y)
            r_ = float(ub @ ub)
            if r_ < bestrss:
                best, bestrss, bestb = ts, r_, bb_
        bt.append(best); bb_edd.append(bestb[iE]); bb_gdd.append(0.0 if NOGDD else bestb[0])
        bw1.append(np.mean(warming_effect(bestb, best)))
    bt, bb_edd, bb_gdd, bw1 = map(np.array, (bt, bb_edd, bb_gdd, bw1))
    np.savez_compressed(f"{work}/bootstrap_{crop}_{TAG}.npz", tstar=bt, b_edd=bb_edd,
                        b_gdd=bb_gdd, warm1=bw1, draw_years=np.asarray(draws, dtype=np.int16),
                        boot_block=BOOT_BLOCK, nuisance_refitted=True)

    if NOGDD:
        wald = float(b[iE] ** 2 / V[iE, iE])
    else:
        R = np.zeros((2, X.shape[1])); R[0, 0] = R[1, 1] = 1
        wald = float((R @ b) @ np.linalg.solve(R @ V @ R.T, R @ b))
    names = ([] if NOGDD else ["gdd"]) + ["edd", "P", "P2"] + CTRL
    row = {"crop": crop, "calendar": CAL, "yvar": YVAR, "tag": TAG,
           "controls": "+".join(CTRL), "nogdd": NOGDD, "n_obs": len(yv),
           "n_departments": n_fe, "n_years": len(uy), "Tstar": ts_best,
           "Tstar_fixed": crop in FIXT,
           "Tstar_boot_p05": np.percentile(bt, 5), "Tstar_boot_p50": np.percentile(bt, 50),
           "Tstar_boot_p95": np.percentile(bt, 95),
           "b_gdd_pct_per_dd": np.nan if NOGDD else b[0], "se_gdd": np.nan if NOGDD else se[0],
           "b_edd_pct_per_dd": b[iE], "se_edd": se[iE], "t_edd": b[iE] / se[iE],
           "p_edd_cluster_t": 2 * stats.t.sf(abs(b[iE] / se[iE]), len(uy) - 1),
           "p_heat_joint_F": stats.f.sf(wald / (1 if NOGDD else 2), 1 if NOGDD else 2, len(uy) - 1),
           "bootstrap_block_years": BOOT_BLOCK,
           "b_edd_boot_p05": np.percentile(bb_edd, 5), "b_edd_boot_p95": np.percentile(bb_edd, 95),
           "r2_within": 1 - float(u @ u) / float(yv @ yv), "r2_within_water_only": r2_w,
           "wald_heat": wald, "mean_EDD": g[f"DD_{ts_best}"].mean(),
           "sd_EDD_detrended": Z[f"DD_{ts_best}"].std(),
           "warm1C_pct": w1.mean(), "warm1C_boot_p05": np.percentile(bw1, 5),
           "warm1C_boot_p95": np.percentile(bw1, 95)}
    for nm, bv, sv in zip(names, b, se):
        if nm not in ("gdd", "edd"):
            row[f"b_{nm}"], row[f"se_{nm}"] = bv, sv
    rows.append(row)

    # flexible binned model: days in 3-degC bins (reference bin 24-27 degC)
    edges = [24, 27, 30, 33, 36, 39]
    tb = {f"{edges[i]}-{edges[i + 1]}": Z[f"TA_{edges[i]}"].values - Z[f"TA_{edges[i + 1]}"].values
          for i in range(len(edges) - 1)}
    tb[">=39"] = Z["TA_39"].values
    bnames = [n for n in tb if n != "24-27"]
    Xb = np.column_stack([tb[n] for n in bnames] + water).astype(float)
    bb, ub, XtXb = ols(Xb, yv)
    seb = np.sqrt(np.diag(cluster_cov(Xb, ub, XtXb, years, n_fe)))
    raw_days = {f"{edges[i]}-{edges[i + 1]}": (g[f"TA_{edges[i]}"] - g[f"TA_{edges[i + 1]}"]).mean()
                for i in range(len(edges) - 1)}
    raw_days[">=39"] = g["TA_39"].mean()
    raw_days["<24"] = (g.window_days - g["TA_24"]).mean()
    for n_, coef, s_ in zip(bnames, bb[:len(bnames)], seb[:len(bnames)]):
        binrows.append({"crop": crop, "bin": n_, "coef_pct_per_day": coef, "se": s_,
                        "mean_days": raw_days[n_]})
    binrows.append({"crop": crop, "bin": "24-27", "coef_pct_per_day": 0.0, "se": 0.0,
                    "mean_days": raw_days["24-27"]})
    binrows.append({"crop": crop, "bin": "<24", "coef_pct_per_day": np.nan, "se": np.nan,
                    "mean_days": raw_days["<24"]})
    gtxt = "" if NOGDD else f" b_GDD={b[0]:.3f} (t={b[0] / se[0]:.2f}),"
    print(f"[{TAG}] {crop}: T*={ts_best} ({np.percentile(bt, 5):.0f}-{np.percentile(bt, 95):.0f}),"
          f" b_EDD={b[iE]:.3f}%/dd (t={b[iE] / se[iE]:.2f}),{gtxt}"
          f" R2w={row['r2_within']:.3f} (water only {r2_w:.3f}), +1C={w1.mean():.1f}%"
          f" [{np.percentile(bw1, 5):.1f}, {np.percentile(bw1, 95):.1f}]")

pd.DataFrame(rows).to_csv(f"{work}/model_results_{TAG}.csv", index=False, float_format="%.5f")
pd.DataFrame(prof).to_csv(f"{work}/threshold_profiles_{TAG}.csv", index=False, float_format="%.5f")
pd.DataFrame(binrows).to_csv(f"{work}/binned_models_{TAG}.csv", index=False, float_format="%.5f")
d.to_csv(f"{work}/panel_{TAG}.csv", index=False, float_format="%.4f")
