"""
22_threshold_coefficients.py - Table S4: EDD coefficient across candidate thresholds.

Refits the main specification (department intercepts and trends absorbed, GDD + EDD + water-and-timing
controls, year-clustered standard errors with the same small-sample correction as 07_heat_models.py)
at every candidate threshold T* = 26-40 degC and reports the EDD coefficient of each crop.
Point estimates and standard errors at the selected thresholds reproduce model_results_<tag>.csv.

Usage: python3 22_threshold_coefficients.py <work_dir> <tables_dir> [tag]
Writes <work_dir>/edd_by_threshold_<tag>.csv and <tables_dir>/TableS4_edd_by_threshold.{csv,md}
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats

work, tdir = sys.argv[1:3]
TAG = sys.argv[3] if len(sys.argv) > 3 else "main"
CROPS = ["MILLET", "GROUNDNUT"]          # E1: the two simulated crops only
LAB = {"MILLET": "Millet", "GROUNDNUT": "Groundnut", "SORGHUM": "Sorghum", "MAIZE": "Maize"}
TSTAR = list(range(26, 41))
YVAR = "anom_lin_pct"

res = pd.read_csv(f"{work}/model_results_{TAG}.csv").set_index("crop")
panel = pd.read_csv(f"{work}/panel_{TAG}.csv")


def detrend_within(df, cols):
    r = pd.DataFrame(index=df.index, columns=cols, dtype=float)
    for _, g in df.groupby("department"):
        t = g.year.values - g.year.mean()
        X = np.column_stack([np.ones(len(g)), t])
        B = np.linalg.lstsq(X, g[cols].values.astype(float), rcond=None)[0]
        r.loc[g.index, cols] = g[cols].values - X @ B
    return r


def fit_cluster(X, yv, groups, n_fe):
    XtX = X.T @ X
    b = np.linalg.solve(XtX, X.T @ yv)
    u = yv - X @ b
    inv = np.linalg.inv(XtX)
    meat = np.zeros_like(XtX)
    for gg in np.unique(groups):
        s = groups == gg
        v = X[s].T @ u[s]
        meat += np.outer(v, v)
    G, N, K = len(np.unique(groups)), len(u), X.shape[1] + 2 * n_fe     # intercept and slope absorbed
    V = inv @ meat @ inv * G / (G - 1) * (N - 1) / (N - K)
    return b, np.sqrt(np.diag(V))


rows = []
for crop in CROPS:
    g = panel[panel.crop == crop].copy()
    ctrl = res.loc[crop, "controls"].split("+")
    g["P100"] = g.P_mm / 100.0
    g["P100sq"] = g.P100 ** 2
    cols = [f"DD_{b}" for b in range(7, 43)] + ["P100", "P100sq"] + ctrl + [YVAR]
    Z = detrend_within(g, cols)
    yv = Z[YVAR].values.astype(float)
    water = [Z.P100.values, Z.P100sq.values] + [Z[c].values for c in ctrl]
    n_fe = g.department.nunique()
    df_t = g.year.nunique() - 1
    for ts in TSTAR:
        X = np.column_stack([Z.DD_8.values - Z[f"DD_{ts}"].values, Z[f"DD_{ts}"].values] + water).astype(float)
        b, se = fit_cluster(X, yv, g.year.values, n_fe)
        rows.append({"crop": crop, "Tstar": ts, "b_edd": b[1], "se_edd": se[1], "t": b[1] / se[1],
                     "p": 2 * stats.t.sf(abs(b[1] / se[1]), df_t),
                     "selected": ts == int(res.loc[crop, "Tstar"]), "fixed": bool(res.loc[crop, "Tstar_fixed"])})
sweep = pd.DataFrame(rows)
sweep.to_csv(f"{work}/edd_by_threshold_{TAG}.csv", index=False, float_format="%.5f")

# consistency with the main model
for crop in CROPS:
    s = sweep[(sweep.crop == crop) & sweep.selected].iloc[0]
    assert abs(s.b_edd - res.loc[crop, "b_edd_pct_per_dd"]) < 1e-3, crop
    assert abs(s.se_edd - res.loc[crop, "se_edd"]) < 1e-3, crop


def fmt(x, nd=2):
    s = f"{x:.{nd}f}"
    return s.replace("-", "−") if x < 0 and float(s) != 0 else s.replace("-", "")


def star(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


tab = []
for ts in TSTAR:
    row = {"Threshold T* (°C)": ts}
    for crop in CROPS:
        s = sweep[(sweep.crop == crop) & (sweep.Tstar == ts)].iloc[0]
        row[LAB[crop]] = f"{fmt(s.b_edd)}{star(s.p)} ({fmt(s.se_edd)})"
    tab.append(row)
tab = pd.DataFrame(tab)
tab.to_csv(f"{tdir}/TableS4_edd_by_threshold.csv", index=False)
cols = list(tab.columns)
with open(f"{tdir}/TableS4_edd_by_threshold.md", "w") as fh:
    fh.write("| " + " | ".join(cols) + " |\n| " + " | ".join("---" for _ in cols) + " |\n")
    for _, r in tab.iterrows():
        fh.write("| " + " | ".join(str(r[c]) for c in cols) + " |\n")
print(tab.to_string(index=False))
