"""
25_threshold_uncertainty.py - how well the threshold is located.

(1) Distribution of the threshold selected in the 1,000 whole-year bootstrap replicates of
    07_heat_models.py (main and three-year block schemes): share of replicates at each candidate T*,
    within +/-1 and +/-2 degC of the estimate, and at the 26 and 40 degC bounds.
(2) Leave-one-year-out selection: each year is removed in turn, department intercepts and trends are
    re-estimated on the remaining years, and T* is searched again over 26-40 degC (minimum RSS).

Usage: python3 25_threshold_uncertainty.py <work_dir> [tag]
Writes <work_dir>/threshold_bootstrap_distribution_<tag>.csv and <work_dir>/threshold_uncertainty_<tag>.csv
"""
import sys
import numpy as np
import pandas as pd

work = sys.argv[1]
TAG = sys.argv[2] if len(sys.argv) > 2 else "main"
CROPS = ["MILLET", "GROUNDNUT"]          # E1: the two simulated crops only
TSTAR = list(range(26, 41))
YVAR = "anom_lin_pct"
res = pd.read_csv(f"{work}/model_results_{TAG}.csv").set_index("crop")
panel = pd.read_csv(f"{work}/panel_{TAG}.csv")
panel["P100"] = panel.P_mm / 100.0
panel["P100sq"] = panel.P100 ** 2


def detrend_within(df, cols):
    r = pd.DataFrame(index=df.index, columns=cols, dtype=float)
    for _, g in df.groupby("department"):
        t = g.year.values - g.year.mean()
        X = np.column_stack([np.ones(len(g)), t])
        Bc = np.linalg.lstsq(X, g[cols].values.astype(float), rcond=None)[0]
        r.loc[g.index, cols] = g[cols].values - X @ Bc
    return r


def select_T(df, ctrl):
    cols = ["DD_8"] + [f"DD_{b}" for b in TSTAR] + ["P100", "P100sq"] + ctrl + [YVAR]
    Z = detrend_within(df, cols)
    y = Z[YVAR].values.astype(float)
    W = Z[["P100", "P100sq"] + ctrl].values.astype(float)
    rss = {}
    for ts in TSTAR:
        X = np.column_stack([Z.DD_8.values - Z[f"DD_{ts}"].values, Z[f"DD_{ts}"].values, W])
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        rss[ts] = float(((y - X @ b) ** 2).sum())
    return min(rss, key=rss.get)


dist, summ = [], []
for crop in CROPS:
    t0 = int(res.loc[crop, "Tstar"])
    fixed = bool(res.loc[crop, "Tstar_fixed"])
    row = {"crop": crop, "Tstar": t0, "fixed": fixed}
    for scheme, tag_b in (("year", TAG), ("block3", "r_block3")):
        try:
            ts = np.load(f"{work}/bootstrap_{crop}_{tag_b}.npz")["tstar"].astype(int)
        except FileNotFoundError:
            continue
        if fixed:
            continue
        share = pd.Series(ts).value_counts(normalize=True).reindex(TSTAR, fill_value=0.0)
        for T, s in share.items():
            dist.append({"crop": crop, "scheme": scheme, "Tstar_candidate": T, "share": s})
        row.update({f"{scheme}_p05": np.percentile(ts, 5), f"{scheme}_p95": np.percentile(ts, 95),
                    f"{scheme}_within1": float((abs(ts - t0) <= 1).mean()),
                    f"{scheme}_within2": float((abs(ts - t0) <= 2).mean()),
                    f"{scheme}_at26": float((ts == 26).mean()), f"{scheme}_at40": float((ts == 40).mean())})
    if not fixed:
        g = panel[panel.crop == crop]
        ctrl = res.loc[crop, "controls"].split("+")
        assert select_T(g.copy(), ctrl) == t0
        loyo = [select_T(g[g.year != y0].copy(), ctrl) for y0 in sorted(g.year.unique())]
        row.update({"loyo_min": min(loyo), "loyo_max": max(loyo),
                    "loyo_share_at_estimate": float(np.mean(np.array(loyo) == t0))})
    summ.append(row)
    print(row, flush=True)

pd.DataFrame(dist).to_csv(f"{work}/threshold_bootstrap_distribution_{TAG}.csv", index=False, float_format="%.4f")
pd.DataFrame(summ).to_csv(f"{work}/threshold_uncertainty_{TAG}.csv", index=False, float_format="%.4f")
