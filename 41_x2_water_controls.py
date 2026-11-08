"""X2 analysis: the AFM chain (07_heat_models.py, main specification, unchanged) with extra water controls, on the
DSSAT department yields of the rainfed runs (heat functions original, neutralised, thresholds -3 degC; 1/9 pixel
subsample) and on DAPSA. Every set keeps the AFM controls (P, P^2, rain days, longest dry spell, window start):
  main : nothing more
  spei : SPEI of the crop window (P - Hargreaves PET, log-logistic) and its square
  wsi  : DSSAT-simulated water stress on photosynthesis before and after anthesis, and their squares, from the run of
         the same treatment (for DAPSA: the run with the original heat functions)
Then the selection-adjusted tests (24_selection_adjusted_tests.py, B = 9999, as for the observed analysis) in each directory.
usage: python x2_analyse.py [source ...]   sources: dapsa orig neutral minus3 -> x2x3/X2_summary.csv"""
import os
import subprocess
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import department_yields as E  # noqa: E402
import paths  # noqa: E402

AFM = paths.AFM
E2 = paths.DSSAT + "e2/"
X2 = paths.DSSAT + "x2x3/"
S24 = paths.SCRIPTS + "04_search_adjusted_tests.py"
CROPS = ["MILLET", "GROUNDNUT"]
BASE = "rain_days,cdd_max,start_day_from_1may"
CTRL = {"main": BASE, "spei": BASE + ",spei,spei_sq",
        "wsi": BASE + ",wsi_pre,wsi_post,wsi_pre_sq,wsi_post_sq"}
obs = pd.read_csv(AFM + "outputs/DAPSA_yields_detrended_1986-2016.csv")
obs = obs[obs.crop.isin(CROPS)]


def detrend(g):
    v = g[g.yield_kg_ha.notna()]
    t, tt = (v.year.values - 1986).astype(float), (g.year.values - 1986).astype(float)
    out = g.copy()
    for deg, lab in ((1, "lin"), (2, "quad")):
        X = np.column_stack([t ** k for k in range(deg + 1)])
        b = np.linalg.lstsq(X, v.yield_kg_ha.values, rcond=None)[0]
        yhat = np.column_stack([tt ** k for k in range(deg + 1)]) @ b
        yhat = np.where((tt >= t.min()) & (tt <= t.max()), yhat, np.nan)
        out[f"trend_{lab}_kg_ha"] = yhat
        out[f"anom_{lab}_kg_ha"] = out.yield_kg_ha - yhat
        out[f"anom_{lab}_pct"] = 100 * (out.yield_kg_ha - yhat) / yhat
    return out


def dept_mean(res, col, crop):
    d = E.dept_table(res.assign(HWAM=res[col]), crop=crop)
    return d[["dept", "year", "sim_kg_ha"]].rename(columns={"dept": "department", "sim_kg_ha": col})


def sim_yields(heat):
    parts = []
    for crop in CROPS:
        res = pd.read_csv(E2 + f"{crop.lower()}_{heat}_rainfed.csv")
        sim = dept_mean(res, "HWAM" if crop == "MILLET" else "PWAM", crop).rename(
            columns={"HWAM": "sim", "PWAM": "sim"})
        o = obs[obs.crop == crop].merge(sim, on=["department", "year"], how="left")
        o["yield_kg_ha"] = np.where(o.yield_kg_ha.notna(), o["sim"], np.nan)
        parts.append(pd.concat([detrend(g) for _, g in o.drop(columns="sim").groupby("department")],
                               ignore_index=True))
    return pd.concat(parts, ignore_index=True)


def wsi_table(heat):
    parts = []
    for crop in CROPS:
        f = E2 + f"wsi_{crop.lower()}_{heat}.csv"
        if not os.path.exists(f):
            return None
        w = pd.read_csv(f)
        a = dept_mean(w, "wspd_pre", crop).merge(dept_mean(w, "wspd_post", crop), on=["department", "year"])
        a = a.rename(columns={"wspd_pre": "wsi_pre", "wspd_post": "wsi_post"})
        a[["wsi_pre", "wsi_post"]] *= 100.0      # in % of full stress: the panel file keeps 4 decimals
        a["crop"] = crop
        parts.append(a)
    t = pd.concat(parts, ignore_index=True)
    t["wsi_pre_sq"], t["wsi_post_sq"] = t.wsi_pre ** 2, t.wsi_post ** 2
    return t


def exposures(wsi):
    ex = pd.read_csv(AFM + "work/exposures_department_crop.csv")
    sp = pd.read_csv(X2 + "spei_department_crop.csv")
    ex = ex.merge(sp, on=["department", "crop", "calendar", "year"], how="left")
    ex["spei_sq"] = ex.spei ** 2
    if wsi is not None:
        ex = ex.merge(wsi, on=["department", "crop", "year"], how="left")
    return ex


def run(source, ctrl):
    heat = "orig" if source == "dapsa" else source
    wsi = wsi_table(heat)
    if ctrl == "wsi" and wsi is None:
        print("no WSI yet for", source, flush=True)
        return None
    base = X2 + f"chain/{source}_{ctrl}/"
    d_out, d_work = base + "outputs", base + "work"
    os.makedirs(d_out, exist_ok=True); os.makedirs(d_work, exist_ok=True)
    y = obs.copy() if source == "dapsa" else sim_yields(heat)
    ex = exposures(wsi)
    if ctrl == "wsi":
        # keep the department-years that have the simulated stress (DAPSA: departments without subsample pixels)
        ok = y.merge(ex[ex.calendar == "dynamic"][["department", "crop", "year", "wsi_pre"]],
                     on=["department", "crop", "year"], how="left").wsi_pre.notna().values
        for c in ("anom_lin_pct", "anom_quad_pct"):
            y.loc[~ok, c] = np.nan
    y.to_csv(d_out + "/DAPSA_yields_detrended_1986-2016.csv", index=False, float_format="%.3f")
    ex.to_csv(d_work + "/exposures_department_crop.csv", index=False, float_format="%.5f")
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    subprocess.run([sys.executable, "-I", paths.SCRIPTS + "02_threshold_panel_model.py", d_out, d_work, "main",
                    f"controls={CTRL[ctrl]}"], check=True, env=env, capture_output=True)
    subprocess.run([sys.executable, "-I", S24, d_work, "9999"], check=True, env=env, capture_output=True)
    r = pd.read_csv(d_work + "/model_results_main.csv")
    a = pd.read_csv(d_work + "/selection_adjusted_tests_main.csv").set_index("crop")
    rows = []
    for _, x in r.iterrows():
        rows.append(dict(source=source, controls=ctrl, crop=x.crop, n_obs=int(x.n_obs), Tstar=int(x.Tstar),
                         Tstar_p05=x.Tstar_boot_p05, Tstar_p95=x.Tstar_boot_p95, b_edd=x.b_edd_pct_per_dd,
                         t_edd=x.t_edd, p_edd_adj=a.loc[x.crop, "p_edd_search"],
                         p_edd_adj_max=a.loc[x.crop, "p_edd_search_max_over_TR"],
                         p_joint_adj=a.loc[x.crop, "p_joint_search"], r2_within=x.r2_within,
                         warm1C=x.warm1C_pct, warm1C_p05=x.warm1C_boot_p05, warm1C_p95=x.warm1C_boot_p95))
    print(pd.DataFrame(rows).round(3).to_string(index=False), flush=True)
    return rows


if __name__ == "__main__":
    sources = [a for a in sys.argv[1:] if "=" not in a] or ["dapsa", "orig", "neutral", "minus3"]
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    ctrls = opt.get("ctrl", "main,spei,wsi").split(",")
    out = X2 + "X2_summary.csv"
    old = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame()
    new = []
    for s in sources:
        for c in ctrls:
            rows = run(s, c)
            if rows:
                new += rows
    if new:
        N = pd.DataFrame(new)
        if len(old):
            old = old[~old.set_index(["source", "controls"]).index.isin(N.set_index(["source", "controls"]).index)]
        T = pd.concat([old, N], ignore_index=True)
        T.to_csv(out, index=False, float_format="%.4f")
        pd.set_option("display.width", 250)
        print(T.round(3).to_string(index=False))
