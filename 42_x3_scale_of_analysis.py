"""X3: thresholds found at pixel scale against department scale, in the DSSAT world (1/9 pixel subsample).
Pixel chain: every pixel is a 'department' of the AFM chain (07 unchanged, main specification, 500 bootstrap
draws; selection-adjusted tests with B = 499; department scale: 1000 draws and B = 9999): its DSSAT yield, detrended linearly over the years when its department
has a DAPSA yield (anomaly in % of the pixel trend), with its own exposures over the department window
(x23_exposures.py). Pixels whose trend falls below 25 % of their mean yield are left out.
Mixed chain: the same pixel yields with the exposures of their department (AFM file), to see whether aggregating
yields or aggregating exposures moves the threshold.
Department chain: the same yields aggregated by department, AFM exposures (as E2, with the tests).
usage: python x3_analyse.py [treatment ...]   (orig_rainfed neutral_rainfed minus3_rainfed orig_nowaterstress)"""
import os
import subprocess
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import department_yields as E  # noqa: E402
import paths  # noqa: E402
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("x2_water_controls", paths.SCRIPTS + "41_x2_water_controls.py")
X2A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(X2A)

AFM = paths.AFM
E2 = paths.DSSAT + "e2/"
X3 = paths.DSSAT + "x2x3/"
S24 = paths.SCRIPTS + "04_search_adjusted_tests.py"
CROPS = ["MILLET", "GROUNDNUT"]
BASE = "rain_days,cdd_max,start_day_from_1may"
px = pd.read_csv(X3 + "pixel_exposures_subsample.csv")
valid = E._obs[["dept", "crop", "year"]].rename(columns={"dept": "dept_name"})
exA = pd.read_csv(AFM + "work/exposures_department_crop.csv").query("calendar == 'dynamic'")


def pixel_yields(treatment):
    parts, dropped = [], {}
    for crop in CROPS:
        res = pd.read_csv(E2 + f"{crop.lower()}_{treatment}.csv")
        res["yield_kg_ha"] = res["HWAM" if crop == "MILLET" else "PWAM"]
        d = res.merge(px[px.crop == crop][["pix", "year", "dept_name"]], on=["pix", "year"])
        d = d.merge(valid[valid.crop == crop], on=["dept_name", "year"])
        rows, nd = [], 0
        for k, g in d.groupby("pix"):
            t = (g.year - g.year.mean()).values.astype(float)
            b = np.polyfit(t, g.yield_kg_ha.values, 1)
            tr = np.polyval(b, t)
            if tr.min() < 0.25 * g.yield_kg_ha.mean():
                nd += 1
                continue
            rows.append(pd.DataFrame({"department": f"P{k}", "crop": crop, "year": g.year.values,
                                      "yield_kg_ha": g.yield_kg_ha.values, "trend_lin_kg_ha": tr,
                                      "anom_lin_pct": 100 * (g.yield_kg_ha.values - tr) / tr,
                                      "anom_quad_pct": np.nan, "dept_name": g.dept_name.values}))
        dropped[crop] = (nd, d.pix.nunique())
        parts += rows
    return pd.concat(parts, ignore_index=True), dropped


def dept_yields(treatment):
    obs = pd.read_csv(AFM + "outputs/DAPSA_yields_detrended_1986-2016.csv")
    parts = []
    for crop in CROPS:
        res = pd.read_csv(E2 + f"{crop.lower()}_{treatment}.csv")
        d = E.dept_table(res.assign(HWAM=res["HWAM" if crop == "MILLET" else "PWAM"]), crop=crop)
        sim = d[["dept", "year", "sim_kg_ha"]].rename(columns={"dept": "department", "sim_kg_ha": "sim"})
        o = obs[obs.crop == crop].merge(sim, on=["department", "year"], how="left")
        o["yield_kg_ha"] = np.where(o.yield_kg_ha.notna(), o["sim"], np.nan)
        parts.append(pd.concat([X2A.detrend(g) for _, g in o.drop(columns="sim").groupby("department")],
                               ignore_index=True))
    y = pd.concat(parts, ignore_index=True)
    y["dept_name"] = y.department
    return y, {c: (0, 0) for c in CROPS}


def run(treatment, scale):
    base = X3 + f"chain_pixel/{treatment}_{scale}/"
    d_out, d_work = base + "outputs", base + "work"
    os.makedirs(d_out, exist_ok=True); os.makedirs(d_work, exist_ok=True)
    y, dropped = dept_yields(treatment) if scale == "department" else pixel_yields(treatment)
    if scale == "department":
        ex = exA.assign(dept_name=exA.department)
    elif scale == "pixel":
        ex = px.assign(department="P" + px.pix.astype(str), calendar="dynamic").drop(columns=["pix"])
    else:
        key = y[["department", "crop", "dept_name"]].drop_duplicates()
        ex = key.merge(exA.rename(columns={"department": "dept_name"}), on=["dept_name", "crop"])
    y.drop(columns="dept_name").to_csv(d_out + "/DAPSA_yields_detrended_1986-2016.csv", index=False,
                                       float_format="%.3f")
    ex.drop(columns=["dept_name"]).to_csv(d_work + "/exposures_department_crop.csv", index=False,
                                          float_format="%.4f")
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    subprocess.run([sys.executable, "-I", paths.SCRIPTS + "02_threshold_panel_model.py", d_out, d_work, "main",
                    f"controls={BASE}"] + ([] if scale == "department" else ["nboot=500"]), check=True, env=env,
                   capture_output=True)
    subprocess.run([sys.executable, "-I", S24, d_work, "9999" if scale == "department" else "499"], check=True, env=env,
                   capture_output=True)
    r = pd.read_csv(d_work + "/model_results_main.csv")
    a = pd.read_csv(d_work + "/selection_adjusted_tests_main.csv").set_index("crop")
    rows = []
    for _, x in r.iterrows():
        rows.append(dict(treatment=treatment, scale=scale, crop=x.crop, n_units=int(x.n_departments),
                         units_dropped=dropped[x.crop][0], n_obs=int(x.n_obs), Tstar=int(x.Tstar),
                         Tstar_p05=x.Tstar_boot_p05, Tstar_p95=x.Tstar_boot_p95, b_edd=x.b_edd_pct_per_dd,
                         t_edd=x.t_edd, p_edd_adj=a.loc[x.crop, "p_edd_search"],
                         p_joint_adj=a.loc[x.crop, "p_joint_search"], r2_within=x.r2_within,
                         warm1C=x.warm1C_pct, warm1C_p05=x.warm1C_boot_p05, warm1C_p95=x.warm1C_boot_p95))
    print(pd.DataFrame(rows).round(3).to_string(index=False), flush=True)
    return rows


if __name__ == "__main__":
    trs = sys.argv[1:] or ["orig_rainfed", "neutral_rainfed", "minus3_rainfed", "orig_nowaterstress"]
    out = X3 + "X3_summary.csv"
    old = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame()
    new = []
    for t in trs:
        for s in ("department", "pixel", "mixed"):
            new += run(t, s)
    N = pd.DataFrame(new)
    if len(old):
        old = old[~old.set_index(["treatment", "scale"]).index.isin(N.set_index(["treatment", "scale"]).index)]
    T = pd.concat([old, N], ignore_index=True)
    T.to_csv(out, index=False, float_format="%.4f")
    pd.set_option("display.width", 250)
    print(T.round(3).to_string(index=False))
