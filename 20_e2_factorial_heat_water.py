"""E2 analysis: the AFM chain (07_heat_models.py, main specification, unchanged) on the department yields of every
factorial treatment, then an exact 2 x 2 decomposition of the statistical +1 degC effect:
    full (orig heat, rainfed) = neither + A + B + AB
    neither = heat functions neutralised, no water stress (the other temperature effects: development rate,
              respiration, ...)
    A  = heat functions, without water stress        = E(orig, no water stress) - E(neutral, no water stress)
    B  = water stress, without heat functions         = E(neutral, rainfed)       - E(neutral, no water stress)
    AB = interaction                                   = full - E(neutral, rainfed) - E(orig, no water stress) + neither
usage: python e2_analyse.py"""
import glob
import os
import subprocess
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import department_yields as E          # noqa: E402
import paths  # noqa: E402

AFM = paths.AFM
E2 = paths.DSSAT + "e2/"
CH = E2 + "chain/"
obs = pd.read_csv(AFM + "outputs/DAPSA_yields_detrended_1986-2016.csv")


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


def chain(crop, f):
    name = os.path.basename(f)[:-4]
    d_out, d_work = CH + name + "/outputs", CH + name + "/work"
    res_file = d_work + "/model_results_main.csv"
    if os.path.exists(res_file) and os.path.getmtime(res_file) > os.path.getmtime(os.path.realpath(f)):
        return pd.read_csv(res_file).iloc[0], name
    os.makedirs(d_out, exist_ok=True); os.makedirs(d_work, exist_ok=True)
    res = pd.read_csv(f)
    d = E.dept_table(res, crop="MILLET") if crop == "MILLET" else E.dept_groundnut(res)
    sim = d[["dept", "year", "sim_kg_ha"]].rename(columns={"dept": "department"})
    o = obs[obs.crop == crop].merge(sim, on=["department", "year"], how="left")
    o["yield_kg_ha"] = np.where(o.yield_kg_ha.notna(), o.sim_kg_ha, np.nan)
    o = pd.concat([detrend(g) for _, g in o.drop(columns="sim_kg_ha").groupby("department")], ignore_index=True)
    o.to_csv(d_out + "/DAPSA_yields_detrended_1986-2016.csv", index=False, float_format="%.3f")
    link = d_work + "/exposures_department_crop.csv"
    if not os.path.lexists(link):
        os.symlink(AFM + "work/exposures_department_crop.csv", link)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    subprocess.run([sys.executable, "-I", paths.SCRIPTS + "02_threshold_panel_model.py", d_out, d_work, "main",
                    "controls=rain_days,cdd_max,start_day_from_1may", "fixT=MAIZE:32"], check=True, env=env,
                   capture_output=True)
    return pd.read_csv(res_file).iloc[0], name


rows = []
for f in sorted(glob.glob(E2 + "*.csv")):
    base = os.path.basename(f)
    if not (base.startswith("millet_") or base.startswith("groundnut_")):
        continue
    crop = "MILLET" if base.startswith("millet_") else "GROUNDNUT"
    r, name = chain(crop, f)
    parts = name.split("_", 1)[1].split("_")
    rows.append(dict(crop=crop, treatment=name.split("_", 1)[1], heat=parts[0], water=parts[1],
                     variant="_".join(parts[2:]) or "base", Tstar=int(r.Tstar), Tstar_p05=r.Tstar_boot_p05,
                     Tstar_p95=r.Tstar_boot_p95, b_edd=r.b_edd_pct_per_dd, t_edd=r.t_edd, b_gdd=r.b_gdd_pct_per_dd,
                     p_joint_naive=r.p_heat_joint_F, r2_within=r.r2_within, r2_water_only=r.r2_within_water_only,
                     warm1C=r.warm1C_pct, warm1C_p05=r.warm1C_boot_p05, warm1C_p95=r.warm1C_boot_p95))
T = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(T.round(3).to_string(index=False))
T.to_csv(E2 + "E2_treatments.csv", index=False, float_format="%.4f")

dec = []
for (crop, variant), g in T.groupby(["crop", "variant"]):
    e = g.set_index(["heat", "water"]).warm1C
    need = [("orig", "rainfed"), ("neutral", "rainfed"), ("orig", "nowaterstress"), ("neutral", "nowaterstress")]
    if not all(k in e.index for k in need):
        continue
    full, nh, nw, nn = (e[k] for k in need)
    dec.append(dict(crop=crop, variant=variant, full=full, other_temperature_effects=nn, heat_functions=nw - nn,
                    water_stress=nh - nn, interaction=full - nh - nw + nn))
D = pd.DataFrame(dec)
if len(D):
    print("\n+1 degC effect (%) and its decomposition:")
    print(D.round(2).to_string(index=False))
    D.to_csv(E2 + "E2_decomposition.csv", index=False, float_format="%.4f")
