"""E1 inputs: simulated department yields in the exact format of the AFM yield file, so that the AFM scripts
(07, 22, 24, 25) run unchanged. Same departments, years and weights as DAPSA (valid reported years of the kept
series), same linear (main) and quadratic detrending as 01_detrend_dapsa.py.
usage: python build_e1_inputs.py <variant> <millet_csv|-> <groundnut_csv|-> [years=2001-2016] [drop=1997]
       variant 'obs' writes the observed reference (millet and groundnut only)."""
import os
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import department_yields as E          # noqa: E402
import paths  # noqa: E402

AFM = paths.AFM
E1 = paths.DSSAT + "e1/"
variant, f_ml, f_gn = sys.argv[1:4]
opt = dict(a.split("=", 1) for a in sys.argv[4:])
obs = pd.read_csv(AFM + "outputs/DAPSA_yields_detrended_1986-2016.csv")
obs = obs[obs.crop.isin(["MILLET", "GROUNDNUT"])].copy()


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


if variant.startswith("obs"):
    y = obs
else:
    parts = []
    for crop, f in (("MILLET", f_ml), ("GROUNDNUT", f_gn)):
        res = pd.read_csv(f)
        d = E.dept_table(res, crop="MILLET") if crop == "MILLET" else E.dept_groundnut(res)
        sim = d[["dept", "year", "sim_kg_ha"]].rename(columns={"dept": "department"})
        o = obs[obs.crop == crop].merge(sim, on=["department", "year"], how="left")
        o["yield_kg_ha"] = np.where(o.yield_kg_ha.notna(), o.sim_kg_ha, np.nan)   # same valid department-years
        parts.append(o.drop(columns="sim_kg_ha"))
    y = pd.concat(parts, ignore_index=True)
    y = pd.concat([detrend(g) for _, g in y.groupby(["department", "crop"])], ignore_index=True)
if "years" in opt:
    a, b = map(int, opt["years"].split("-"))
    y = y[y.year.between(a, b)]
if "drop" in opt:
    y = y[~y.year.isin([int(v) for v in opt["drop"].split(",")])]
os.makedirs(E1 + variant + "/outputs", exist_ok=True)
os.makedirs(E1 + variant + "/work", exist_ok=True)
y.to_csv(E1 + variant + "/outputs/DAPSA_yields_detrended_1986-2016.csv", index=False, float_format="%.3f")
link = E1 + variant + "/work/exposures_department_crop.csv"
if not os.path.lexists(link):
    os.symlink(AFM + "work/exposures_department_crop.csv", link)
v = y[y.anom_lin_pct.notna()]
print(f"{variant}: {len(v)} department-years with anomalies "
      f"({v[v.crop == 'MILLET'].department.nunique()} millet, {v[v.crop == 'GROUNDNUT'].department.nunique()} groundnut series); "
      f"min trend {y.trend_lin_kg_ha.min():.0f} kg/ha; anomaly sd {v.groupby('crop').anom_lin_pct.std().round(1).to_dict()}")
