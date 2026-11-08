"""X1 analysis: true effect of a uniform warming in DSSAT against the effect implied by the AFM chain (E2).
True effect = mean, over the department-years used by the chain, of 100 x (Y_warm - Y_base) / trend_base, where
trend_base is the linear trend of the base department series (the denominator of the chain's anomalies): the same
unit and weighting as the chain's +1 degC effect (07: warm1C_pct). Year-bootstrap 5-95 % of that mean (1000 draws).
Also the ratio of totals 100 x (sum Y_warm / sum Y_base - 1).
Exact 2 x 2 decomposition of the true +1 degC effect (heat functions x water stress), as E2 does for the chain.
usage: python x1_analyse.py -> x2x3/X1_true_vs_chain.csv, x2x3/X1_decomposition.csv"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import department_yields as E  # noqa: E402
import paths  # noqa: E402

E2 = paths.DSSAT + "e2/"
X2 = paths.DSSAT + "x2x3/"
rng = np.random.default_rng(1)
chainT = pd.read_csv(E2 + "E2_treatments.csv").query("variant == 'base'").set_index(["crop", "heat", "water"])


def chain_warming(name, dt):
    """Effect of +dt degC implied by the chain fitted on the base yields (same formula as 07 for dt = 1);
    DD_6 = DD_7 + window days (daily Tmin stays above 7 degC in the crop windows)."""
    w = E2 + f"chain/{name}/work/"
    r = pd.read_csv(w + "model_results_main.csv").iloc[0]
    p = pd.read_csv(w + "panel_main.csv")
    p = p[p.anom_lin_pct.notna()]
    assert (p.tmin_mean > 12).all()
    p["DD_6"] = p.DD_7 + p.window_days
    ts = int(r.Tstar)
    de = p[f"DD_{ts - dt}"] - p[f"DD_{ts}"]
    dg = (p[f"DD_{8 - dt}"] - p[f"DD_{ts - dt}"]) - (p.DD_8 - p[f"DD_{ts}"])
    return float((r.b_edd_pct_per_dd * de + r.b_gdd_pct_per_dd * dg).mean()), float(r.warm1C_pct)


def dept(f, crop):
    res = pd.read_csv(f)
    d = E.dept_table(res.assign(HWAM=res["HWAM" if crop == "MILLET" else "PWAM"]), crop=crop)
    return d[["dept", "year", "sim_kg_ha"]].rename(columns={"dept": "department", "sim_kg_ha": "y_warm"})


rows = []
for crop in ("MILLET", "GROUNDNUT"):
    for heat in ("orig", "neutral", "minus3"):
        for water in ("rainfed", "nowaterstress"):
            name = f"{crop.lower()}_{heat}_{water}"
            base = pd.read_csv(E2 + f"chain/{name}/outputs/DAPSA_yields_detrended_1986-2016.csv")
            base = base[base.anom_lin_pct.notna()]
            for dt in (1, 2):
                f = E2 + f"{name}_p{dt}.csv"
                if not os.path.exists(f):
                    continue
                m = base.merge(dept(f, crop), on=["department", "year"], how="left")
                assert m.y_warm.notna().all()
                m["true_pct"] = 100 * (m.y_warm - m.yield_kg_ha) / m.trend_lin_kg_ha
                yrs = np.sort(m.year.unique())
                by = m.groupby("year").true_pct.agg(["sum", "count"]).reindex(yrs)
                bs = []
                for _ in range(1000):
                    dr = rng.choice(yrs, len(yrs))
                    s = by.loc[dr]
                    bs.append(s["sum"].sum() / s["count"].sum())
                c = chainT.loc[(crop, heat, water)]
                pred, w1 = chain_warming(name, dt)
                if dt == 1:
                    assert abs(pred - w1) < 1e-3, (pred, w1)
                rows.append(dict(crop=crop, heat=heat, water=water, dT=dt, n_obs=len(m), chain_pred_dT=pred,
                                 true_pct=m.true_pct.mean(), true_p05=np.percentile(bs, 5),
                                 true_p95=np.percentile(bs, 95),
                                 true_ratio_pct=100 * (m.y_warm.sum() / m.yield_kg_ha.sum() - 1),
                                 chain_Tstar=int(c.Tstar), chain_warm1C=c.warm1C, chain_p05=c.warm1C_p05,
                                 chain_p95=c.warm1C_p95))
T = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(T.round(2).to_string(index=False))
T.to_csv(X2 + "X1_true_vs_chain.csv", index=False, float_format="%.4f")

dec = []
for crop in ("MILLET", "GROUNDNUT"):
    for heat in ("orig", "minus3"):
        for kind, col in (("true", "true_pct"), ("chain", "chain_warm1C")):
            g = T[(T.crop == crop) & (T.dT == 1)].set_index(["heat", "water"])[col]
            need = [(heat, "rainfed"), ("neutral", "rainfed"), (heat, "nowaterstress"), ("neutral", "nowaterstress")]
            if not all(k in g.index for k in need):
                continue
            full, nh, nw, nn = (g[k] for k in need)
            dec.append(dict(crop=crop, heat=heat, kind=kind, full=full, other_temperature_effects=nn,
                            heat_functions=nw - nn, water_stress=nh - nn, interaction=full - nh - nw + nn))
D = pd.DataFrame(dec)
if len(D):
    print("\n+1 degC effect (% of trend) and its decomposition, true (DSSAT) vs implied by the chain:")
    print(D.round(2).to_string(index=False))
    D.to_csv(X2 + "X1_decomposition.csv", index=False, float_format="%.4f")
