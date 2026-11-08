"""Full-resolution runs (every cropped pixel) against DAPSA: calibration 1986-2000, validation 2001-2016, compared
with the 1/9 subsample used during calibration. Figure: national series and department means, both crops."""
import json
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import department_yields as E  # noqa: E402
import paths  # noqa: E402
P = E.P
RUNS = {"MILLET": ("millet_full_1986_2016.csv", "millet_calibrated_v2_1986_2016.csv", "HWAM"),
        "GROUNDNUT": ("groundnut_full_1986_2016.csv", "groundnut_calibrated_1986_2016.csv", "PWAM")}


def dept(res, crop):
    return E.dept_table(res, crop="MILLET") if crop == "MILLET" else E.dept_groundnut(res)


rows, figdata = [], {}
for crop, (full_f, sub_f, ycol) in RUNS.items():
    if not os.path.exists(P + full_f):
        continue
    for label, f in (("all pixels", full_f), ("1/9 subsample", sub_f)):
        res = pd.read_csv(P + f)
        d = dept(res, crop)
        for period, (a, b) in (("1986-2000", (1986, 2000)), ("2001-2016", (2001, 2016)), ("2001-2016 without 1997", (2001, 2016))):
            if "without" in period:
                continue
            m, nat, dd = E.metrics(d[(d.year >= a) & (d.year <= b)])
            dv = dd.groupby("dept")[["yield_kg_ha", "sim_kg_ha"]].mean()
            rows.append(dict(crop=crop, run=label, period=period, pixels=res.pix.nunique(), seasons=len(res),
                             fail_lt100=(res[ycol] < 100).mean(), bias_pct=m["bias_pct"], nrmse_pct=m["nrmse_pct"],
                             r_national=m["r_national"], r_dept_median=m["r_dept_median"], sd_ratio=m["sd_ratio_national"],
                             r_dept_means=np.corrcoef(dv.yield_kg_ha, dv.sim_kg_ha)[0, 1],
                             slope_dept_means=np.polyfit(dv.yield_kg_ha, dv.sim_kg_ha, 1)[0]))
        m_all, nat_all, d_all = E.metrics(d)
        m_no97, _, _ = E.metrics(d[d.year != 1997])
        rows.append(dict(crop=crop, run=label, period="1986-2016", pixels=res.pix.nunique(), seasons=len(res),
                         fail_lt100=(res[ycol] < 100).mean(), bias_pct=m_all["bias_pct"], nrmse_pct=m_all["nrmse_pct"],
                         r_national=m_all["r_national"], r_dept_median=m_all["r_dept_median"],
                         sd_ratio=m_all["sd_ratio_national"]))
        rows.append(dict(crop=crop, run=label, period="1986-2016 without 1997", r_national=m_no97["r_national"],
                         r_dept_median=m_no97["r_dept_median"], sd_ratio=m_no97["sd_ratio_national"],
                         bias_pct=m_no97["bias_pct"], nrmse_pct=m_no97["nrmse_pct"]))
        if label == "all pixels":
            w = d.dept.map(d.groupby("dept").production_t.mean())
            nat_kg = d.assign(w=w).groupby("year").apply(lambda x: pd.Series(
                {"obs": np.average(x.yield_kg_ha, weights=x.w), "sim": np.average(x.sim_kg_ha, weights=x.w)}))
            dv = d[d.year >= 2001].groupby("dept")[["yield_kg_ha", "sim_kg_ha"]].mean()
            figdata[crop] = (nat_kg, dv, nat_all)
tab = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(tab.round(2).to_string(index=False))
tab.to_csv(P + "full_resolution_metrics.csv", index=False)
for crop, (nat_kg, dv, nat_anom) in figdata.items():
    print(crop, "1997 national anomaly (%% of trend): observed %.0f, simulated %.0f" % (nat_anom.loc[1997, "obs"], nat_anom.loc[1997, "sim"]))

if len(figdata) == 2:
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.6), gridspec_kw={"width_ratios": [1.7, 1]})
    # one scale per variable: both crops share the yield axis of the national series and of the department means
    ymax_nat = max(v[0].max().max() for v in figdata.values()) * 1.18
    lim = [0, max(v[1].max().max() for v in figdata.values()) * 1.1]
    for i, (crop, col, lab) in enumerate((("MILLET", "#2166ac", "Millet grain yield (kg/ha)"),
                                          ("GROUNDNUT", "#b2182b", "Groundnut pod yield (kg/ha)"))):
        nat_kg, dv, _ = figdata[crop]
        a = ax[i, 0]
        a.axvspan(1985.5, 2000.5, color="0.93", zorder=0)
        a.text(1993, 0.97, "calibration", transform=a.get_xaxis_transform(), ha="center", va="top", fontsize=9, color="0.35")
        a.text(2008.5, 0.97, "validation", transform=a.get_xaxis_transform(), ha="center", va="top", fontsize=9, color="0.35")
        a.plot(nat_kg.index, nat_kg.obs, color="black", lw=1.8, label="DAPSA (observed)")
        a.plot(nat_kg.index, nat_kg.sim, color=col, lw=1.8, label="DSSAT, all pixels")
        a.set_ylabel(lab); a.grid(alpha=0.25)
        a.set_ylim(0, ymax_nat)
        a.legend(frameon=False, fontsize=9, loc="lower left")
        a.set_title(f"({'a' if i == 0 else 'c'}) {'Millet' if crop == 'MILLET' else 'Groundnut'}: national yield, production-weighted",
                    loc="left", fontsize=10.5)
        b = ax[i, 1]
        b.plot(lim, lim, color="0.6", lw=1)
        b.scatter(dv.yield_kg_ha, dv.sim_kg_ha, s=26, color=col, edgecolor="white", lw=0.5)
        r = np.corrcoef(dv.yield_kg_ha, dv.sim_kg_ha)[0, 1]
        b.set_xlim(lim); b.set_ylim(lim); b.grid(alpha=0.25)
        b.set_xlabel("DAPSA mean 2001–2016 (kg/ha)"); b.set_ylabel("DSSAT mean 2001–2016 (kg/ha)")
        b.set_title(f"({'b' if i == 0 else 'd'}) Department means, validation   r = {r:.2f}", loc="left", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(P + "Fig_full_resolution_validation.png", dpi=200)
    fig.savefig(paths.DSSAT + "figures_article/Fig2_validation.png", dpi=1000)
    print("figure saved")
