"""E1 summary table: AFM main specification on DAPSA and DSSAT yields, with the controls
(2001-2016 only, without 1997, SLPF -0.1/+0.1 on the 1/9 subsample, with its own baseline)."""
import os
import pandas as pd
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402

E1 = paths.DSSAT + "e1/"
VARIANTS = [("obs", "DAPSA", "1986-2016"), ("sim", "DSSAT", "1986-2016"),
            ("obs_2001", "DAPSA", "2001-2016 only"), ("sim_2001", "DSSAT", "2001-2016 only"),
            ("obs_no1997", "DAPSA", "without 1997"), ("sim_no1997", "DSSAT", "without 1997"),
            ("sim_s3", "DSSAT 1/9 subsample", "calibrated SLPF"), ("sim_s3_lo", "DSSAT 1/9 subsample", "SLPF - 0.1"),
            ("sim_s3_hi", "DSSAT 1/9 subsample", "SLPF + 0.1")]
rows = []
for v, src, ctrl in VARIANTS:
    f = E1 + f"{v}/work/model_results_main.csv"
    if not os.path.exists(f):
        continue
    r = pd.read_csv(f)
    for _, x in r.iterrows():
        row = dict(crop=x.crop, source=src, control=ctrl, n_obs=int(x.n_obs), Tstar=int(x.Tstar),
                   Tstar_p05=x.Tstar_boot_p05, Tstar_p95=x.Tstar_boot_p95, b_edd=x.b_edd_pct_per_dd, t_edd=x.t_edd,
                   b_gdd=x.b_gdd_pct_per_dd, r2_within=x.r2_within, r2_water_only=x.r2_within_water_only,
                   warm1C=x.warm1C_pct, warm1C_p05=x.warm1C_boot_p05, warm1C_p95=x.warm1C_boot_p95)
        fa = E1 + f"{v}/work/selection_adjusted_tests_main.csv"
        if os.path.exists(fa):
            a = pd.read_csv(fa).set_index("crop").loc[x.crop]
            row.update(p_edd_adjusted=a.p_edd_search, p_edd_adjusted_max=a.p_edd_search_max_over_TR,
                       p_joint_adjusted=a.p_joint_search)
        fu = E1 + f"{v}/work/threshold_uncertainty_main.csv"
        if os.path.exists(fu):
            u = pd.read_csv(fu).set_index("crop").loc[x.crop]
            row.update(boot_within2=u.year_within2, boot_at26=u.year_at26, loyo_min=u.loyo_min, loyo_max=u.loyo_max)
        rows.append(row)
tab = pd.DataFrame(rows).sort_values(["crop", "control", "source"], ascending=[False, True, True])
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30)
print(tab.round(3).to_string(index=False))
tab.to_csv(E1 + "E1_summary.csv", index=False, float_format="%.4f")
