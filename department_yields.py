"""Department aggregation of pixel yields and comparison with DAPSA (millet)."""
import numpy as np
import pandas as pd
import xarray as xr
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402

P = paths.DSSAT + "pipeline/"
_pix = pd.read_csv(P + "pixels.csv")
_m = xr.open_dataset(paths.OUTPUTS + "crop_masks_0p0375.nc")
_dep = xr.open_dataset(paths.OUTPUTS + "department_mask_0p0375.nc").dept_id.values
_names = pd.read_csv(paths.OUTPUTS + "department_ids.csv").set_index("dept_id").department
_obs = pd.read_csv(paths.OUTPUTS + "DAPSA_yields_detrended_1986-2016.csv")
_keep = pd.read_csv(paths.OUTPUTS + "DAPSA_series_selection.csv").query("kept")
_obs = _obs.merge(_keep[["department", "crop"]], on=["department", "crop"])
_obs = _obs[_obs.yield_kg_ha.notna()].rename(columns={"department": "dept"})


def dept_table(res, crop="MILLET"):
    r = res.merge(_pix[["i", "j", "lat"]], left_on="pix", right_index=True)
    r["dept"] = _names.reindex(_dep[r.i, r.j]).values
    r["w"] = _m[f"frac_{crop}"].values[r.i, r.j] * np.cos(np.deg2rad(r.lat))
    s = (r.assign(wy=r.w * r.HWAM).groupby(["dept", "year"]).agg(wy=("wy", "sum"), w=("w", "sum")).reset_index())
    s["sim_kg_ha"] = s.wy / s.w
    o = _obs[_obs.crop == crop]
    return o.merge(s[["dept", "year", "sim_kg_ha"]], on=["dept", "year"])


def anom(g, col):
    t = g.year - g.year.mean()
    tr = np.polyval(np.polyfit(t, g[col], 1), t)
    return 100 * (g[col] - tr) / tr


def metrics(d):
    d = d.copy()
    d["sim_anom"] = d.groupby("dept", group_keys=False).apply(lambda g: anom(g, "sim_kg_ha"))
    d["obs_anom"] = d.groupby("dept", group_keys=False).apply(lambda g: anom(g, "yield_kg_ha"))
    w = d.dept.map(d.groupby("dept").production_t.mean())
    nat = d.assign(w=w).groupby("year").apply(
        lambda x: pd.Series({"obs": np.average(x.obs_anom, weights=x.w), "sim": np.average(x.sim_anom, weights=x.w),
                             "obs_kg": np.average(x.yield_kg_ha, weights=x.w),
                             "sim_kg": np.average(x.sim_kg_ha, weights=x.w)}))
    r_dept = d.groupby("dept").apply(lambda g: np.corrcoef(g.sim_anom, g.obs_anom)[0, 1])
    e = d.sim_kg_ha - d.yield_kg_ha
    return dict(dept_years=len(d), obs_mean=d.yield_kg_ha.mean(), sim_mean=d.sim_kg_ha.mean(),
                bias_pct=100 * e.mean() / d.yield_kg_ha.mean(), rmse=np.sqrt((e ** 2).mean()),
                nrmse_pct=100 * np.sqrt((e ** 2).mean()) / d.yield_kg_ha.mean(),
                r_levels=np.corrcoef(d.sim_kg_ha, d.yield_kg_ha)[0, 1],
                r_dept_median=r_dept.median(), r_national=np.corrcoef(nat.obs, nat.sim)[0, 1],
                sd_ratio_national=nat.sim.std() / nat.obs.std()), nat, d


def dept_groundnut(res):
    """Groundnut: department pod yield (PWAM), aggregated like dept_table."""
    return dept_table(res.assign(HWAM=res.PWAM), crop="GROUNDNUT")
