"""Point 7 check: groundnut, plant thresholds -3 degC, no water stress.
(1) independent refit of t(EDD) at T* = 37 with statsmodels (explicit department intercepts and
    trends, year-clustered SEs with the same small-sample factor);
(2) rebuild of the restricted wild cluster bootstrap under the EDD null, with the draws kept;
(3) diagnostics: where the large bootstrap statistics come from."""
import sys
import numpy as np, pandas as pd
import statsmodels.api as sm
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402
sys.argv = ["x", paths.DSSAT + "e2/chain/groundnut_minus3_nowaterstress/work", "9999"]
src = open(paths.SCRIPTS + "04_search_adjusted_tests.py").read()
src = src.split("rows, size_rows = [], []")[0]          # keep the definitions only
exec(src)
W = sys.argv[1]
P = Panel("GROUNDNUT")
g = panel[panel.crop == "GROUNDNUT"].reset_index(drop=True)

# (1) independent t at 37
ctrl = res.loc["GROUNDNUT", "controls"].split("+")
D = pd.get_dummies(g.department).astype(float).values
tc = g.year.values - g.year.mean()
X = np.column_stack([g.DD_8 - g.DD_37, g.DD_37, g.P100, g.P100sq, g[ctrl].values, D, D * tc[:, None]])
m = sm.OLS(g.anom_lin_pct.values, X).fit(cov_type="cluster", cov_kwds={"groups": g.year.values})
print("statsmodels: b_EDD %.4f  se %.4f  t %.3f" % (m.params[1], m.bse[1], m.tvalues[1]),
      "| stored t", round(res.loc["GROUNDNUT", "t_edd"], 3))
# statsmodels cluster correction uses K = number of columns; same as the script's K -> identical t expected

# observed statistics at every candidate T
rss, tE, wald = P.full_stats(P.y[:, None])
prof = pd.DataFrame({"T": P.grid, "rss": rss[:, 0], "t_EDD": tE[:, 0]})
prof["r2_within"] = 1 - prof.rss / (P.y @ P.y)
print(prof.round(4).to_string(index=False))

# (2) restricted wild cluster bootstrap under the EDD null (as in script 24)
rng = np.random.default_rng(20261030)
v = rng.choice([-1.0, 1.0], size=(P.G, 9999))
fit, e, T_R = P.restricted(P.y, "edd")
print("T_R (plateau under the null):", T_R, " restricted within R2 %.3f" % (1 - e @ e / (P.y @ P.y)))
t_b, _, T_b, tE_b, _ = P.selected(P.bootstrap(fit, e, v))
t_obs = tE[P.grid.index(37), 0]
p = (1 + np.sum(np.abs(t_b) >= abs(t_obs))) / (len(t_b) + 1)
print("search-adjusted p reproduced: %.4f" % p)
q = np.percentile(np.abs(t_b), [50, 75, 90, 95, 99])
print("bootstrap |t| at the selected T: median %.2f, p75 %.2f, p90 %.2f, p95 %.2f, p99 %.2f" % tuple(q))
sel = pd.Series(T_b)
big = np.abs(t_b) >= abs(t_obs)
print("selected T in bootstrap draws (share):", sel.value_counts(normalize=True).sort_index().round(3).to_dict())
print("selected T among draws with |t| >= %.2f:" % abs(t_obs), sel[big].value_counts(normalize=True).sort_index().round(3).to_dict())
i37 = P.grid.index(37)
print("p at fixed T=37: %.4f" % ((1 + np.sum(np.abs(tE_b[i37]) >= abs(t_obs))) / (len(t_b) + 1)))
# other seed
v2 = np.random.default_rng(12345).choice([-1.0, 1.0], size=(P.G, 9999))
t_b2 = P.selected(P.bootstrap(fit, e, v2))[0]
print("other seed p: %.4f" % ((1 + np.sum(np.abs(t_b2) >= abs(t_obs))) / 10000))

# (3) diagnostics: residual structure and effective clusters at high thresholds
ed = {T: P.edd[T] for T in P.grid}
yr = g.year.values
for T in [36, 37, 38, 39, 40]:
    raw = g[f"DD_{T}"].values
    by = pd.Series(raw).groupby(yr).sum()
    share = (by / by.sum()).sort_values(ascending=False)
    print(f"T={T}: years with EDD>0: {int((by > 1e-6).sum())}/31; top-3 years share of EDD: {share.iloc[:3].sum():.2f}")
# correlation of restricted residuals with EDD(37)
print("corr(restricted residual, EDD37 detrended) = %.3f" % np.corrcoef(e, P.edd[37])[0, 1])
# how much of the restricted residual is structured: R2 of residual on EDD at all T (unrestricted bins)
Xb = np.column_stack([P.edd[T] for T in P.grid])
bb = np.linalg.lstsq(Xb, e, rcond=None)[0]
print("share of restricted-residual variance explained by EDD(26..40): %.2f" % (1 - ((e - Xb @ bb) @ (e - Xb @ bb)) / (e @ e)))
np.savez_compressed(paths.DSSAT + "e2/gn_minus3_ns_bootstrap.npz", t_b=t_b, T_b=T_b, t_obs=t_obs)
