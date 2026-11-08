"""
24_selection_adjusted_tests.py - heat tests that account for the threshold search.

The threshold T* is chosen by minimum residual sum of squares over 26-40 degC. Tests computed at the
selected T* ignore this search and are too liberal (Hansen, 1996). Each test below is therefore calibrated by
a restricted wild cluster bootstrap (Cameron et al., 2008): bootstrap samples are generated from a model that
satisfies the null hypothesis of that test, residuals are multiplied by a Rademacher sign shared by all
departments in a year, each sample is detrended within department, and the WHOLE procedure (threshold search,
then year-clustered statistic at the selected threshold) is repeated. p = (1 + #{T_b >= T_obs}) / (B + 1).

  Joint heat test, H0: beta_GDD = beta_EDD = 0.
      Restricted model = water-only model (rainfall, rainfall^2, rain days, longest dry spell, window start).
      Statistic = year-clustered Wald test of GDD and EDD at the selected threshold.

  EDD test, H0: beta_EDD = 0 (GDD left free).
      Restricted model = GDD + water model, i.e. yields may respond to warmth up to a plateau T_R, with no
      effect above it. T_R is estimated under the restriction (minimum RSS over the same grid).
      Statistic = |year-clustered t| of the EDD coefficient at the selected threshold.
      Because T_R is weakly identified under H0, a conservative p value is also reported: the largest p value
      over all plateau values T_R in the grid.

  For transparency the script also reports the EDD statistic calibrated under the water-only model (the
  previous version of this test), and p values with the threshold held at its selected value (no search).
  Maize keeps its fixed 32 degC threshold (no search).

Optional Monte Carlo size check (argument mc=M): M pseudo-datasets are generated under each null from the
fitted restricted model; each is analysed exactly like the real data, with B_mc bootstrap samples, and the
rejection rate at the 5% level is recorded for the bootstrap test and for the naive t(30) / F(2,30) tests
computed at the selected threshold.

Usage: python3 24_selection_adjusted_tests.py <work_dir> [B] [mc=M] [bmc=B_mc]
Writes <work_dir>/selection_adjusted_tests_main.csv (and selection_adjusted_size_check_main.csv with mc)
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats

work = sys.argv[1]
B = int(sys.argv[2]) if len(sys.argv) > 2 and "=" not in sys.argv[2] else 9999
opt = dict(a.split("=") for a in sys.argv[2:] if "=" in a)
MC = int(opt.get("mc", 0))
BMC = int(opt.get("bmc", 399))
TAG = "main"
CROPS = ["MILLET", "GROUNDNUT"]          # E1: the two simulated crops only
GRID = list(range(26, 41))
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


class Panel:
    """Detrended design of one crop and the vectorised statistics used by every test."""

    def __init__(self, crop):
        g = panel[panel.crop == crop].reset_index(drop=True)
        ctrl = res.loc[crop, "controls"].split("+")
        self.fixed = bool(res.loc[crop, "Tstar_fixed"])
        self.grid = [int(res.loc[crop, "Tstar"])] if self.fixed else GRID
        cols = ["DD_8"] + [f"DD_{b}" for b in GRID] + ["P100", "P100sq"] + ctrl + [YVAR]
        Z = detrend_within(g, cols)
        self.y = Z[YVAR].values.astype(float)
        self.W = Z[["P100", "P100sq"] + ctrl].values.astype(float)
        self.gdd = {T: Z.DD_8.values - Z[f"DD_{T}"].values for T in GRID}
        self.edd = {T: Z[f"DD_{T}"].values for T in GRID}
        N, n_fe = len(self.y), g.department.nunique()
        years = np.sort(g.year.unique())
        self.yidx = np.searchsorted(years, g.year.values)
        self.G = len(years)
        K = 2 + self.W.shape[1] + 2 * n_fe              # intercept and slope absorbed per department
        self.cfac = self.G / (self.G - 1) * (N - 1) / (N - K)
        self.groups = [np.flatnonzero(self.yidx == j) for j in range(self.G)]
        self.dep = []
        for _, d in g.groupby("department"):
            ix = d.index.values
            t = d.year.values - d.year.mean()
            Xd = np.column_stack([np.ones(len(ix)), t])
            self.dep.append((ix, Xd @ np.linalg.inv(Xd.T @ Xd) @ Xd.T))
        self.X = {T: np.column_stack([self.gdd[T], self.edd[T], self.W]) for T in GRID}
        self.inv = {T: np.linalg.inv(self.X[T].T @ self.X[T]) for T in GRID}

    def redetrend(self, Y):
        Y = Y.copy()
        for ix, H in self.dep:
            Y[ix] -= H @ Y[ix]
        return Y

    def full_stats(self, Y):
        """For each candidate T: RSS, t of EDD, Wald of (GDD, EDD); arrays (len(grid), b)."""
        out = []
        for T in self.grid:
            X, inv = self.X[T], self.inv[T]
            beta = inv @ X.T @ Y
            U = Y - X @ beta
            rss = (U ** 2).sum(axis=0)
            S = np.stack([X[ix].T @ U[ix] for ix in self.groups])
            V = self.cfac * np.einsum("ij,gjb,glb,lk->bik", inv, S, S, inv, optimize=True)
            tE = beta[1] / np.sqrt(V[:, 1, 1])
            bH = beta[:2].T
            wald = np.einsum("bi,bij,bj->b", bH, np.linalg.inv(V[:, :2, :2]), bH)
            out.append((rss, tE, wald))
        rss, tE, wald = (np.array(z) for z in zip(*out))
        return rss, tE, wald

    def selected(self, Y):
        rss, tE, wald = self.full_stats(Y)
        sel = rss.argmin(axis=0)
        cols = np.arange(Y.shape[1])
        return tE[sel, cols], wald[sel, cols], np.array(self.grid)[sel], tE, wald

    def restricted(self, y, null, T_R=None):
        """Fitted values and residuals of the restricted model; for the EDD null, T_R by minimum RSS."""
        if null == "joint":
            X = self.W
            b = np.linalg.lstsq(X, y, rcond=None)[0]
            return X @ b, y - X @ b, None
        cand = self.grid if T_R is None else [T_R]
        best = None
        for T in cand:
            X = np.column_stack([self.gdd[T], self.W])
            b = np.linalg.lstsq(X, y, rcond=None)[0]
            r = y - X @ b
            if best is None or r @ r < best[0]:
                best = (r @ r, X @ b, r, T)
        return best[1], best[2], best[3]

    def bootstrap(self, fit, e, v):
        return self.redetrend(fit[:, None] + e[:, None] * v[self.yidx])


def pval(boot, obs):
    return (1 + np.sum(boot >= obs)) / (len(boot) + 1)


rows, size_rows = [], []
for crop in CROPS:
    P = Panel(crop)
    rng = np.random.default_rng(20261030)
    v = rng.choice([-1.0, 1.0], size=(P.G, B))
    t_obs, w_obs, T_obs, tE_all, w_all = P.selected(P.y[:, None])
    t_obs, w_obs, T_obs = t_obs[0], w_obs[0], int(T_obs[0])
    i_obs = P.grid.index(T_obs)
    assert T_obs == int(res.loc[crop, "Tstar"])
    assert abs(t_obs - res.loc[crop, "t_edd"]) < 1e-3 and abs(w_obs - res.loc[crop, "wald_heat"]) < 1e-2

    # joint test: water-only null
    fit, e, _ = P.restricted(P.y, "joint")
    t_b, w_b, _, tE_b, w_all_b = P.selected(P.bootstrap(fit, e, v))
    p_joint = pval(w_b, w_obs)
    p_joint_fixed = pval(w_all_b[i_obs], w_obs)
    p_edd_wateronly = pval(np.abs(t_b), abs(t_obs))

    # EDD test: beta_EDD = 0 with GDD free; plateau T_R estimated under the restriction
    fit, e, T_R = P.restricted(P.y, "edd")
    t_b, _, _, tE_b, _ = P.selected(P.bootstrap(fit, e, v))
    p_edd = pval(np.abs(t_b), abs(t_obs))
    p_edd_fixed = pval(np.abs(tE_b[i_obs]), abs(t_obs))
    # conservative: maximum p over all plateau values
    p_by_TR = {}
    for T in P.grid:
        fit_T, e_T, _ = P.restricted(P.y, "edd", T_R=T)
        tb_T = P.selected(P.bootstrap(fit_T, e_T, v))[0]
        p_by_TR[T] = pval(np.abs(tb_T), abs(t_obs))
    row = {"crop": crop, "Tstar": T_obs, "threshold_searched": not P.fixed, "B": B,
           "t_edd_obs": t_obs, "wald_heat_obs": w_obs,
           "p_joint_search": p_joint, "p_joint_fixedT": p_joint_fixed,
           "T_R_plateau": T_R, "p_edd_search": p_edd, "p_edd_fixedT": p_edd_fixed,
           "p_edd_search_max_over_TR": max(p_by_TR.values()),
           "TR_at_max_p": max(p_by_TR, key=p_by_TR.get),
           "p_edd_search_wateronly_null": p_edd_wateronly,
           "p_edd_naive_t30": 2 * stats.t.sf(abs(t_obs), 30),
           "p_joint_naive_F230": stats.f.sf(w_obs / 2, 2, 30)}
    rows.append(row)
    print({k: (round(v_, 4) if isinstance(v_, float) else v_) for k, v_ in row.items()}, flush=True)

    # ---------------- optional Monte Carlo size check ----------------
    if MC and not P.fixed:
        rng_mc = np.random.default_rng(7)
        crit_t, crit_f = stats.t.ppf(0.975, 30), stats.f.ppf(0.95, 2, 30)
        for null in ("joint", "edd"):
            fit0, e0, _ = P.restricted(P.y, null)
            rej_boot = rej_naive = 0
            for m in range(MC):
                y_m = P.bootstrap(fit0, e0, rng_mc.choice([-1.0, 1.0], size=(P.G, 1)))[:, 0]
                t_m, w_m, *_ = P.selected(y_m[:, None])
                fit_m, e_m, _ = P.restricted(y_m, null)
                v_m = rng_mc.choice([-1.0, 1.0], size=(P.G, BMC))
                t_mb, w_mb, *_ = P.selected(P.bootstrap(fit_m, e_m, v_m))
                if null == "joint":
                    rej_boot += pval(w_mb, w_m[0]) <= 0.05
                    rej_naive += w_m[0] / 2 > crit_f
                else:
                    rej_boot += pval(np.abs(t_mb), abs(t_m[0])) <= 0.05
                    rej_naive += abs(t_m[0]) > crit_t
            size_rows.append({"crop": crop, "null": null, "M": MC, "B_mc": BMC,
                              "rejection_bootstrap_5pct": rej_boot / MC,
                              "rejection_naive_at_selected_T_5pct": rej_naive / MC})
            print(size_rows[-1], flush=True)

pd.DataFrame(rows).to_csv(f"{work}/selection_adjusted_tests_{TAG}.csv", index=False, float_format="%.5f")
if size_rows:
    pd.DataFrame(size_rows).to_csv(f"{work}/selection_adjusted_size_check_{TAG}.csv", index=False,
                                   float_format="%.4f")
