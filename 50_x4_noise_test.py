"""X4, noise test: does the statistical chain still recover the true warming sensitivity, and where does its threshold
fall, when DSSAT yields carry noise as large as in DAPSA?
Noise = the residuals of the chain fitted on DAPSA (main specification), each year's cross-department pattern moved to
a random year (permutation of the 31 years; a missing department-year takes a random residual of the same
department), scaled by k = 0 to 3 (step 0.5), added to the DSSAT anomalies (% of trend). Each noisy panel is analysed
like 07 (within-department detrending, T* by minimum RSS over 26-40 degC, year-clustered t of EDD, +1 degC effect);
the estimator is vectorised over replicates and checked against 07 at k = 0 and on DAPSA.
usage: python x4_noise.py [R]   -> x2x3/X4_noise_replicates.csv, x2x3/X4_noise_summary.csv
       python x4_noise.py summary  (summary only, from the replicates file)"""
import sys
import numpy as np
import pandas as pd
from scipy import stats
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402

AFM = paths.AFM
E2 = paths.DSSAT + "e2/"
X = paths.DSSAT + "x2x3/"
ONLY_SUMMARY = len(sys.argv) > 1 and sys.argv[1] == "summary"
R = int(sys.argv[1]) if len(sys.argv) > 1 and not ONLY_SUMMARY else 1000
CTRL = ["rain_days", "cdd_max", "start_day_from_1may"]
GRID = list(range(26, 41))
KS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
rng = np.random.default_rng(2611)
ex = pd.read_csv(AFM + "work/exposures_department_crop.csv").query("calendar == 'dynamic'")
true = pd.read_csv(X + "X1_true_vs_chain.csv").query("dT == 1").set_index(["crop", "heat", "water"]).true_pct


class Chain:
    """The 07 estimator for one crop panel, vectorised over many outcome vectors."""

    def __init__(self, yfile, crop):
        y = pd.read_csv(yfile)
        d = y[(y.crop == crop) & y.anom_lin_pct.notna()].merge(ex, on=["department", "crop", "year"], how="left")
        d = d.sort_values(["department", "year"]).reset_index(drop=True)
        d["P100"] = d.P_mm / 100.0
        d["P100sq"] = d.P100 ** 2
        self.d = d
        self.groups = [np.flatnonzero(d.department.values == g) for g in d.department.unique()]
        self.H = []
        for ix in self.groups:
            t = d.year.values[ix] - d.year.values[ix].mean()
            Xd = np.column_stack([np.ones(len(ix)), t])
            self.H.append(Xd @ np.linalg.pinv(Xd))
        cols = [f"DD_{b}" for b in range(7, 43)] + ["P100", "P100sq"] + CTRL
        Z = self.detrend(d[cols].values.astype(float))
        Z = pd.DataFrame(Z, columns=cols)
        W = Z[["P100", "P100sq"] + CTRL].values
        self.X = {T: np.column_stack([Z.DD_8 - Z[f"DD_{T}"], Z[f"DD_{T}"], W]) for T in GRID}
        self.inv = {T: np.linalg.inv(self.X[T].T @ self.X[T]) for T in GRID}
        years = np.sort(d.year.unique())
        self.yidx = np.searchsorted(years, d.year.values)
        self.clusters = [np.flatnonzero(self.yidx == j) for j in range(len(years))]
        N, K, G = len(d), self.X[GRID[0]].shape[1], len(years)
        self.cfac = G / (G - 1) * (N - 1) / (N - (K + 2 * len(self.groups)))
        self.G = G
        # change of GDD and EDD for +1 degC (raw exposures, as 07)
        self.dE = {T: (d[f"DD_{T - 1}"] - d[f"DD_{T}"]).values for T in GRID}
        self.dG = {T: ((d.DD_7 - d[f"DD_{T - 1}"]) - (d.DD_8 - d[f"DD_{T}"])).values for T in GRID}

    def detrend(self, A):
        A = np.array(A, dtype=float, copy=True)
        for ix, H in zip(self.groups, self.H):
            A[ix] -= H @ A[ix]
        return A

    def fit(self, Y):
        """Y (N, R) anomalies -> T*, b_gdd, b_edd, t_edd, +1 degC effect, within R2 for every column."""
        Y = self.detrend(Y)
        tss = (Y ** 2).sum(axis=0)
        res = []
        for T in GRID:
            X, inv = self.X[T], self.inv[T]
            B = inv @ X.T @ Y
            U = Y - X @ B
            res.append(((U ** 2).sum(axis=0), B, U))
        rss = np.array([r[0] for r in res])
        best = rss.argmin(axis=0)
        cols = np.arange(Y.shape[1])
        out = dict(Tstar=np.array(GRID)[best], r2=1 - rss[best, cols] / tss)
        bg, be, te, w1 = (np.empty(Y.shape[1]) for _ in range(4))
        for i, T in enumerate(GRID):
            m = best == i
            if not m.any():
                continue
            X, inv = self.X[T], self.inv[T]
            B, U = res[i][1][:, m], res[i][2][:, m]
            S = np.stack([X[ix].T @ U[ix] for ix in self.clusters])          # (G, K, r)
            meat = np.einsum("gkr,glr->rkl", S, S)
            V = self.cfac * np.einsum("ij,rjl,lk->rik", inv, meat, inv)
            bg[m], be[m] = B[0], B[1]
            te[m] = B[1] / np.sqrt(V[:, 1, 1])
            w1[m] = B[1] * self.dE[T].mean() + B[0] * self.dG[T].mean()
        out.update(b_gdd=bg, b_edd=be, t_edd=te, warm1C=w1)
        return out


if ONLY_SUMMARY:
    D = pd.read_csv(X + "X4_noise_replicates.csv")
# DAPSA residuals of the main specification (noise source) and check against 07
obsfile = AFM + "outputs/DAPSA_yields_detrended_1986-2016.csv"
mr = pd.read_csv(AFM + "work/model_results_main.csv").set_index("crop")
noise = {}
for crop in (() if ONLY_SUMMARY else ("MILLET", "GROUNDNUT")):
    C = Chain(obsfile, crop)
    y0 = C.d.anom_lin_pct.values[:, None]
    f = C.fit(y0)
    T = int(f["Tstar"][0])
    assert T == int(mr.loc[crop, "Tstar"]), (crop, T)
    assert abs(f["b_edd"][0] - mr.loc[crop, "b_edd_pct_per_dd"]) < 1e-3
    assert abs(f["warm1C"][0] - mr.loc[crop, "warm1C_pct"]) < 1e-2
    assert abs(f["t_edd"][0] - mr.loc[crop, "t_edd"]) < 1e-2
    Y = C.detrend(y0)[:, 0]
    u = Y - C.X[T] @ (C.inv[T] @ C.X[T].T @ Y)
    noise[crop] = C.d[["department", "year"]].assign(u=u)
    print(f"DAPSA {crop}: T*={T}, b_EDD={f['b_edd'][0]:.3f}, +1C={f['warm1C'][0]:.2f} (07 reproduced); "
          f"residual SD {u.std():.1f} %", flush=True)

rows = []
for crop in (() if ONLY_SUMMARY else ("MILLET", "GROUNDNUT")):
    for heat in ("orig", "neutral"):
        name = f"{crop.lower()}_{heat}_rainfed"
        C = Chain(E2 + f"chain/{name}/outputs/DAPSA_yields_detrended_1986-2016.csv", crop)
        ref = pd.read_csv(E2 + f"chain/{name}/work/model_results_main.csv").iloc[0]
        base = C.fit(C.d.anom_lin_pct.values[:, None])
        assert int(base["Tstar"][0]) == int(ref.Tstar) and abs(base["warm1C"][0] - ref.warm1C_pct) < 1e-2
        # noise matrix: residual of department d in a permuted year
        nz = noise[crop].pivot(index="department", columns="year", values="u")
        years = nz.columns.values
        dep, yr = C.d.department.values, C.d.year.values
        di = nz.index.get_indexer(dep)
        E = np.empty((len(C.d), R))
        for r in range(R):
            perm = dict(zip(years, rng.permutation(years)))
            src = np.array([perm[v] for v in yr])
            e = nz.values[di, np.searchsorted(years, src)]
            miss = np.isnan(e)
            for i in np.flatnonzero(miss):
                pool = nz.values[di[i]][~np.isnan(nz.values[di[i]])]
                e[i] = rng.choice(pool)
            E[:, r] = e
        y = C.d.anom_lin_pct.values[:, None]
        for k in KS:
            f = C.fit(y + k * E) if k > 0 else C.fit(y)
            n = len(f["Tstar"])
            rows.append(pd.DataFrame(dict(crop=crop, heat=heat, k=k, rep=np.arange(n), **f)))
        print(name, "done", flush=True)
if not ONLY_SUMMARY:
    D = pd.concat(rows, ignore_index=True)
    D.to_csv(X + "X4_noise_replicates.csv", index=False, float_format="%.4f")


def summ(g):
    tr = true.loc[(g.name[0], g.name[1], "rainfed")]
    sig = (g.t_edd < -stats.t.ppf(0.975, 30))
    return pd.Series(dict(n=len(g), r2_median=g.r2.median(), Tstar_median=g.Tstar.median(),
                          share_T_31_34=((g.Tstar >= 31) & (g.Tstar <= 34)).mean(),
                          share_T_35_40=(g.Tstar >= 35).mean(), share_T_26_27=(g.Tstar <= 27).mean(),
                          share_edd_neg_naive=sig.mean(), true_pct=tr, warm1C_median=g.warm1C.median(),
                          warm1C_p05=g.warm1C.quantile(0.05), warm1C_p95=g.warm1C.quantile(0.95),
                          bias=g.warm1C.mean() - tr, rmse=np.sqrt(((g.warm1C - tr) ** 2).mean())))


S = D.groupby(["crop", "heat", "k"])[["Tstar", "r2", "t_edd", "warm1C"]].apply(summ).reset_index()
pd.set_option("display.width", 250)
print(S.round(3).to_string(index=False))
S.to_csv(X + "X4_noise_summary.csv", index=False, float_format="%.4f")
