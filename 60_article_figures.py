"""The eight figures of the article, in English, 1000 dpi (FIG_DPI). Figures 2 and 3 come from eval_full.py and e1_figure.py
(copied here); this script draws figures 1 and 4-8.
  1 study area: cropped pixels, cultivar zones, rainfall, soil water
  4 threshold found in DSSAT yields by heat-function x water treatment (E2)
  5 mechanism: binned temperature response, and simulated water stress against extreme degree-days
  6 true against inferred +1 degC effect, and its decomposition (X1)
  7 water controls: threshold and +1 degC effect (X2)
  8 robustness: scale of analysis (X3) and yield noise (X4)
usage: python article_figures.py [n ...]"""
import os
import re
import sys
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import department_yields as E  # noqa: E402
import paths  # noqa: E402

OUT = paths.DSSAT + "figures_article/"
AFM = paths.AFM
P = paths.DSSAT + "pipeline/"
E2 = paths.DSSAT + "e2/"
X = paths.DSSAT + "x2x3/"
INK, MUTED = "#0b0b0b", "#52514e"
CROP = {"MILLET": "#2166ac", "GROUNDNUT": "#b2182b"}
C_TRUE, C_BASE, C_SPEI, C_WSI = "#2b2b2b", "#2a78d6", "#eb6834", "#1baf7a"
NAME = {"MILLET": "Millet", "GROUNDNUT": "Groundnut"}
TSTAR_SIM = {"MILLET": 36, "GROUNDNUT": 37}
DPI = int(os.environ.get("FIG_DPI", "1000"))      # 1000 dpi for submission
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10.5, "axes.labelsize": 9})


def same_scale(axes, which="y"):
    """One scale per variable: the millet and groundnut panels share the union of their axis limits."""
    for w in which:
        lims = [getattr(a, f"get_{w}lim")() for a in axes]
        lo, hi = min(l[0] for l in lims), max(l[1] for l in lims)
        for a in axes:
            getattr(a, f"set_{w}lim")(lo, hi)


def tidy(a, grid="y"):
    if grid:
        a.grid(alpha=0.25, axis=grid)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)


def errpt(a, x, v, lo, hi, color, mk, filled=True, ms=7):
    a.errorbar([x], [v], yerr=[[v - lo], [hi - v]], fmt=mk, color=color, ms=ms, capsize=2.5, lw=1.3,
               mfc=color if filled else "white", mec="white" if filled else color, mew=1.0 if filled else 1.5)


# ------------------------------------------------------------------ figure 1
def fig1():
    m = xr.open_dataset(AFM + "outputs/crop_masks_0p0375.nc")
    dep = gpd.read_file(AFM + "outputs/departments_30.gpkg")
    pix = pd.read_csv(P + "pixels.csv")
    lat, lon = m.Lat.values, m.Lon.values
    aw, sid, top = {}, None, 0.0
    for line in open(P + "HW.SOL"):
        if line.startswith("*HW"):
            sid, top = line[1:11], 0.0
            aw[sid] = 0.0
        elif sid and re.match(r"^\s+\d+\s+-99 ", line):
            v = line.split()
            slb, ll, dul = float(v[0]), float(v[2]), float(v[3])
            aw[sid] += (dul - ll) * max(0.0, min(slb, 100.0) - top) * 10.0
            top = slb
    rain = np.full((len(lat), len(lon)), np.nan)
    soil = rain.copy()
    ok = ~pix.rain_missing
    rain[pix.i[ok], pix.j[ok]] = pix.rain_mean_mm[ok]
    soil[pix.i, pix.j] = pix.soil_id.map(aw).values
    fig, axs = plt.subplots(2, 2, figsize=(11, 8.2), layout="constrained")
    fb = [0.0025, 0.01, 0.025, 0.05, 0.10, 0.20, 0.30, 1.0]
    fcm = ListedColormap(plt.get_cmap("Greens")(np.linspace(0.25, 0.95, 7)))
    fnorm = BoundaryNorm(fb, 7)
    panels = [("MILLET", "(a) Millet: harvested fraction and cultivars"),
              ("GROUNDNUT", "(b) Groundnut: harvested fraction and cultivars"),
              ("rain", "(c) Mean annual rainfall, 1986–2016"), ("soil", "(d) Soil available water, top metre")]
    for a, (key, title) in zip(axs.flat, panels):
        if key in ("MILLET", "GROUNDNUT"):
            f = m[f"frac_{key}"].where(m[f"mask_{key}"] == 1)
            pc = a.pcolormesh(lon, lat, f, cmap=fcm, norm=fnorm, shading="nearest")
            cb = fig.colorbar(pc, ax=a, shrink=0.8)
            cb.set_ticks(fb, labels=["0.25 %", "1 %", "2.5 %", "5 %", "10 %", "20 %", "30 %", "100 %"])
            cb.set_label("Harvested fraction of the pixel")
        elif key == "rain":
            rb = np.arange(100, 1400, 100)
            pc = a.pcolormesh(lon, lat, rain, cmap=ListedColormap(plt.get_cmap("Blues")(np.linspace(0.15, 0.95, 12))),
                              norm=BoundaryNorm(rb, 12), shading="nearest")
            cs = a.contour(lon, lat, np.ma.masked_invalid(rain), levels=[300, 500, 700, 900, 1100], colors=MUTED,
                           linewidths=0.6)
            a.clabel(cs, fmt="%d", fontsize=7)
            fig.colorbar(pc, ax=a, shrink=0.8, label="Rainfall (mm per year)")
        else:
            sb = [40, 60, 80, 100, 120, 140, 160, 200]
            pc = a.pcolormesh(lon, lat, soil, cmap=ListedColormap(plt.get_cmap("Oranges")(np.linspace(0.2, 0.9, 7))),
                              norm=BoundaryNorm(sb, 7), shading="nearest")
            fig.colorbar(pc, ax=a, shrink=0.8, label="Available water, 0–100 cm (mm)")
        dep.boundary.plot(ax=a, color="#3b3b3b", lw=0.45)
        if key == "MILLET":
            a.plot([-17.6, -11.2], [15, 15], color=INK, lw=1.1, ls="--")
            a.text(-12.9, 15.12, "ICTP 8203", fontsize=8.5, color=INK, ha="center")
            a.text(-12.9, 14.72, "Souna 3", fontsize=8.5, color=INK, ha="center")
            a.plot([-16.47], [14.70], marker="*", ms=11, color=INK, mec="white", mew=0.6)
            a.text(-16.40, 14.50, "Bambey", fontsize=8, color=INK)
        if key == "GROUNDNUT":
            cs = a.contour(lon, lat, np.ma.masked_invalid(rain), levels=[700], colors=INK, linewidths=1.1,
                           linestyles="--")
            a.text(-15.2, 14.45, "Fleur 11", fontsize=8.5, color=INK, ha="center")
            a.text(-15.3, 12.8, "73-33", fontsize=8.5, color=INK, ha="center")
            a.text(-12.9, 13.66, "700 mm", fontsize=7.5, color=INK, ha="center")
        a.set_title(title, loc="left", color=INK)
        a.set_xlim(-17.6, -11.2); a.set_ylim(12.2, 16.8); a.set_aspect("equal")
        a.set_xlabel("Longitude (°E)", color=MUTED); a.set_ylabel("Latitude (°N)", color=MUTED)
        a.tick_params(colors=MUTED, labelsize=8)
    fig.savefig(OUT + "Fig1_study_area.png", dpi=DPI)


# ------------------------------------------------------------------ figure 4
def fig4():
    T = pd.read_csv(E2 + "E2_treatments.csv").query("variant == 'base'")
    obs = {"MILLET": 33, "GROUNDNUT": 31}
    heats = [("orig", "Original"), ("neutral", "Selected thermal\nfunctions\nneutralised"), ("minus3", "Thresholds\n−3 °C"),
             ("plus3", "Thresholds\n+3 °C")]
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for j, (a, crop) in enumerate(zip(axs, ("MILLET", "GROUNDNUT"))):
        g = T[T.crop == crop]
        for k, (h, _) in enumerate(heats):
            for water, dx, mk in (("rainfed", -0.12, "o"), ("nowaterstress", 0.12, "s")):
                r = g[(g.heat == h) & (g.water == water)].iloc[0]
                errpt(a, k + dx, r.Tstar, r.Tstar_p05, r.Tstar_p95, CROP[crop], mk, filled=water == "rainfed")
        a.axhline(obs[crop], color=INK, lw=1.0, ls="--")
        a.text(3.45, obs[crop] + 0.3, f"DAPSA {obs[crop]} °C", ha="right", va="bottom", fontsize=8.5)
        a.set_xticks(range(4)); a.set_xticklabels([h[1] for h in heats])
        a.set_xlim(-0.5, 3.5); a.set_ylim(25, 41.5)
        a.set_title(f"({'ab'[j]}) {NAME[crop]}", loc="left", color=INK)
        a.set_xlabel("Heat functions of the crop model")
        tidy(a)
    axs[0].set_ylabel("Threshold T* found in DSSAT yields (°C)")
    hand = [Line2D([], [], marker="o", color="0.35", ls="none", ms=7, label="Rainfed"),
            Line2D([], [], marker="s", color="0.35", mfc="white", ls="none", ms=7, label="No water stress")]
    fig.legend(handles=hand, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT + "Fig4_threshold_by_heat_and_water.png", dpi=DPI)


# ------------------------------------------------------------------ figure 5
def detrend_by(df, cols, key="department"):
    out = df.copy()
    for _, g in df.groupby(key):
        t = g.year - g.year.mean()
        Xd = np.column_stack([np.ones(len(g)), t])
        for c in cols:
            b = np.linalg.lstsq(Xd, g[c].values, rcond=None)[0]
            out.loc[g.index, c] = g[c].values - Xd @ b
    return out


def fig5():
    bins = ["24-27", "27-30", "30-33", "33-36", "36-39", ">=39"]
    xb = [25.5, 28.5, 31.5, 34.5, 37.5, 40.5]
    src = [("DAPSA (observed)", AFM + "work/binned_models_main.csv", INK, "o"),
           ("DSSAT, rainfed", E2 + "chain/{c}_orig_rainfed/work/binned_models_main.csv", C_BASE, "s"),
           ("DSSAT, no water stress", E2 + "chain/{c}_orig_nowaterstress/work/binned_models_main.csv", C_SPEI, "D")]
    ex = pd.read_csv(AFM + "work/exposures_department_crop.csv").query("calendar == 'dynamic'")
    fig, axs = plt.subplots(2, 2, figsize=(10.5, 7.6))
    for j, crop in enumerate(("MILLET", "GROUNDNUT")):
        a = axs[0, j]
        for k, (lab, f, col, mk) in enumerate(src):
            b = pd.read_csv(f.format(c=crop.lower())).query("crop == @crop").set_index("bin").loc[bins]
            dx = (k - 1) * 0.35
            a.errorbar(np.array(xb) + dx, b.coef_pct_per_day, yerr=1.96 * b.se, fmt=mk + "-", color=col, ms=6, lw=1.4,
                       capsize=2.5, mec="white", mew=0.8, label=lab)
        a.axhline(0, color="0.6", lw=0.8)
        a.set_xticks(xb); a.set_xticklabels(["24–27\n(ref.)", "27–30", "30–33", "33–36", "36–39", "≥ 39"])
        a.set_xlabel("Temperature bin (°C)")
        a.set_title(f"({'ab'[j]}) {NAME[crop]}: yield response to one more day in a bin", loc="left", color=INK)
        tidy(a)
        # water stress against extreme degree-days, DSSAT original functions, rainfed
        w = pd.read_csv(E2 + f"wsi_{crop.lower()}_orig.csv")
        d = E.dept_table(w.assign(HWAM=100 * w.wspd_cycle), crop=crop)[["dept", "year", "sim_kg_ha"]]
        d = d.rename(columns={"dept": "department", "sim_kg_ha": "wsi"})
        T = TSTAR_SIM[crop]
        d = d.merge(ex[ex.crop == crop][["department", "year", f"DD_{T}"]], on=["department", "year"])
        d = detrend_by(d.reset_index(drop=True), ["wsi", f"DD_{T}"])
        d["dec"] = pd.qcut(d[f"DD_{T}"], 10, labels=False)
        s = d.groupby("dec").agg(x=(f"DD_{T}", "mean"), y=("wsi", "mean"), se=("wsi", "sem"))
        r = np.corrcoef(d[f"DD_{T}"], d.wsi)[0, 1]
        b = axs[1, j]
        b.errorbar(s.x, s.y, yerr=1.96 * s.se, fmt="o-", color=C_BASE, ms=6, lw=1.4, capsize=2.5, mec="white", mew=0.8)
        b.axhline(0, color="0.6", lw=0.8); b.axvline(0, color="0.8", lw=0.8)
        b.text(0.03, 0.93, f"r = {r:.2f} (department-years, detrended)", transform=b.transAxes, fontsize=8.5,
               color=INK)
        b.set_xlabel(f"Degree-days above {T} °C in the crop window, anomaly (°C day)")
        b.set_title(f"({'cd'[j]}) {NAME[crop]}: simulated water stress by decile of extreme heat", loc="left",
                    color=INK)
        tidy(b, grid="both")
    axs[0, 0].set_ylabel("Yield change per day (% of trend)")
    axs[1, 0].set_ylabel("Water stress on photosynthesis,\nanomaly (% of full stress)")
    same_scale(axs[0], "y"); same_scale(axs[1], "xy")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT + "Fig5_mechanism_heat_and_water.png", dpi=DPI)


# ------------------------------------------------------------------ figure 6
CATS = [("orig", "rainfed", "Original"), ("neutral", "rainfed", "Selected\nthermal\nfunctions\nneutralised"),
        ("minus3", "rainfed", "Thresholds\n−3 °C"), ("orig", "nowaterstress", "Original"),
        ("neutral", "nowaterstress", "Selected\nthermal\nfunctions\nneutralised"), ("minus3", "nowaterstress", "Thresholds\n−3 °C")]


def fig6():
    T = pd.read_csv(X + "X1_true_vs_chain.csv").query("dT == 1")
    D = pd.read_csv(X + "X1_decomposition.csv")
    comps = [("other_temperature_effects", "Other temperature effects"), ("heat_functions", "Heat functions"),
             ("water_stress", "Water stress"), ("interaction", "Interaction")]
    fig = plt.figure(figsize=(12, 8.4)); dec = []
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.38, wspace=0.08)
    for j, crop in enumerate(("MILLET", "GROUNDNUT")):
        a = fig.add_subplot(gs[0, j], sharey=None)
        g = T[T.crop == crop].set_index(["heat", "water"])
        for k, (h, w, _) in enumerate(CATS):
            r = g.loc[(h, w)]
            errpt(a, k - 0.12, r.true_pct, r.true_p05, r.true_p95, C_TRUE, "o")
            errpt(a, k + 0.12, r.chain_warm1C, r.chain_p05, r.chain_p95, C_BASE, "s")
        a.axhline(0, color="0.6", lw=0.8); a.axvline(2.5, color="0.8", lw=0.8)
        a.set_xticks(range(6)); a.set_xticklabels([c[2] for c in CATS], fontsize=8.5)
        a.set_xlim(-0.6, 5.6); a.set_ylim(-38, 8)
        a.text(1, 5.2, "Rainfed", ha="center", fontsize=9.5); a.text(4, 5.2, "No water stress", ha="center", fontsize=9.5)
        a.set_title(f"({'ab'[j]}) {NAME[crop]}: effect of +1 °C", loc="left", color=INK)
        if j == 0:
            a.set_ylabel("Yield change for +1 °C (% of trend)")
        tidy(a)
        b = fig.add_subplot(gs[1, j]); dec.append(b)
        rows = []
        for heat, hl in (("orig", "Original functions"), ("minus3", "Thresholds −3 °C")):
            tr = D[(D.crop == crop) & (D.heat == heat) & (D.kind == "true")].iloc[0]
            ch = D[(D.crop == crop) & (D.heat == heat) & (D.kind == "chain")].iloc[0]
            for c, cl in comps:
                rows.append((f"{cl}", hl, tr[c], ch[c]))
        ypos = []
        for i, (cl, hl, tv, cv) in enumerate(rows):
            y = -(i + (1 if i >= 4 else 0))
            ypos.append(y)
            b.plot([tv, cv], [y, y], color="0.75", lw=1.2, zorder=1)
            b.plot([tv], [y], "o", color=C_TRUE, ms=7, zorder=2)
            b.plot([cv], [y], "s", color=C_BASE, ms=7, mec="white", mew=0.8, zorder=3)
        b.set_yticks(ypos); b.set_yticklabels([r[0] for r in rows] if j == 0 else [], fontsize=8.5)
        b.set_ylim(min(ypos) - 0.7, 1.1)
        for yh, txt in ((0.75, "Original heat functions"), (-4.25, "Plant thresholds −3 °C")):
            b.text(0.01, yh, txt, transform=b.get_yaxis_transform(), ha="left", va="center", fontsize=8.5,
                   color=MUTED, fontweight="bold")
        b.axvline(0, color="0.6", lw=0.8)
        b.set_xlabel("Contribution to the +1 °C effect (% of trend)")
        b.set_title(f"({'cd'[j]}) {NAME[crop]}: decomposition, heat functions × water", loc="left", color=INK)
        tidy(b, grid="x")
    same_scale(dec, "x")
    hand = [Line2D([], [], marker="o", color=C_TRUE, ls="none", ms=7, label="True: +1 °C simulated in DSSAT"),
            Line2D([], [], marker="s", color=C_BASE, ls="none", ms=7, label="Inferred by the statistical model")]
    fig.legend(handles=hand, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.95))
    fig.savefig(OUT + "Fig6_true_vs_inferred_sensitivity.png", dpi=DPI, bbox_inches="tight")


# ------------------------------------------------------------------ figure 7
def fig7():
    A = pd.read_csv(X + "X2_summary_harmonised.csv")      # one bootstrap output per treatment
    T = pd.read_csv(X + "X1_true_vs_chain.csv").query("dT == 1 and water == 'rainfed'").set_index(["crop", "heat"])
    src = [("dapsa", "DAPSA\n(observed)"), ("orig", "DSSAT\noriginal"), ("neutral", "DSSAT, selected\nthermal functions\nneutralised"),
           ("minus3", "DSSAT thresh.\n−3 °C")]
    ctl = [("main", "Rainfall controls", C_BASE, "s"), ("spei", "+ SPEI", C_SPEI, "^"),
           ("wsi", "+ simulated water stress", C_WSI, "D")]
    fig, axs = plt.subplots(2, 2, figsize=(12, 8.2))
    for j, crop in enumerate(("MILLET", "GROUNDNUT")):
        a, b = axs[0, j], axs[1, j]
        for k, (s, _) in enumerate(src):
            if s != "dapsa":
                t = T.loc[(crop, s)]
                errpt(b, k - 0.3, t.true_pct, t.true_p05, t.true_p95, C_TRUE, "o")
            for c, (cn, _, col, mk) in enumerate(ctl):
                r = A[(A.source == s) & (A.controls == cn) & (A.crop == crop)].iloc[0]
                sig = (r.p_edd_adj < 0.05) and (r.b_edd < 0)
                x = k + (c - 1) * 0.2 + (0.05 if s != "dapsa" else 0)
                errpt(a, x, r.Tstar, r.Tstar_p05, r.Tstar_p95, col, mk, filled=sig)
                errpt(b, x, r.warm1C, r.warm1C_p05, r.warm1C_p95, col, mk)
        for ax_, lab in ((a, "Threshold T* (°C)"), (b, "Yield change for +1 °C (% of trend)")):
            ax_.set_xticks(range(4)); ax_.set_xticklabels([s[1] for s in src], fontsize=8.5)
            ax_.set_xlim(-0.55, 3.45)
            tidy(ax_)
            if j == 0:
                ax_.set_ylabel(lab)
        a.set_ylim(25, 41.5)
        b.axhline(0, color="0.6", lw=0.8)
        a.set_title(f"({'ab'[j]}) {NAME[crop]}: threshold", loc="left", color=INK)
        b.set_title(f"({'cd'[j]}) {NAME[crop]}: effect of +1 °C", loc="left", color=INK)
    same_scale(axs[1], "y")
    hand = [Line2D([], [], marker=mk, color=col, ls="none", ms=7, label=lab) for _, lab, col, mk in ctl]
    hand += [Line2D([], [], marker="o", color=C_TRUE, ls="none", ms=7, label="True (DSSAT, +1 °C)"),
             Line2D([], [], marker="s", color="0.45", ls="none", ms=7, label="Significant negative EDD coefficient (adj. p < 0.05)"),
             Line2D([], [], marker="s", color="0.45", mfc="white", ls="none", ms=7, label="No significant negative EDD coefficient")]
    fig.legend(handles=hand, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT + "Fig7_water_controls.png", dpi=DPI)


# ------------------------------------------------------------------ figure 8
def fig8():
    B = pd.read_csv(X + "X3_summary_harmonised.csv")
    N = pd.read_csv(X + "X4_noise_replicates.csv")
    S = pd.read_csv(X + "X4_noise_summary.csv")
    obs = pd.read_csv(AFM + "work/model_results_main.csv").set_index("crop")
    trt = [("orig_rainfed", "Original\nrainfed"), ("neutral_rainfed", "Selected thermal\nfunctions neutralised\nrainfed"),
           ("minus3_rainfed", "Thresholds −3 °C\nrainfed"), ("orig_nowaterstress", "Original\nno water stress")]
    scl = [("department", "Department", C_BASE, "o"), ("mixed", "Pixel yields, department exposures", C_SPEI, "s"),
           ("pixel", "Pixel", C_WSI, "D")]
    classes = [("T* ≤ 27 °C (no threshold)", lambda t: t <= 27), ("28–30 °C", lambda t: (t >= 28) & (t <= 30)),
               ("31–34 °C", lambda t: (t >= 31) & (t <= 34)), ("35–40 °C", lambda t: t >= 35)]
    ccol = plt.get_cmap("Purples")(np.linspace(0.25, 0.9, 4))
    fig, axs = plt.subplots(3, 2, figsize=(12, 11.2))
    for j, crop in enumerate(("MILLET", "GROUNDNUT")):
        a = axs[0, j]
        for k, (t, _) in enumerate(trt):
            for c, (sc, _, col, mk) in enumerate(scl):
                r = B[(B.treatment == t) & (B.scale == sc) & (B.crop == crop)].iloc[0]
                sig = (r.p_edd_adj < 0.05) and (r.b_edd < 0)
                errpt(a, k + (c - 1) * 0.22, r.Tstar, r.Tstar_p05, r.Tstar_p95, col, mk, filled=sig)
        a.set_xticks(range(4)); a.set_xticklabels([t[1] for t in trt], fontsize=8.5)
        a.set_xlim(-0.55, 3.45); a.set_ylim(25, 41.5)
        a.set_title(f"({'ab'[j]}) {NAME[crop]}: scale of analysis", loc="left", color=INK)
        tidy(a)
        # noise: distribution of T*, original functions, rainfed
        g = N[(N.crop == crop) & (N.heat == "orig") & (N.k > 0)]
        s = S[(S.crop == crop) & (S.heat == "orig")].sort_values("k")
        kstar = float(np.interp(obs.loc[crop, "r2_within"], s.r2_median.values[::-1], s.k.values[::-1]))
        ks = sorted(g.k.unique())
        b = axs[1, j]
        bottom = np.zeros(len(ks))
        for (lab, fn), col in zip(classes, ccol):
            sh = np.array([fn(g[g.k == k].Tstar.values).mean() for k in ks])
            b.bar(ks, 100 * sh, width=0.36, bottom=100 * bottom, color=col, edgecolor="white", lw=0.8, label=lab)
            bottom += sh
        b.plot([kstar, kstar], [0, 101], color=INK, lw=1.0, ls="--")
        b.text(kstar, 103, "DAPSA-like\nnoise", fontsize=7.5, color=INK, ha="center", va="bottom")
        b.set_xticks(ks); b.set_xlim(0.2, 3.3); b.set_ylim(0, 136); b.set_yticks(range(0, 101, 20))
        b.set_title(f"({'cd'[j]}) {NAME[crop]}: threshold found under yield noise", loc="left", color=INK)
        tidy(b)
        c_ = axs[2, j]
        q = g.groupby("k").warm1C.quantile([0.05, 0.5, 0.95]).unstack()
        k0 = s[s.k == 0].warm1C_median.iloc[0]
        xs = [0.0] + list(q.index)
        c_.fill_between(xs, [k0] + list(q[0.05]), [k0] + list(q[0.95]), color="0.85", lw=0, label="5–95 % of noisy panels")
        c_.plot(xs, [k0] + list(q[0.5]), "o-", color=C_BASE, ms=5, lw=1.5, label="Median estimate")
        c_.axhline(s.true_pct.iloc[0], color=C_TRUE, lw=1.2, ls="--", label="True: +1 °C simulated in DSSAT")
        c_.plot([kstar], [obs.loc[crop, "warm1C_pct"]], marker="*", ms=12, color=INK, ls="none", mec="white",
                label="DAPSA estimate")
        c_.axhline(0, color="0.6", lw=0.8)
        c_.set_xlim(-0.1, 3.1)
        c_.set_xlabel("Noise level (multiple of the DAPSA residuals)")
        c_.set_title(f"({'ef'[j]}) {NAME[crop]}: +1 °C effect under yield noise", loc="left", color=INK)
        tidy(c_)
    axs[0, 0].set_ylabel("Threshold T* (°C)")
    axs[1, 0].set_ylabel("Share of noisy panels (%)")
    axs[2, 0].set_ylabel("Yield change for +1 °C (% of trend)")
    same_scale(axs[2], "y")
    for j in range(2):
        axs[1, j].set_xlabel("Noise level (multiple of the DAPSA residuals)")
    h1 = [Line2D([], [], marker=mk, color=col, ls="none", ms=7, label=lab) for _, lab, col, mk in scl]
    h1 += [Line2D([], [], marker="o", color="0.45", ls="none", ms=7, label="Significant negative EDD coefficient (adj. p < 0.05)"),
           Line2D([], [], marker="o", color="0.45", mfc="white", ls="none", ms=7, label="No significant negative EDD coefficient")]
    for j in range(2):
        axs[1, j].legend(title=None, loc="upper left", ncol=4, fontsize=7.5, frameon=False, handlelength=1.2,
                         columnspacing=0.8, bbox_to_anchor=(0.0, 1.0))
    axs[2, 1].legend(loc="lower left", fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.legend(handles=h1, loc="upper center", ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.savefig(OUT + "Fig8_robustness_scale_and_noise.png", dpi=DPI)


if __name__ == "__main__":
    todo = [int(a) for a in sys.argv[1:]] or [1, 4, 5, 6, 7, 8]
    for n in todo:
        {1: fig1, 4: fig4, 5: fig5, 6: fig6, 7: fig7, 8: fig8}[n]()
        print("figure", n, "saved", flush=True)
