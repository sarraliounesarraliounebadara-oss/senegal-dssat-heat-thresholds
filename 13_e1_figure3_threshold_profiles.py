"""E1 figure: the AFM statistical chain applied to DAPSA and to DSSAT department yields.
Row per crop: (1) threshold profile = gain in within-R2 over the water-only model, (2) EDD coefficient at each
candidate threshold (90 % interval from year-clustered SE), (3) effect of +1 degC (bootstrap 5-95 %)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402

E1 = paths.DSSAT + "e1/"
COL = {"obs": "black", "sim": {"MILLET": "#2166ac", "GROUNDNUT": "#b2182b"}}
LAB = {"obs": "DAPSA (observed)", "sim": "DSSAT (simulated)"}
fig, ax = plt.subplots(2, 3, figsize=(12.5, 7.2), gridspec_kw={"width_ratios": [1.25, 1.25, 0.8]})
for i, crop in enumerate(("MILLET", "GROUNDNUT")):
    name = "Millet" if crop == "MILLET" else "Groundnut"
    for j, v in enumerate(("obs", "sim")):
        col = COL[v] if v == "obs" else COL[v][crop]
        res = pd.read_csv(E1 + f"{v}/work/model_results_main.csv").set_index("crop").loc[crop]
        prof = pd.read_csv(E1 + f"{v}/work/threshold_profiles_main.csv")
        prof = prof[prof.crop == crop]
        gain = 100 * (prof.r2_within - res.r2_within_water_only)
        a = ax[i, 0]
        a.plot(prof.Tstar, gain, color=col, lw=2, marker="o", ms=3.5, label=LAB[v])
        ts = int(res.Tstar)
        a.plot([ts], [gain[prof.Tstar == ts].iloc[0]], marker="o", ms=9, mfc="white", mec=col, mew=2, zorder=5)
        a.annotate(f"T* = {ts} °C", (ts, gain[prof.Tstar == ts].iloc[0]), textcoords="offset points",
                   xytext=(0, 11), ha="center", fontsize=8.5, color="0.2", zorder=6,
                   bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
        sw = pd.read_csv(E1 + f"{v}/work/edd_by_threshold_main.csv")
        sw = sw[sw.crop == crop]
        b = ax[i, 1]
        b.fill_between(sw.Tstar, sw.b_edd - 1.645 * sw.se_edd, sw.b_edd + 1.645 * sw.se_edd, color=col, alpha=0.12, lw=0)
        b.plot(sw.Tstar, sw.b_edd, color=col, lw=2, label=LAB[v])
        b.plot([ts], [sw[sw.Tstar == ts].b_edd.iloc[0]], marker="o", ms=8, mfc="white", mec=col, mew=2, zorder=5)
        c = ax[i, 2]
        c.errorbar([j], [res.warm1C_pct], yerr=[[res.warm1C_pct - res.warm1C_boot_p05], [res.warm1C_boot_p95 - res.warm1C_pct]],
                   fmt="o", color=col, ms=7, capsize=4, lw=1.8)
        c.annotate(f"{res.warm1C_pct:.1f}%", (j, res.warm1C_pct), textcoords="offset points", xytext=(10, -3), fontsize=8.5,
                   color="0.2")
    a = ax[i, 0]
    a.set_title(f"({'a' if i == 0 else 'd'}) {name}: threshold profile", loc="left", fontsize=10.5)
    a.set_ylabel("Within-R² gain over water-only model (points)", fontsize=9)
    a.grid(alpha=0.25); a.set_xlim(25.5, 40.5)
    lo, hi = a.get_ylim(); a.set_ylim(lo, hi + 0.12 * (hi - lo))
    b = ax[i, 1]
    b.axhline(0, color="0.5", lw=0.8)
    b.set_title(f"({'b' if i == 0 else 'e'}) {name}: EDD coefficient by threshold", loc="left", fontsize=10.5)
    b.set_ylabel("Yield change per degree-day above T* (%)", fontsize=9)
    b.grid(alpha=0.25); b.set_xlim(25.5, 40.5)
    b.set_ylim(max(b.get_ylim()[0], -8), 1)
    c = ax[i, 2]
    c.axhline(0, color="0.5", lw=0.8)
    c.set_xticks([0, 1]); c.set_xticklabels(["DAPSA", "DSSAT"])
    c.set_xlim(-0.6, 1.6)
    c.set_title(f"({'c' if i == 0 else 'f'}) {name}: +1 °C", loc="left", fontsize=10.5)
    c.set_ylabel("Yield change (%), bootstrap 5–95 %", fontsize=9)
    c.grid(alpha=0.25, axis="y")
    if i == 1:
        ax[i, 0].set_xlabel("Candidate threshold T* (°C)"); ax[i, 1].set_xlabel("Candidate threshold T* (°C)")
for j in range(3):          # one scale per variable: millet and groundnut rows share each y axis
    lo = min(ax[i, j].get_ylim()[0] for i in range(2)); hi = max(ax[i, j].get_ylim()[1] for i in range(2))
    for i in range(2):
        ax[i, j].set_ylim(lo, hi)
h = [plt.Line2D([], [], color="black", lw=2), plt.Line2D([], [], color="#2166ac", lw=2), plt.Line2D([], [], color="#b2182b", lw=2)]
fig.legend(h, ["DAPSA (observed)", "DSSAT, millet", "DSSAT, groundnut"], loc="upper center", ncol=3, frameon=False, fontsize=9.5)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(E1 + "Fig_E1_thresholds_observed_vs_DSSAT.png", dpi=200)
fig.savefig(paths.DSSAT + "figures_article/Fig3_threshold_profiles.png", dpi=1000)
print("saved")
