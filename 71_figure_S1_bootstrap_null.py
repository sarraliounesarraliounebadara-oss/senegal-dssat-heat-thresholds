"""Figure S1: null distribution of the search-adjusted EDD test, groundnut, plant thresholds -3 degC,
no water stress. Reads the draws saved by check_gn_minus3_ns.py (9,999 restricted wild cluster samples).
Usage: python3 figS1_bootstrap_null.py <gn_minus3_ns_bootstrap.npz> <output.png>"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
z = np.load(sys.argv[1])
t, T, to = np.abs(z["t_b"]), z["T_b"], abs(float(z["t_obs"]))
p = (1 + (t >= to).sum()) / (len(t) + 1)
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
ax[0].hist(t, bins=np.arange(0, 16.5, 0.25), color="#9bb7d4", edgecolor="white", linewidth=0.3)
ax[0].axvline(to, color="#c0392b", lw=1.2)
ax[0].text(to + 0.3, ax[0].get_ylim()[1] * 0.95, f"observed |t| = {to:.2f}\np = {p:.3f}", color="#c0392b",
           va="top", fontsize=7.5)
ax[0].set_xlabel("|t| of the EDD coefficient at the re-selected threshold")
ax[0].set_ylabel("Bootstrap samples"); ax[0].set_title("(a)", loc="left"); ax[0].set_xlim(0, 16)
grid = np.arange(26, 41); big = t >= to
ax[1].bar(grid - 0.2, [np.mean(T == g) for g in grid], 0.4, color="#9bb7d4", label="All samples")
ax[1].bar(grid + 0.2, [np.mean(T[big] == g) for g in grid], 0.4, color="#c0392b",
          label=f"Samples with |t| ≥ {to:.2f}")
ax[1].set_xlabel("Threshold selected in the sample (°C)"); ax[1].set_ylabel("Share of samples")
ax[1].legend(frameon=False); ax[1].set_xticks(range(26, 41, 2)); ax[1].set_title("(b)", loc="left")
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=600)
