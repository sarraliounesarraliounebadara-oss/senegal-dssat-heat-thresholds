"""One bootstrap output per treatment.

07_heat_models.py seeds one random generator before its loop over crops. A run that holds millet and groundnut
therefore draws the groundnut bootstrap years after the millet ones, while a run that holds groundnut alone draws
them first: the point estimates are identical, the groundnut bootstrap intervals are not. The search-adjusted tests
(24) seed each crop separately and do not depend on this.

The single-crop runs of the factorial experiment (e2/chain/<crop>_<heat>_<water>) are the reference for the main
specification at department scale. This script copies their threshold and warming intervals (and their point
estimates, which are the same) into the rows of X2_summary.csv (controls = main) and X3_summary.csv
(scale = department) that describe the same treatments, and writes *_harmonised.csv files used by the tables,
the figures and the text. The SPEI and simulated-water-stress models, the pixel and mixed scales and the observed
yields are different specifications and keep their own runs.
usage: python3 harmonise_bootstrap.py"""
import numpy as np
import pandas as pd
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402

X = paths.DSSAT + "x2x3/"
E2 = pd.read_csv(paths.DSSAT + "e2/E2_treatments.csv").query("variant == 'base'")
COLS = ["Tstar", "Tstar_p05", "Tstar_p95", "b_edd", "t_edd", "r2_within", "warm1C", "warm1C_p05", "warm1C_p95"]


def e2(crop, heat, water):
    r = E2[(E2.crop == crop) & (E2.heat == heat) & (E2.water == water)]
    assert len(r) == 1, (crop, heat, water)
    return r.iloc[0]


x2 = pd.read_csv(X + "X2_summary.csv")
for i, r in x2.iterrows():
    if r.source in ("orig", "neutral", "minus3") and r.controls == "main":
        ref = e2(r.crop, r.source, "rainfed")
        assert int(ref.Tstar) == int(r.Tstar) and abs(ref.b_edd - r.b_edd) < 1e-3, (r.source, r.crop)
        x2.loc[i, COLS] = ref[COLS].values
x2.to_csv(X + "X2_summary_harmonised.csv", index=False, float_format="%.4f")

x3 = pd.read_csv(X + "X3_summary.csv")
for i, r in x3.iterrows():
    if r.scale == "department":
        heat, water = r.treatment.split("_", 1)
        ref = e2(r.crop, heat, water)
        assert int(ref.Tstar) == int(r.Tstar) and abs(ref.b_edd - r.b_edd) < 1e-3, (r.treatment, r.crop)
        x3.loc[i, COLS] = ref[COLS].values
x3.to_csv(X + "X3_summary_harmonised.csv", index=False, float_format="%.4f")

# report what moved
a = pd.read_csv(X + "X2_summary.csv").merge(x2, on=["source", "controls", "crop"], suffixes=("", "_h"))
d = a[(a.controls == "main") & (a.source != "dapsa")]
print(d[["source", "crop", "Tstar_p05", "Tstar_p05_h", "Tstar_p95", "Tstar_p95_h", "warm1C_p05", "warm1C_p05_h",
         "warm1C_p95", "warm1C_p95_h"]].round(2).to_string(index=False))
