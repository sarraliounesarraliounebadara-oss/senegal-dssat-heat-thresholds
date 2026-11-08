"""Read all supplied ANACIM segments without a merge-period cutoff or imputation.
The analysis period is separate from the input merge. Crop windows may cross years.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

STUDY_PERIOD = (1986, 2016)     # yields and climate; same period as 01_detrend_dapsa.py


def open_climate(data_dir):
    root = Path(data_dir)
    def temperature(which):
        directory = root / "chirts"
        single = directory / f"CHIRTS_{which}_merged.nc"
        pieces = sorted(directory.glob(f"CHIRTS_{which}_merged_*.nc"))
        paths = pieces if pieces else ([single] if single.exists() else [])
        if not paths:
            raise FileNotFoundError(f"No {which} files in {directory}")
        if len(paths) == 1:
            ds = xr.open_dataset(paths[0])
        else:
            ds = xr.open_mfdataset(paths, combine="by_coords", join="exact", chunks={"time": 366},
                                  data_vars="minimal", coords="minimal", compat="equals", combine_attrs="drop")
        da = ds.temp.transpose("time", "Lat", "Lon")
        index = da.indexes["time"]
        if not index.is_unique or not index.is_monotonic_increasing:
            raise ValueError(f"Duplicate or unordered {which} dates; resolve overlaps explicitly")
        return da
    tx, tn = temperature("Tmax"), temperature("Tmin")
    pr = xr.open_dataset(f"{data_dir}/arc2/ARC2_merged.nc").precip
    for da in (tn, pr):
        for coordinate in ("Lat", "Lon"):
            if not np.array_equal(tx[coordinate].values, da[coordinate].values):
                raise ValueError(f"Climate coordinate mismatch: {coordinate}")
    return tx, tn, pr


def read_period(da, start, end):
    """Explicitly complete requested calendar; no silent truncation or interpolation."""
    full = pd.date_range(start, end, freq="D")
    selected = da.sel(time=slice(start, end))
    index = pd.DatetimeIndex(selected.time.values)
    missing = full.difference(index)
    if len(missing) or not index.is_unique:
        raise ValueError(f"Incomplete requested window {start} to {end}: {len(missing)} missing dates")
    x = selected.astype("float32").load().reindex(time=full).values
    if np.isnan(x).all(axis=(1, 2)).any():
        raise ValueError(f"An entire climate day is empty in {start} to {end}")
    return full, x
