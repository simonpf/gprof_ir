"""
gprof_ir.imerg
==============

This module provides utility functions to load IMERG precipitation estimates that serve as a baseline
for validating GPROF-IR precipitation estimates.
"""
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
import xarray as xr


def load_imerg_data(path: Path, bounds: Optional[Tuple[float, float, float, float]] = None) -> xr.Dataset:
    """
    Load IMERG precipitation.

    Args:
        path: A Path object pointing to the IMERG file to load.
        bounds: A tuple ``(lon_min, lat_min, lon_max, lat_max)`` defining a bounding box for the data to load.

    Return:
    """
    with h5py.File(path) as data:
        if bounds is None:
            lon_slc = slice(0, None)
            lat_slc = slice(0, None)
        else:
            lon_min, lat_min, lon_max, lat_max = bounds
            lons = data["Grid/lon"][:]
            lats = data["Grid/lat"][:]
            lon_mask = np.where((lon_min <= lons) * (lons <= lon_max))[0]
            lon_start, lon_end = lon_mask.min(), lon_mask.max()
            lon_slc = slice(lon_start, lon_end)

            lat_mask = np.where((lat_min <= lats) * (lats <= lat_max))[0]
            lat_start, lat_end = lat_mask.min(), lat_mask.max()
            lat_slc = slice(lat_start, lat_end)

        surface_precip = data["Grid/precipitation"][0, lon_slc, lat_slc]
        surface_precip[surface_precip < 0] = np.nan
        surface_precip_mw = data["Grid/Intermediate/MWprecipitation"][0, lon_slc, lat_slc]
        surface_precip_mw[surface_precip < 0] = np.nan
        surface_precip_ir = data["Grid/Intermediate/IRprecipitation"][0, lon_slc, lat_slc]
        surface_precip_ir[surface_precip < 0] = np.nan
        prob_liquid = data["Grid/probabilityLiquidPrecipitation"][0, lon_slc, lat_slc]
        mw_precip_source = data["Grid/Intermediate/MWprecipSource"][0, lon_slc, lat_slc]
        lons = data["Grid/lon"][lon_slc]
        lats = data["Grid/lat"][lat_slc]

    return xr.Dataset({
        "longitude": (("longitude"), lons),
        "latitude": (("latitude"), lats),
        "surface_precip": (("longitude", "latitude"), surface_precip),
        "surface_precip_mw": (("longitude", "latitude"), surface_precip_mw),
        "surface_precip_ir": (("longitude", "latitude"), surface_precip_ir),
        "probability_of_liquid_precip": (("longitude", "latitude"), prob_liquid / 100.0),
        "mw_sensor": (("longitude", "latitude"), mw_precip_source)
    })
