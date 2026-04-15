"""
gprof_ir.satrain
================

Provides an interface class for use with the SatRain benchmark dataset.
"""
from contextlib import nullcontext
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import h5py
import numpy as np
import torch
from pytorch_retrieve import load_model
from scipy.ndimage import binary_closing
from torch.nn.functional import interpolate
import xarray as xr

from .retrieval import download_model, run_retrieval_multi


def get_date(path: Path) -> np.datetime64:
    """
    Extract date from merged_ir file.
    """
    time = np.datetime64(datetime.strptime(path.stem, "merg_%Y%m%d%H_4km-pixel"))
    return time


class InputLoader:
    def __init__(
            self,
            input_files: Dict[np.datetime64, Path],
            input_times: List[np.datetime64],
            roi: List[float],
            n_steps: int
    ):
        self.input_files = input_files
        self.input_times = input_times
        self.roi = roi
        self.n_steps = n_steps

    def __len__(self) -> int:
        return len(self.input_times)


    @cached_property
    def get_slices(self):
        """
        Get slices

        """
        path = next(iter(self.input_files.values()))
        with xr.open_dataset(path) as data:
            lats = data.lat.data
            lons = data.lon.data
        lon_min, lat_min, lon_max, lat_max = self.roi
        lat_mask = (lat_min <= lats) * (lats <= lat_max)
        lon_mask = (lon_min <= lons) * (lons <= lon_max)
        lat_inds = np.where(lat_mask)[0]
        lon_inds = np.where(lon_mask)[0]

        lat_c = int(0.5 * (lat_inds[0] + lat_inds[-1]))
        lat_start = max(lat_c - 256, 0)
        lat_end = lat_end = lat_start + 512

        lon_c = int(0.5 * (lon_inds[0] + lon_inds[-1]))
        lon_start = max(lon_c - 256, 0)
        lon_end = lon_end = lon_start + 512

        return {
            "lat": slice(lat_start, lat_end),
            "lon": slice(lon_start, lon_end),
        }



    def __getitem__(self, ind):

        time = self.input_times[ind]
        input_times = [
            time - np.timedelta64(60, "m") * ((step + 1 - self.n_steps) // 2) for step in range(self.n_steps)
        ]
        input_times = set(input_times)

        input_data = []
        for input_time in input_times:
            with xr.open_dataset(self.input_files[input_time]) as data:
                data = data[self.get_slices()].load()
                input_data.append(data)

        input_data = xr.concat(input_data, dim="time")
        inpt = {
            "merged_ir": input_data.Tb.data[-self.n_steps:]
        }

        # Calculate invalid input mask
        valid = np.isfinite(input_data.Tb.data)
        elem = np.ones((1, 8, 8))
        valid = binary_closing(valid, elem, border_value=1)

        lats = input_data.lat.data
        lats = 0.5 * (lats[0::2] + lats[1::2])
        lons = input_data.lon.data
        lons = 0.5 * (lons[0::2] + lons[1::2])

        aux = {
            "latitude": lats,
            "longitude": lons,
            "time": input_data.time.data,
            "valid_input": valid[..., ::2, ::2],
            "n_steps": self.n_steps,
            "variant": self.variant
        }
        date_str = self.input_files[input_time].stem.split("_")[1]
        return inpt, aux, f"gprof_ir_{date_str}.nc"





class GPROFIRRetrieval:
    """
    Helper class implementing a SatRain retrieval callback for the GPROF IR retrievals.
    """

    def __init__(
        self,
        merged_ir_path: Path,
        variant: str = "gmi",
        n_steps: int = 3,
        device: str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Args:
            variant: The variant of the GPROF-IR retrieval ('gmi' or 'cmb').
            n_steps: The number of input timesteps (1, 3, or 5)
            device: The device to use for inference
            dtype: The dtype to use for inference.
        """
        if not isinstance(merged_ir_path, (list, tuple)):
            merged_ir_path = [merged_ir_path]

        self.input_files = {}
        for path in merged_ir_path:
            ir_files = Path(path).glob("**/merg_*.nc4")
            for ir_file in ir_files:
                date = get_date(ir_file).astype("datetime64[s]")
                self.input_files[date] = ir_file

        self.variant = variant
        self.n_steps = n_steps
        self.device = device
        self.dtype = dtype


    def __call__(
        self,
        retrieval_input
    ):
        time = retrieval_input.time.data
        valid = np.isfinite(time)
        time_min = time[valid].min().astype("datetime64[s]")
        time_max = time[valid].max().astype("datetime64[s]")
        time = retrieval_input.time.fillna(time_min)

        lats = retrieval_input.latitude.data
        lons = retrieval_input.longitude.data
        lat_min, lat_max = lats.min() - 1.0, lats.max() + 1.0
        lon_min, lon_max = lons.min() - 1.0, lons.max() + 1.0

        time_start = (time_min - np.timedelta64(60, "m")).astype("datetime64[h]").astype("datetime64[s]")
        time_end = (time_max + np.timedelta64(30, "m")).astype("datetime64[h]").astype("datetime64[s]")
        input_times = np.arange(time_start, time_end + np.timedelta64(30, "m"), np.timedelta64(1, "h"))

        input_files = [self.input_files[input_time.astype("datetime64[s]")] for input_time in input_times]
        results = []
        for input_file in input_files:
            results.append(
                run_retrieval_multi(
                    input_file,
                    device=self.device,
                    dtype=self.dtype,
                    variant=self.variant,
                    n_steps=self.n_steps,
                    roi=(lon_min, lat_min, lon_max, lat_max),
                    progress=False
                )[0]
            )
        results = xr.concat(results, dim="time")

        results = results.interp(
            latitude=retrieval_input.latitude,
            longitude=retrieval_input.longitude,
            time=time,
        )
        return results[["surface_precip"]]



def load_imerg_precip(path: Path, bounds: Tuple[float, float, float, float]) -> xr.Dataset:
    """
    Load IMERG precipitation.

    Args:
        path: A Path object pointing to the IMERG file to load.
        bounds: A tuple ``(lon_min, lat_min, lon_max, lat_max)`` defining a bounding box for the data to load.

    Return:
    """
    lon_min, lat_min, lon_max, lat_max = bounds
    with h5py.File(path) as data:
        lons = data["Grid/lon"][:]
        lats = data["Grid/lat"][:]
        lon_mask = np.where((lon_min <= lons) * (lons <= lon_max))[0]
        lon_start, lon_end = lon_mask.min(), lon_mask.max()
        lon_slc = slice(lon_start, lon_end)

        lat_mask = np.where((lat_min <= lats) * (lats <= lat_max))[0]
        lat_start, lat_end = lat_mask.min(), lat_mask.max()
        lat_slc = slice(lat_start, lat_end)

        ir_influence = data["Grid/Intermediate/IRinfluence"][0, lon_slc, lat_slc]
        mw_precip_source = data["Grid/Intermediate/MWprecipSource"][0, lon_slc, lat_slc]
        mw_observation_time = data["Grid/Intermediate/MWobservationTime"][0, lon_slc, lat_slc]
        mw_precipitation = data["Grid/Intermediate/MWprecipitation"][0, lon_slc, lat_slc]
        prob_liquid = data["Grid/probabilityLiquidPrecipitation"][0, lon_slc, lat_slc]
        ir_precipitation = data["Grid/Intermediate/IRprecipitation"][0, lon_slc, lat_slc]

        surface_precip = data["Grid/precipitation"][0, lon_slc, lat_slc]
        surface_precip[surface_precip < 0] = np.nan
        precip_flag = surface_precip > 1e-2
        heavy_precip_flag = surface_precip > 10.0
        lons = lons[lon_start:lon_end]
        lats = lats[lat_start:lat_end]
        time = np.datetime64("1980-01-06") + data["Grid/time"][:].astype("timedelta64[s]")

    return xr.Dataset({
        "longitude": (("longitude"), lons),
        "latitude": (("latitude"), lats),
        "surface_precip": (("longitude", "latitude"), surface_precip),
        "precip_flag": (("longitude", "latitude"), precip_flag.astype(np.int8)),
        "heavy_precip_flag": (("longitude", "latitude"), precip_flag.astype(np.int8)),
        "ir_influence": (("longitude", "latitude"), ir_influence / 100.0),
        "mw_precip_source": (("longitude", "latitude"), mw_precip_source),
        "mw_observation_time": (("longitude", "latitude"), mw_precip_source),
        "mw_precipitation": (("longitude", "latitude"), mw_precipitation),
        "ir_precipitation": (("longitude", "latitude"), ir_precipitation),
        "probability_of_liquid_precip": (("longitude", "latitude"), prob_liquid / 100.0),
        "time": time.astype("datetime64[ns]")
    })


def load_imerg_precip_mw(path: Path, bounds: Tuple[float, float, float, float]) -> xr.Dataset:
    """
    Load IMERG precipitation.

    Args:
        path: A Path object pointing to the IMERG file to load.
        bounds: A tuple ``(lon_min, lat_min, lon_max, lat_max)`` defining a bounding box for the data to load.

    Return:
    """
    lon_min, lat_min, lon_max, lat_max = bounds
    with h5py.File(path) as data:
        lons = data["Grid/lon"][:]
        lats = data["Grid/lat"][:]
        lon_mask = np.where((lon_min <= lons) * (lons <= lon_max))[0]
        lon_start, lon_end = lon_mask.min(), lon_mask.max()
        lon_slc = slice(lon_start, lon_end)

        lat_mask = np.where((lat_min <= lats) * (lats <= lat_max))[0]
        lat_start, lat_end = lat_mask.min(), lat_mask.max()
        lat_slc = slice(lat_start, lat_end)

        surface_precip = data["Grid/Intermediate/MWprecipitation"][0, lon_slc, lat_slc]
        surface_precip[surface_precip < 0] = np.nan
        precip_flag = surface_precip > 1e-2
        heavy_precip_flag = surface_precip > 10.0
        lons = lons[lon_start:lon_end]
        lats = lats[lat_start:lat_end]
        time = np.datetime64("1980-01-06") + data["Grid/time"][:].astype("timedelta64[s]")

    return xr.Dataset({
        "longitude": (("longitude"), lons),
        "latitude": (("latitude"), lats),
        "surface_precip": (("longitude", "latitude"), surface_precip),
        "precip_flag": (("longitude", "latitude"), precip_flag.astype(np.int8)),
        "heavy_precip_flag": (("longitude", "latitude"), precip_flag.astype(np.int8)),
        "time": time.astype("datetime64[ns]")
    })


def load_imerg_precip_ir(path: Path, bounds: Tuple[float, float, float, float]) -> xr.Dataset:
    """
    Load IMERG precipitation.

    Args:
        path: A Path object pointing to the IMERG file to load.
        bounds: A tuple ``(lon_min, lat_min, lon_max, lat_max)`` defining a bounding box for the data to load.

    Return:
    """
    lon_min, lat_min, lon_max, lat_max = bounds
    with h5py.File(path) as data:
        lons = data["Grid/lon"][:]
        lats = data["Grid/lat"][:]
        lon_mask = np.where((lon_min <= lons) * (lons <= lon_max))[0]
        lon_start, lon_end = lon_mask.min(), lon_mask.max()
        lon_slc = slice(lon_start, lon_end)

        lat_mask = np.where((lat_min <= lats) * (lats <= lat_max))[0]
        lat_start, lat_end = lat_mask.min(), lat_mask.max()
        lat_slc = slice(lat_start, lat_end)

        surface_precip = data["Grid/Intermediate/IRprecipitation"][0, lon_slc, lat_slc]
        surface_precip[surface_precip < 0] = np.nan
        precip_flag = surface_precip > 1e-2
        heavy_precip_flag = surface_precip > 10.0
        lons = lons[lon_start:lon_end]
        lats = lats[lat_start:lat_end]
        time = np.datetime64("1980-01-06") + data["Grid/time"][:].astype("timedelta64[s]")

    return xr.Dataset({
        "longitude": (("longitude"), lons),
        "latitude": (("latitude"), lats),
        "surface_precip": (("longitude", "latitude"), surface_precip),
        "precip_flag": (("longitude", "latitude"), precip_flag.astype(np.int8)),
        "heavy_precip_flag": (("longitude", "latitude"), precip_flag.astype(np.int8)),
        "time": time.astype("datetime64[ns]")
    })
