"""
gprof_ir.satrain
================

Provides an interface class for use with the SatRain benchmark dataset.
"""
from contextlib import nullcontext
from pathlib import Path
from typing import Tuple
import warnings

import h5py
import numpy as np
import torch
from torch.nn.functional import interpolate
from pytorch_retrieve import load_model
import xarray as xr

from .retrieval import download_model


class GPROFIRRetrieval:
    """
    Helper class implementing a SatRain retrieval callback for the GPROF IR retrievals.
    """

    def __init__(
        self,
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
        model = download_model(variant=variant, n_steps=n_steps)
        warnings.filterwarnings("ignore", module="torch")
        model = load_model(model).eval()
        model = load_model(model)
        device = torch.device(device)
        use_autocast = dtype in (torch.bfloat16, torch.float16)
        self.dtype = torch.float32 if use_autocast else dtype
        self.autocast_ctx = (
                torch.autocast(device_type=device.type, dtype=dtype) if use_autocast else nullcontext()
        )
        self.model = model.eval().to(device=device, dtype=self.dtype)
        self.device = device
        self.dtype = dtype
        self.n_steps = n_steps

    def __call__(
        self,
        retrieval_input
    ):
        time = retrieval_input.time
        obs_time = retrieval_input.obs_geo_ir_time.data

        obs_stacked = []
        for batch in range(retrieval_input.batch.size):
            time = retrieval_input.time[{"batch": batch}].mean().data
            nearest = np.argmin(np.abs(obs_time - time))
            chans = slice(nearest - self.n_steps, nearest)
            obs_stacked.append(
                 torch.tensor(retrieval_input[{"batch": batch}].obs_geo_ir.data[chans])
            )
        obs = torch.stack(obs_stacked).to(device=self.device, dtype=self.dtype)

        with torch.inference_mode(), self.autocast_ctx:
            pred = interpolate(self.model(obs)[:, :, 0], scale_factor=2.0)
            sp_mean = pred.expected_value().float().cpu().numpy()
            pop = pred.probability_greater_than(1e-3).float().cpu().numpy()
            pflag = 0.5 < pop
            pop_heavy = pred.probability_greater_than(10.3).float().cpu().numpy()
            hpflag = 0.5 < pop_heavy
        return xr.Dataset({
            "surface_precip": (("batch", "latitude", "longitude",), sp_mean),
            "probability_of_precip": (("batch", "latitude", "longitude",), pop),
            "precip_flag": (("batch", "latitude", "longitude",), pflag),
            "probability_of_heavy_precip": (("batch", "latitude", "longitude",), pop_heavy),
            "heavy_precip_flag": (("batch", "latitude", "longitude",), hpflag),
        })


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
