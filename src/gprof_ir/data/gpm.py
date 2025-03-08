"""
gprof_ir.data.gpm
=================

Provide reference data from the combined radar and passive microwave retrievals from the GPM core observatory.

"""
from datetime import datetime, timedelta
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict,  List, Optional, Tuple, Union

import numpy as np
from scipy.ndimage import gaussian_filter1d
import pansat
from pansat import Geometry, FileRecord
from pansat.granule import merge_granules
from pansat.time import TimeRange
from pansat.products.satellite.gpm import (
    l2b_gpm_cmb,
    l2b_gpm_cmb_b,
    l2b_gpm_cmb_c,
)
from pyresample import AreaDefinition
from scipy.stats import binned_statistic_dd
import torch
import xarray as xr

from chimp.areas import Area
from chimp.data.input import InputDataset
from chimp.data.resample import resample_and_split
from chimp.data.reference import ReferenceDataset, RetrievalTarget
from chimp.data.mrms import MRMS_PRECIP_RATE
from chimp.data.utils import get_output_filename, records_to_paths
from chimp.utils import get_date


LOGGER = logging.getLogger(__name__)


class GPMCMB(ReferenceDataset):
    """
    Represents retrieval reference data derived from GPM combined radar-radiometer retrievals.
    """
    def __init__(self):
        super().__init__("gpm", 8, [RetrievalTarget("surface_precip")], quality_index=None)
        self.products = [l2b_gpm_cmb, l2b_gpm_cmb_b, l2b_gpm_cmb_c]
        self.scale = 8

    def find_files(
            self,
            start_time: np.datetime64,
            end_time: np.datetime64,
            time_step: np.timedelta64,
            roi: Optional[Geometry] = None,
            path: Optional[Path] = None
    ) -> List[FileRecord]:
        """
        Find input data files within a given file range from which to extract
        training data.

        Args:
            start_time: Start time of the time range.
            end_time: End time of the time range.
            time_step: The time step of the retrieval.
            roi: An optional geometry object describing the region of interest
                that can be used to restriced the file selection a priori.
            path: If provided, files should be restricted to those available from
                the given path.

        Return:
            A list of locally available files to extract CHIMP training data from.
        """
        if path is not None:
            path = Path(path)
            all_files = sorted(list(path.glob("**/*.HDF5")))
            matching = []
            for prod in self.products:
                matching += [path for path in all_files if prod.matches(path)]
            return matching

        recs = []
        for prod in self.products:
            recs += prod.find_files(TimeRange(start_time, end_time))
        return recs


    def find_random_scene(
        self,
        path: Path,
        rng: np.random.Generator,
        multiple: int = 4,
        scene_size: int = 256,
        quality_threshold: float = 0.8,
        valid_fraction: float = 0.2,
    ) -> Tuple[int, int, int, int]:
        """
        Finds a random scene in the reference data that has given minimum
        fraction of values of values exceeding a given RQI threshold.

        Args:
            path: The path of the reference data file from which to sample a random
                scene.
            rng: A numpy random generator instance to randomize the scene search.
            multiple: Limits the scene position to coordinates that a multiples
                of the given value.
            quality_threshold: If the reference dataset has a quality index,
                all reference data pixels below the given threshold will considered
                invalid outputs.
            valid_fraction: The minimum amount of valid samples in the extracted
                region.

        Return:
            A tuple ``(i_start, i_end, j_start, j_end)`` defining the position
            of the random crop. Or 'None' if no such tuple could be found.
        """
        multiple = max(multiple // 4, 1)
        scene_size = scene_size // 4

        if isinstance(scene_size, (int, float)):
            scene_size = (int(scene_size),) * 2

        try:
            with xr.open_dataset(path) as data:
                qi = data.rqi.data

                rows, cols = np.where(qi > 0.5)
                n_rows, n_cols = qi.shape
                valid_rows = (rows > scene_size[0] // 2) * (rows < n_rows - scene_size[0] // 2) * (rows % multiple == 0)
                valid_cols = (cols > scene_size[1] // 2) * (cols < n_cols - scene_size[1] // 2) * (cols % multiple == 0)
                rows = rows[valid_rows * valid_cols]
                cols = cols[valid_cols * valid_cols]

                found = False
                count = 0
                while not found:
                    if count > 20:
                        return None
                    ind = np.random.choice(rows.size)
                    i_cntr = rows[ind]
                    j_cntr = cols[ind]

                    i_start = i_cntr - scene_size[0] // 2
                    i_end = i_start + scene_size[0]
                    j_start = j_cntr - scene_size[0] // 2
                    j_end = j_start + scene_size[0]
                    row_slice = slice(i_start, i_end)
                    col_slice = slice(j_start, j_end)

                    if (qi[row_slice, col_slice] > quality_threshold).mean() > 0.01:
                        found = True
                    count += 1

            return (4 * i_start, 4 * i_end, 4 * j_start, 4 * j_end)
        except Exception:
            return None


    def process_file(
            self,
            path: Path,
            domain: Area,
            output_folder: Path,
            time_step: np.timedelta64
    ):
        """
        Extract training samples from a given source file.

        Args:
           path: A Path object pointing to the file to process.
           domain: An area object defining the training domain.
           output_folder: A path pointing to the folder to which to write
               the extracted training data.
           time_step: A timedelta object defining the retrieval time step.
        """
        path = records_to_paths(path)
        output_folder = Path(output_folder) / self.name
        output_folder.mkdir(exist_ok=True)

        input_data = self.products[0].open(path)
        data = input_data.rename({
            "scan_time": "time",
            "estim_surf_precip_tot_rate": "surface_precip"
        })[["time", "surface_precip"]]

        surface_precip = data.surface_precip.data
        surface_precip[surface_precip < 0] = np.nan
        lons = data.longitude.data
        lats = data.latitude.data

        # Need to expand scan time to full dimensions.
        time, _ = xr.broadcast(data.time, data.surface_precip)
        valid_times = time.data[np.isfinite(time.data)]
        min_time = valid_times.min()
        max_time = valid_times.max()

        start_of_day = min_time.astype("datetime64[D]").astype(min_time.dtype)
        min_time_delta = (min_time - start_of_day).astype(time_step.dtype).astype("uint64") // time_step.astype("uint64")
        min_time_r = start_of_day + min_time_delta * time_step
        time_bins = np.arange(min_time_r - time_step / 2.0, max_time + time_step, time_step)
        lons_g, lats_g = domain[self.scale].get_lonlats()

        if not np.isclose(lons_g[0], lons_g[-1]).all():
            raise ValueError(
                "This reference dataset expects a regular lat/lon grid."
            )
        lons_g = lons_g[0]
        lats_g = lats_g[:, 0]

        lons_gg = np.zeros(lons_g.size + 1)
        lons_gg[1:-1] = 0.5 * (lons_g[1:] + lons_g[:-1])
        lons_gg[0] = lons_g[0] - 0.5 * (lons_g[1] - lons_g[0])
        lons_gg[-1] = lons_g[-1] + 0.5 * (lons_g[-1] - lons_g[-2])
        lats_gg = np.zeros(lats_g.size + 1)
        lats_gg[1:-1] = 0.5 * (lats_g[1:] + lats_g[:-1])
        lats_gg[0] = lats_g[0] - 0.5 * (lats_g[1] - lats_g[0])
        lats_gg[-1] = lats_g[-1] + 0.5 * (lats_g[-1] - lats_g[-2])

        valid = (np.isfinite(lons) * np.isfinite(lats) * (0.0 <= surface_precip))
        bins = [time_bins.astype("datetime64[s]").astype(np.float32), lats_gg[::-1], lons_gg]
        sample = [time.data[valid].astype("datetime64[s]").astype(np.float32), lats[valid], lons[valid]]
        sp_r = binned_statistic_dd(sample, surface_precip[valid], bins=bins)[0]
        sp_r = np.flip(sp_r, axis=1)

        for t_ind in range(time_bins.size - 1):
            time = time_bins[t_ind] + 0.5 * (time_bins[t_ind + 1] - time_bins[t_ind])

            precip = sp_r[t_ind]
            rqi = np.isfinite(precip).astype(np.float32)

            if np.isfinite(precip).sum() < 10:
                LOGGER.info(
                    "Less than 100 valid pixels in training sample @ %s.",
                    time
                )
                continue

            encoding = {
                "surface_precip": {"dtype": "float32", "zlib": True},
                "longitude": {"dtype": "float32", "zlib": True},
                "latitude": {"dtype": "float32", "zlib": True},
                "rqi": {"dtype": "int8", "zlib": True, "_FillValue": -1},
            }

            rqi = rqi[::4, ::4]

            data = xr.Dataset({
                "latitude": (("latitude"), lats_g),
                "longitude": (("longitude"), lons_g),
                "surface_precip": (("latitude", "longitude"), precip),
                "rqi": (("latitude_ds", "longitude_ds"), rqi),
            })
            filename = get_output_filename(
                self.name, time, time_step
            )
            output_file = output_folder / filename
            if output_file.exists():
                existing = xr.load_dataset(output_file)
                sp = existing.surface_precip.data
                sp_mask = np.isfinite(data.surface_precip.data)
                sp[sp_mask] = data.surface_precip.data[sp_mask]
                rqi = existing.rqi.data
                rqi_mask = 0.5 < data.rqi.data
                rqi[rqi_mask] = 1.0
                data.to_netcdf(output_file)
            else:
                LOGGER.info(
                        "Writing training samples to %s.",
                        output_folder / filename
                )
                data.to_netcdf(output_folder / filename, encoding=encoding)


gpm = GPMCMB()
