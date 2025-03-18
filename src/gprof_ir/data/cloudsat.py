"""
gprof_ir.data.cloudsat
======================

Provides an interface to extract combined CloudSat/ERA5 reference data.
"""
from datetime import datetime
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from pansat import Geometry, FileRecord
from pansat.time import to_datetime, TimeRange
from pansat.products.satellite.cloudsat import l2c_precip_column, l2c_rain_profile, l2c_snow_profile
from scipy.stats import binned_statistic_dd
import xarray as xr

from chimp.areas import Area
from chimp.data.reference import ReferenceDataset, RetrievalTarget
from chimp.data.utils import get_output_filename, records_to_paths


LOGGER = logging.getLogger(__name__)


def get_era5_files(path: Path, start_time: datetime, end_time: datetime) -> List[Path]:
    """
    Get paths of ERA5 files to load for given time period.
    """
    year_start = start_time.year
    month_start = start_time.month
    day_start = start_time.day
    year_end = end_time.year
    month_end = end_time.month
    day_end = end_time.day

    file_start = (
        path/
        f"{year_start:04}{month_start:02}" /
        f"ERA5_{year_start:04}{month_start:02}{day_start:02}_surf.nc"
    )
    file_end = (
        path/
        f"{year_end:04}{month_end:02}" /
        f"ERA5_{year_end:04}{month_end:02}{day_end:02}_surf.nc"
    )
    if file_start == file_end:
        return [file_start]

    return [file_start, file_end]




class LightPrecip(ReferenceDataset):
    """
    Represents retrieval reference data derived from CloudSat and ERA5 data.
    """
    def __init__(self):
        super().__init__("light_precip", 8, [RetrievalTarget("light_precip")], quality_index=None)
        self.products = [l2c_precip_column, l2c_snow_profile, l2c_rain_profile]
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
            path: Should point to a local folder cotaining ERA5 data.

        Return:
            A list of locally available files to extract CHIMP training data from.
        """
        precip_column_recs = l2c_precip_column.find_files(TimeRange(start_time, end_time))
        recs_combined = []
        for rec in precip_column_recs:
            rec_snow_profile = l2c_snow_profile.find_files(rec.central_time)
            if len(rec_snow_profile) == 0:
                LOGGER.warning(
                    "No matching L2C Snow-Profile record for %s.",
                    rec.filename
                )
                continue
            rec_rain_profile = l2c_rain_profile.find_files(rec.central_time)
            if len(rec_rain_profile) == 0:
                LOGGER.warningg(
                    "No matching L2C Rain-Profile record for %s.",
                    rec.filename
                )
                continue
            start_time = to_datetime(rec.temporal_coverage.start)
            end_time = to_datetime(rec.temporal_coverage.end)
            era5_files = get_era5_files(path, start_time, end_time)
            recs_combined.append([rec, rec_snow_profile[0], rec_rain_profile[0]] + era5_files)
        return recs_combined


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

        pc_data = l2c_precip_column.open(path[0])
        sp_data = l2c_snow_profile.open(path[1])
        rp_data = l2c_rain_profile.open(path[2])

        pflag = pc_data.precip_flag.data
        surface_precip = rp_data["surface_precip"].data
        surface_precip_snow = sp_data["surface_precip"].data

        total_precip = np.nan * np.zeros_like(surface_precip)
        total_precip[pflag == 0] = 0.0

        rain_flag = (0 < pflag) * (pflag < 4) * (surface_precip >= 0.0)
        total_precip[rain_flag] = surface_precip[rain_flag]
        snow_flag = (4 < pflag) * (pflag < 9)
        total_precip[snow_flag] = surface_precip_snow[snow_flag]

        mask = (surface_precip < 0) * (0 < pflag) * (pflag < 4)
        lons = pc_data.longitude.data[mask]
        lats = pc_data.latitude.data[mask]
        time = pc_data.time.data[mask]
        lons[lons < 0] += 360

        if len(lons) > 0:
            era5_precip = []
            for era5_path in path[3:]:
                with xr.open_dataset(era5_path) as data:
                    era5_precip.append(data.tp.compute())
            era5_precip = xr.concat(era5_precip, "time")
            coords = xr.Dataset({
                "time": (("samples",), time),
                "latitude": (("samples",), lats),
                "longitude": (("samples",), lons)
            })
            era5_precip = 1e3 * era5_precip.interp(
                time=coords.time,
                longitude=coords.longitude,
                latitude=coords.latitude
            )
            total_precip[mask] = era5_precip.data

        lons = pc_data.longitude.data
        lats = pc_data.latitude.data
        time = pc_data.time.data

        # Need to expand scan time to full dimensions.
        valid_times = time[np.isfinite(time)]
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
        sample = [time[valid].astype("datetime64[s]").astype(np.float32), lats[valid], lons[valid]]
        sp_r = binned_statistic_dd(sample, total_precip[valid], bins=bins)[0]
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
                "light_precip": {"dtype": "float32", "zlib": True},
                "longitude": {"dtype": "float32", "zlib": True},
                "latitude": {"dtype": "float32", "zlib": True},
                "rqi": {"dtype": "int8", "zlib": True, "_FillValue": -1},
            }

            height, width = rqi.shape
            rqi = rqi[:-1].reshape(height // 4, 4, width // 4, 4)
            rqi = rqi.any(axis=1).any(axis=2)

            data = xr.Dataset({
                "latitude": (("latitude"), lats_g),
                "longitude": (("longitude"), lons_g),
                "light_precip": (("latitude", "longitude"), precip),
                "rqi": (("latitude_ds", "longitude_ds"), rqi),
            })
            filename = get_output_filename(
                self.name, time, time_step
            )
            output_file = output_folder / filename
            if output_file.exists():
                existing = xr.load_dataset(output_file)
                sp = existing.light_precip.data
                sp_mask = np.isfinite(data.light_precip.data)
                sp[sp_mask] = data.light_precip.data[sp_mask]
                rqi = existing.rqi.data
                rqi_mask = 0.5 < data.rqi.data
                rqi[rqi_mask] = 1.0
                data.to_netcdf(output_file, encoding=encoding)
            else:
                LOGGER.info(
                        "Writing training samples to %s.",
                        output_folder / filename
                )
                data.to_netcdf(output_folder / filename, encoding=encoding)

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
                qi = 0.5 * (data.rqi.data[1:] + data.rqi.data[:-1])

                rows, cols = np.where(qi > 0.5)
                n_rows, n_cols = qi.shape
                valid_rows = (rows > scene_size[0] // 2) * (rows < n_rows - scene_size[0] // 2) * (rows % multiple == 0)
                valid_cols = (cols > scene_size[1] // 2) * (cols < n_cols - scene_size[1] // 2) * (cols % multiple == 0)
                rows = rows[valid_rows * valid_cols]
                cols = cols[valid_rows * valid_cols]

                found = False
                count = 0
                while not found:
                    if count > 20:
                        return None
                    ind = rng.choice(rows.size)
                    i_cntr = rows[ind]
                    j_cntr = cols[ind]

                    i_start = i_cntr - scene_size[0] // 2
                    i_end = i_start + scene_size[0]
                    j_start = j_cntr - scene_size[1] // 2
                    j_end = j_start + scene_size[1]
                    row_slice = slice(i_start, i_end)
                    col_slice = slice(j_start, j_end)

                    if (qi[row_slice, col_slice] > quality_threshold).sum() > 1:
                        found = True
                    count += 1

            return (4 * i_start, 4 * i_end, 4 * j_start, 4 * j_end)
        except Exception as exc:
            raise exc
            return None


light_precip = LightPrecip()
