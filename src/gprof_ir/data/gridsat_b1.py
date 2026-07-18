"""
gprof_ir.data.gridsat_b1
========================

This module provides a CHIMP input dataset for the GridSat B1 geostationary observations.
"""
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from pansat.products.satellite.ncei.gridsat import gridsat_b1
from pansat.time import to_datetime64, TimeRange
from pansat.geometry import Geometry
from pyresample import geometry, kd_tree, create_area_def
import xarray as xr

from chimp.areas import Area
from chimp.data.input import InputDataset
from chimp.data.resample import resample_and_split
from chimp.data.utils import get_output_filename, records_to_paths
from chimp.utils import get_date


LOGGER = logging.getLogger(__name__)


class GridSatB1(InputDataset):
    """
    Represents input data derived from merged IR data.
    """
    def __init__(
            self,
            stack: Optional[int] = None
    ):
        name = "gridsat_b1"
        if stack is not None:
            name = f"gridsat_b1_{stack}"
        InputDataset.__init__(
            self,
            name,
            "gridsat_b1",
            8,
            "tbs",
            n_dim=2,
            spatial_dims=("latitude", "longitude")
        )
        self.stack = stack
        self.stacking_method = "before"
        self.stack_drop = 0.2
        self.scale = 4

    @property
    def n_channels(self) -> int:
        return 1

    def find_files(
            self,
            start_time: np.datetime64,
            end_time: np.datetime64,
            time_step: np.timedelta64,
            roi: Optional[Geometry] = None,
            path: Optional[Path] = None
    ) -> List[Path]:
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
        time_range = TimeRange(start_time, end_time)
        if path is not None:
            path = Path(path)
            all_files = sorted(list(path.glob("**/*.nc")))
            matching = [path for path in all_files if gridsat_b1.matches(path) and gridsat_b1.get_temporal_coverage().covers(time_range)]
            return matching

        return gridsat_b1.find_files(TimeRange(start_time, end_time))

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

        with xr.open_dataset(path) as data:
            data = data[["irwin_cdr"]].rename({"irwin_cdr": "tbs"}).compute()

        for time_ind  in range(data.time.size):
            data_t = data[{"time": time_ind}]
            encoding = {
                "tbs": {
                    "scale_factor": 150 / 254,
                    "add_offset": 170,
                    "zlib": True,
                    "dtype": "uint8",
                    "_FillValue": 255
                }
            }
            filename = get_output_filename(
                self.name, data.time[time_ind].data, time_step
            )
            LOGGER.info(
                "Writing training samples to %s.",
                output_folder / filename
            )
            data_t.to_netcdf(output_folder / filename, encoding=encoding)


    def find_training_files(
            self,
            path: Union[Path, List[Path]],
            times: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[Path]]:
        """
        Find training data files.

        Args:
            path: Path to the folder the training data for all input
                and reference datasets.
            times: Optional array containing valid reference data times for static
                inputs.

        Return:
            A tuple ``(times, paths)`` containing the times for which training
            files are available and the paths pointing to the corresponding file.
        """
        pattern = "*????????_??_??.nc"
        if isinstance(path, str):
            paths = [Path(path)]
        elif isinstance(path, Path):
            paths = [path]
        elif isinstance(path, list):
            paths = path
        else:
            raise ValueError(
                "Expected 'path' to be a 'Path' object pointing to a folder "
                "or a list of 'Path' object pointing to input files. Got "
                "%s.",
                path
            )

        files = []
        for path in paths:
            if path.is_dir():
                files += sorted(list((path / "merged_ir").glob(pattern)))
            else:
                if path.match(pattern):
                    files.append(path)

        times = np.array(list(map(get_date, files)))
        return times, files


gridsat = GridSatB1()
gridsat_4 = GridSatB1(stack=4)
gridsat_8 = GridSatB1(stack=8)
gridsat_16 = GridSatB1(stack=16)
