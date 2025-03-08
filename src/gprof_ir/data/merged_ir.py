"""
gprof_ir.data.merged_ir
=======================

This module provides a CHIMP input dataset consisting of the GPM Merged IR observations.
"""
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from pansat.products.satellite import gpm
from pansat.time import to_datetime64, TimeRange
from pansat.geometry import Geometry
from pyresample import geometry, kd_tree, create_area_def
import xarray as xr

from chimp.areas import Area
from chimp.data.input import InputDataset
from chimp.data.resample import resample_and_split
from chimp.data.utils import get_output_filename, records_to_paths


LOGGER = logging.getLogger(__name__)


CPCIR_GRID = create_area_def(
    "cpcir_area",
    {"proj": "longlat", "datum": "WGS84"},
    area_extent=[-180.0, -60.0, 180.0, 60.0],
    resolution= (0.03637833468067906, 0.036385688295936934),
    units="degrees",
    description="CPCIR grid",
)


class MergedIR(InputDataset):
    """
    Represents input data derived from merged IR data.
    """
    def __init__(self):
        InputDataset.__init__(
            self,
            "merged_ir",
            "merged_ir",
            4,
            "tbs",
            n_dim=2,
            spatial_dims=("latitude", "longitude")
        )
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
            matching = [path for path in all_files if gpm.merged_ir.matches(path) and gpm.merged_ir.get_temporal_coverage().covers(time_range)]
            return matching

        return gpm.merged_ir.find_files(TimeRange(start_time, end_time))

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

        data = gpm.merged_ir.open(path).rename({"Tb": "tbs"})
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


merged_ir = MergedIR()
