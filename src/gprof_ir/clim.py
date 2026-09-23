"""
gprof_ir.clim
==================

Provides functionality to run the climate version of GPROF-IR.
"""
from datetime import datetime, timedelta
from importlib.metadata import version
import gzip
from functools import cached_property
import logging
from math import ceil
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import warnings

import click
from filelock import FileLock
from huggingface_hub import hf_hub_download
import numpy as np
from pytorch_retrieve.architectures import RetrievalModel
from pytorch_retrieve.inference import InferenceConfig, load_model, run_inference
from pytorch_retrieve.config import RetrievalOutputConfig
from scipy.ndimage import binary_closing
import torch
import toml
import xarray as xr

from . import config


LOGGER = logging.getLogger(__name__)


def get_date(path: Path) -> datetime:
    """
    Parse date from GridSat B1 filename.

    Args:
        path: A path object pointing to the file.

    Return:
        A datetime object representing the date.
    """
    date = datetime.strptime(Path(path).stem, "GRIDSAT-B1.%Y.%m.%d.%H.v02r01")
    return date


def load_input_data(
        path: Path,
        slices: Optional[Dict[str, slice]] = None
) -> xr.Dataset:
    """
    Load brightness temperatures from NetCDF file.

    Args:
        path: A path object pointing to the file from which to load the input data.
        slices: An optional dictionary to subset loading of input data from xarray.Dataset.

    Return:
        An xarray.Dataset containing the input observations.
    """
    path = Path(path)
    with xr.open_dataset(path) as data:
        if slices is None:
            return data[["irwin_cdr"]].load()
        return data[["irwin_cdr"]][slices].load()


def download_model(n_steps: Optional[int] = None) -> Path:
    """
    Download GPROF-NN 3D model from hugging face.
    """
    repo_id = "simonpf/gprof_ir"
    if n_steps in [None, 1]:
        filename = f"gprof_ir_clim_1.pt"
    else:
        filename = f"gprof_ir_clim_4.pt"
    model_path = Path(config.CONFIG.get("model_path"))
    model_file = model_path / filename
    if not model_file.exists():
        lock = FileLock(model_file.with_suffix(".lock"))
        with lock:
            LOGGER.info("Downloading model to %s", str(model_path))
            model_file = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=model_path)
    return model_file


def load_inference_config(
        model: RetrievalModel,
        device: "str",
        include_probabilities: bool = False
) -> InferenceConfig:
    """
    Load inference config for GPROF-IR model.

    Args:
        model:
    """
    config_path = Path(__file__).parent / "config_files" / "gprof_ir_ss_inference.toml"
    inference_config = toml.loads(open(config_path).read())
    inference_config = InferenceConfig.parse(
        model.output_config,
        inference_config
    )
    if device == "cpu":
        inference_config.batch_size = 1

    if include_probabilities:
        output = inference_config.retrieval_output["surface_precip"]
        output["probability_of_precip"] = RetrievalOutputConfig(
            model.output_config["surface_precip"],
            "ExceedanceProbability",
            {"threshold": 1e-1}
        )
        output["probability_of_heavy_precip"] = RetrievalOutputConfig(
            model.output_config["surface_precip"],
            "ExceedanceProbability",
            {"threshold": 1e1}
        )

    return inference_config


def get_previous_input_file(path: Path) -> Path:
    """
    Get path pointing to GridSat B1 file before the current input file.
    """
    path = Path(path)
    date = get_date(path)
    previous_date = date - timedelta(hours=3)
    fname = previous_date.strftime("GRIDSAT-B1.%Y.%m.%d.%H.v02r01") + path.suffix
    return path.parent / fname

def get_next_input_file(path: Path) -> Path:
    """
    Get path pointing to GridSat B1 file before the current input file.
    """
    path = Path(path)
    date = get_date(path)
    previous_date = date + timedelta(hours=3)
    fname = previous_date.strftime("GRIDSAT-B1.%Y.%m.%d.%H.v02r01") + path.suffix
    return path.parent / fname


def load_ir_tbs_multi_step(
        input_file: Path,
        n_steps: int,
        slices: Optional[Dict[str, slice]] = None
) -> xr.Dataset:
    """
    Get path pointing to GridSat B1 file before the current input file.

    Args:
        path: A path object pointing to the file containing the two time steps for which to retrieve
            precipitation.
        n_steps: The number of previous input steps to load.
        slices: Optional slices to limit the data loaded from the input files.

    Return:
        An xarray.Dataset containing the loaded input data required to run the GPROF-IR retrieval with
        n_steps timesteps.
    """
    if n_steps == 1:
        input_files = [input_file]
    else:
        file_p = get_previous_input_file(input_file)
        file_pp = get_previous_input_file(file_p)
        file_n = get_next_input_file(input_file)
        input_files = [
            file_pp,
            file_p,
            input_file,
            file_n
        ]

    data = []
    for path in input_files:
        path = Path(path)
        if path.exists():
            data.append(load_input_data(path, slices=slices))
        else:
            LOGGER.warning(
                "Tried IR input  data from %s but the file doesn't exist.",
                path
            )
            dummy = load_input_data(input_file).copy(deep=True)
            dummy.irwin_cdr.data[:] = np.nan
            data.append(dummy)

    data = xr.concat(data, dim="time").sortby("time")
    return data


class MultiInputLoader:
    """
    Input loader for loading GridSat B1 input observations.
    """
    def __init__(
            self,
            path: Path,
            n_steps: int,
            start_time: Optional[np.datetime64] = None,
            end_time: Optional[np.datetime64] = None,
            output_format: str = "netcdf",
            output_path: Optional[Path] = None,
            roi: Optional[Tuple[float, float, float, float]] = None
    ):
        """
        Args:
            path: A path object pointing to a directory containing the GridSat B1 files.
            n_steps: The numebr of input steps to load.
            start_time: If given, limits processing to files with timestamps at or after the given start time.
            start_time: If given, limites processing to files with timestamps earlier than the given end_time.
        """
        path = Path(path)
        if path.is_dir():
            files = sorted(list(path.glob("**/GRIDSAT-B1.*v02r01*.nc")))

            filtered = []
            for path in files:
                date = np.datetime64(get_date(path))
                if start_time is not None:
                    if date < start_time:
                        continue
                if end_time is not None:
                    if end_time < date:
                        continue
                filtered.append(path)
            self.files = filtered

            LOGGER.info(
                "Found %s files in %s.",
                len(self.files), str(path)
            )
        else:

            self.files = [path]

        self.n_steps = n_steps

        self.output_format = output_format
        if output_path is None:
            output_path = Path(".")
        else:
            output_path = Path(output_path)
        self.output_path = output_path
        self.roi = roi

    @cached_property
    def roi_slices(self):
        """
        A dictionary containing the slices to subset the loaded input data to the given ROI.
        """
        if self.roi is None:
            return {
                "lat": slice(0, None),
                "lon": slice(0, None)
            }
        lons = np.linspace(-180, 179.94, 5143) + 0.04
        lats = np.linspace(-70, 69.93001, 2000) + 0.04
        lon_min, lat_min, lon_max, lat_max = self.roi
        lat_mask = (lat_min <= lats) * (lats <= lat_max)
        lon_mask = (lon_min <= lons) * (lons <= lon_max)
        lat_inds = np.where(lat_mask)[0]
        lon_inds = np.where(lon_mask)[0]

        height = lat_inds[-1] - lat_inds[0]
        height = max(256 * ceil(height / 256), 512)
        lat_c = int(0.5 * (lat_inds[0] + lat_inds[-1]))
        lat_start = min(max(lat_c - height // 2, 0), lats.size - height)
        lat_end = lat_start + height

        width = lon_inds[-1] - lon_inds[0]
        width = max(256 * ceil(width / 256), 512)
        lon_c = int(0.5 * (lon_inds[0] + lon_inds[-1]))
        lon_start = min(max(lon_c - width // 2, 0), lons.size - width)
        lon_end = lon_start + width

        return {
            "lat": slice(lat_start, lat_end),
            "lon": slice(lon_start, lon_end),
        }

    def __len__(self) -> int:
        return len(self.files)

    def load_input(self, path: Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any], str]:
        """
        Loads the retrieval input data from the given input file.

        Args:
            path: A Path object pointing to the file from which to load the input data.

        Return:
            A tuple ``(inpt, aux, filename)`` containing the retrieval input as PyTorch tensors in
            ``inpt``, auxiliary data in ``aux``, and the input filename.
        """
        if self.n_steps == 1:
            input_data = load_input_data(
                path,
                slices=self.roi_slices
            )
            inpt = {
                "gridsat_b1": torch.tensor(input_data.irwin_cdr.data[:, None])
            }
        else:
            input_data = load_ir_tbs_multi_step(
                path,
                n_steps=self.n_steps,
                slices=self.roi_slices
            )
            n_times = input_data.time.size
            inpt = {
                "gridsat_b1": torch.stack([
                    torch.tensor(input_data.irwin_cdr.data[n_times - self.n_steps - 1: n_times - 1]),
                    torch.tensor(input_data.irwin_cdr.data[n_times - self.n_steps: n_times])
                ])
            }
            input_data = input_data[{"time": slice(n_times - 2, n_times)}]

        # Calculate invalid input mask
        valid = np.isfinite(input_data.irwin_cdr.data)
        elem = np.ones((1, 8, 8))
        valid = binary_closing(valid, elem, border_value=1)

        lats = input_data.lat.data
        lons = input_data.lon.data

        aux = {
            "latitude": lats,
            "longitude": lons,
            "time": input_data.time.data,
            "valid_input": valid,
            "n_steps": self.n_steps,
            "variant": 'clim'
        }
        date = input_data.time.data.astype("datetime64[s]").item()
        date_str = date.strftime("%Y%m%d%H%M")
        return inpt, aux, f"gprof_ir_clim_{date_str}.nc"

    def __iter__(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any], str]:
        """
        Iterate over retrieval input files.
        """
        for path in self.files:
            try:
                yield self.load_input(path)
            except Exception:
                continue

    def __getitem__(self, ind: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any], str]:
        """
        Load input data for file with given index.
        """
        return self.load_input(self.files[ind])

    def finalize_results(
            self,
            results: Dict[str, torch.Tensor],
            aux: Dict[str, Any],
            filename: str,
            **kwargs
    ):
        """
        Finalizes the retrieval results.

        Args:
            results: Dictionary containing the retrieval results.
            aux: Auxiliary data as returned by this input loader.
            filename: The output filename as returned by the input loader.
        """
        surface_precip = results["surface_precip"].data.numpy()[:, 0]
        quality = np.zeros_like(surface_precip, dtype=np.int8)

        invalid = (surface_precip < -0.01) * (surface_precip > 200)
        surface_precip[invalid] = np.nan
        surface_precip = np.maximum(surface_precip, 0.0)
        quality[invalid] = 2

        valid_input = aux["valid_input"]
        surface_precip[~valid_input] = np.nan
        quality[~valid_input] = 1

        if self.output_format != "netcdf":
            surface_precip = np.roll(np.flip(surface_precip, -2), surface_precip.shape[-1] // 2, -1)
            output_path = self.output_path / (Path(filename).stem + ".bin")
            surface_precip.flatten(order='C').tofile(output_path)
            return output_path

        if "probability_of_precip" in results:
            pop = results["probability_of_precip"].data.numpy()[:, 0]
            pop_heavy = results["probability_of_heavy_precip"].data.numpy()[:, 0]
        else:
            pop = None
            pop_heavy = None

        results = xr.Dataset({
            "latitude": (("latitude",), aux["latitude"]),
            "longitude": (("longitude",), aux["longitude"]),
            "time": (("time",), aux["time"]),
            "surface_precip": (("time", "latitude", "longitude"), surface_precip),
            "quality_flag": (("time", "latitude", "longitude"), quality)
        })

        if pop is not None:
            results["probability_of_precip"] = (("time", "latitude", "longitude"), pop)
            results["probability_of_heavy_precip"] = (("time", "latitude", "longitude"), pop_heavy)

        results.surface_precip.encoding = {"dtype": "float32", "zlib": True}
        results.quality_flag.encoding = {"zlib": True}
        results.quality_flag.attrs["meaning"] = (
            "0: Good quality, 1: Missing input, 2: Invalid value returned from retrieval"
        )
        results.attrs["algorithm"] = f"gprof_ir, version {version('gprof_ir')}"
        results.attrs["variant"] = "clim"
        results.attrs["n_steps"] = self.n_steps
        return results, filename


def run_retrieval_multi(
        input_path: Path,
        output_path: Optional[Path] = None,
        device: str = "cpu",
        dtype: str = "float32",
        n_steps: int = 1,
        output_format: str = "netcdf",
        start_time: Optional[np.datetime64] = None,
        end_time: Optional[np.datetime64] = None,
        n_threads: int = 8,
        roi: Optional[Tuple[float, float, float, float]] = None,
        progress: bool = True,
        include_probabilities: bool = False
) -> List[xr.Dataset]:
    """
    Run GPROF-IR retrieval on given input data.

    Args:
        input_path: A path pointing to a single file or a folder containing GridSat B1 input files.
        output_path: The path to which to write the output.
        device: The device to run the retrieval on. dtype: The dtype to use for running theretrieval.
        n_steps: The number of input steps to run.
        output_format: The format to use to write the output files.
        start_time: Optional start time to limit the input files being considered.
        end_time: Optional end time to limit the input files being considered.
        n_threads: The number of threads to use for CPU processing.
        roi: An optional region of interest defined as a tuple (lon_min, lat_min, lon_max, lat_max) to
            run the retrieval on a regional subset of data.
        progress: Whether or not to display a progress bar.
        include_probabilities: Set to 'True' to include precipitation probabilities in output.

    Return:
        A list of xarray.Datasets containing the results for all input files.
    """
    valid = [1, 4]
    if n_steps not in valid:
        raise ValueError(
            f"'n_steps' must be one of {valid}."
        )
        sys.exit(1)

    if output_format.lower() not in ["binary", "netcdf"]:
        raise ValueError(
            "'output_format' should be one of ['binary', 'netcdf']."
        )

    model = download_model(n_steps=n_steps)
    warnings.filterwarnings("ignore", module="torch")
    model = load_model(model).eval()
    n_steps = model.encoder.stages[0].projection.weight.shape[1]

    # Inference config
    inference_config = load_inference_config(
        model,
        device,
        include_probabilities=include_probabilities
    )

    # Input loader
    input_path = Path(input_path)
    if not input_path.exists():
        LOGGER.error(
            "Input path ('%s') must point to an existing file or directory.",
            input_path
        )
        sys.exit(1)

    input_loader = MultiInputLoader(
        input_path,
        n_steps=n_steps,
        output_format=output_format,
        output_path=output_path,
        start_time=start_time,
        end_time=end_time,
        roi=roi
    )
    torch.set_num_threads(n_threads)
    return run_inference(
        model,
        input_loader,
        inference_config,
        output_path=output_path,
        device=device,
        dtype=dtype,
        progress=progress
    )

@click.argument("input_path", type=str)
@click.option(
    "--output_path",
    type=str,
    metavar="PATH",
    default=None,
    help=(
        "Directory to write the retrieval results to. Defaults to current working directory."
    )
)
@click.option(
    "--device",
    type=str,
    default="cpu",
    help=(
        "The device on which to perform inference."
    )
)
@click.option(
    "--dtype",
    type=str,
    default="float32",
    help=(
        "The floating point type to use for inference."
    )
)
@click.option(
    "--n_steps",
    type=int,
    default=3,
    help=(
        "The number of input steps: None, 3, or 5"
    )
)
@click.option(
    "--output_format",
    type=str,
    default="netcdf",
    help=(
        "The format used to store the retrieval results. Shoule be 'netcdf' for NetCDF4 format"
        " (default) or 'binary' for GPROF binary format."
    )
)
@click.option(
    "--start_time",
    type=str,
    default=None,
    help=(
        "Optional start time in YYYY-MM-DDTHH:MM:SS format to limit the input files to consider."
    )
)
@click.option(
    "--end_time",
    type=str,
    default=None,
    help=(
        "Optional end time in YYYY-MM-DDTHH:MM:SS format to limit the input files to consider."
    )
)
@click.option(
    "--n_threads",
    type=int,
    default=8,
    help="The number of threads to use for CPU processing."
)
@click.option(
    "--probabilities",
    is_flag=True
)
def cli_multi_clim(
        input_path: Path,
        output_path: Optional[Path] = None,
        device: str = "cpu",
        dtype: str = "float32",
        n_steps: Optional[int] = None,
        output_format: str = "netcdf",
        start_time: Optional[np.datetime64] = None,
        end_time: Optional[np.datetime64] = None,
        n_threads: int = 8,
        probabilities: bool = False
) -> None:
    """
    Run GPROF IR retrieval on INPUT_PATH.
    """
    # Output path
    if output_path is None:
        output_path = Path(".")

    if start_time is not None:
        try:
            start_time = np.datetime64(start_time)
        except ValueError as err:
            LOGGER.error(
                "Error parsing start time '%s'",
                start_time
            )
            sys.exit(1)

    if end_time is not None:
        try:
            end_time = np.datetime64(end_time)
        except ValueError as err:
            LOGGER.error(
                "Error parsing end time '%s'",
                end_time
            )
            sys.exit(1)

    res = run_retrieval_multi(
        input_path=input_path,
        output_path=output_path,
        device=device,
        dtype=dtype,
        n_steps=n_steps,
        output_format=output_format,
        start_time=start_time,
        end_time=end_time,
        n_threads=n_threads,
        include_probabilities=probabilities
    )
    # Return error code.
    if isinstance(res, int):
        sys.exit(res)


class SingleInputLoader:
    """
    Input loader for loading the input for a single retrieval.
    """
    def __init__(
            self,
            input_file: Path,
            output_file: Path,
            n_steps: int,
            output_format: str = "netcdf",
            output_path: Optional[Path] = None
    ):
        """
        Args:
            input_file: A path object pointing to the input file.
            output_file: A path object pointing to the output file.
            n_steps: The numebr of input steps to load.
            output_format: The format to use for the retrieval results.
            output_path: The folder to which to write the results.
        """
        self.input_file = input_file
        self.output_file = output_file
        self.n_steps = n_steps
        self.output_format = output_format
        if output_path is None:
            output_path = Path(".")
        else:
            output_path = Path(output_path)
        self.output_path = output_path

    def __len__(self) -> int:
        return 1

    def load_input(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any], str]:
        """
        Loads the retrieval input data from the given input file.

        Args:
            path: A Path object pointing to the file from which to load the input data.

        Return:
            A tuple ``(inpt, aux, filename)`` containing the retrieval input as PyTorch tensors in
            ``inpt``, auxiliary data in ``aux``, and the input filename.
        """
        if self.n_steps == 1:
            input_data = load_input_data(self.input_file)
            inpt = {
                "gridsat_b1": torch.tensor(input_data.irwin_cdr.data[:, None])
            }
        else:
            input_data = load_ir_tbs_multi_step(
                self.input_file,
                n_steps=self.n_steps,
            )
            n_times = input_data.time.size
            inpt = {
                "gridsat_b1": torch.tensor(input_data.irwin_cdr.data)[None]
            }
            input_data = input_data[{"time": 1}]

        # Calculate invalid input mask
        valid = np.isfinite(input_data.irwin_cdr.data[0])
        elem = np.ones((8, 8))
        valid = binary_closing(valid, elem, border_value=1)

        lats = input_data.lat.data
        lons = input_data.lon.data

        aux = {
            "latitude": lats,
            "longitude": lons,
            "time": input_data.time.data,
            "valid_input": valid,
            "n_steps": self.n_steps,
        }
        return inpt, aux, self.output_file

    def __iter__(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any], str]:
        """
        Iterate over retrieval input files.
        """
        yield self.load_input()

    def __getitem__(self, ind: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any], str]:
        """
        Load input data for file with given index.
        """
        inpt = self.load_input()
        return self.load_input()


    def finalize_results(
            self,
            results: Dict[str, torch.Tensor],
            aux: Dict[str, Any],
            filename: str,
            **kwargs
    ):
        """
        Finalizes the retrieval results.

        Args:
            results: Dictionary containing the retrieval results.
            aux: Auxiliary data as returned by this input loader.
            filename: The output filename as returned by the input loader.
        """
        surface_precip = results["surface_precip"].data.numpy()[0, 0]
        quality = np.zeros_like(surface_precip, dtype=np.int8)

        invalid = (surface_precip < -0.01) * (surface_precip > 200)
        surface_precip[invalid] = np.nan
        surface_precip = np.maximum(surface_precip, 0.0)
        quality[invalid] = 2

        valid_input = aux["valid_input"]
        surface_precip[~valid_input] = np.nan
        quality[~valid_input] = 1

        if self.output_format != "netcdf":
            surface_precip = np.roll(np.flip(surface_precip, -2), surface_precip.shape[-1] // 2, -1)
            output_path = self.output_path
            surface_precip.flatten(order='C').tofile(output_path)
            return output_path

        if "probability_of_precip" in results:
            pop = results["probability_of_precip"].data.numpy()[:, 0]
            pop_heavy = results["probability_of_heavy_precip"].data.numpy()[:, 0]
        else:
            pop = None
            pop_heavy = None

        results = xr.Dataset({
            "latitude": (("latitude",), aux["latitude"]),
            "longitude": (("longitude",), aux["longitude"]),
            "time": (("time",), [aux["time"]]),
            "surface_precip": (("latitude", "longitude"), surface_precip),
            "quality_flag": (("latitude", "longitude"), quality)
        })

        if pop is not None:
            results["probability_of_precip"] = (("time", "latitude", "longitude"), pop)
            results["probability_of_heavy_precip"] = (("time", "latitude", "longitude"), pop_heavy)

        results.surface_precip.encoding = {"dtype": "float32", "zlib": True}
        results.quality_flag.encoding = {"zlib": True}
        results.quality_flag.attrs["meaning"] = (
            "0: Good quality, 1: Missing input, 2: Invalid value returned from retrieval"
        )
        results.attrs["algorithm"] = f"gprof_ir, version {version('gprof_ir')}"
        results.attrs["variant"] = "clim"
        results.attrs["n_steps"] = self.n_steps
        return results, filename


def run_retrieval_single(
        input_path: Path,
        output_path: Path,
        n_steps: int = 4,
        device: str = "cpu",
        dtype: str = "float32",
        output_format: str = "netcdf",
        n_threads: int = 8,
) -> List[xr.Dataset]:
    """
    Run GPROF-IR retrieval on given input data.

    Args:
        input_path: A path pointing to a single file or a folder containing GridSat B1 input files.
        output_path: The path to which to write the output.
        device: The device to run the retrieval on. dtype: The dtype to use for running theretrieval.
        output_format: The format to use to write the output files.
        n_threads: The number of threads to use for CPU processing.

    Return:
        A list of xarray.Datasets containing the results for all input files.
    """
    valid = [1, 4]
    if n_steps not in valid:
        raise ValueError(
            f"'n_steps' must be one of {valid}."
        )
        sys.exit(1)

    if output_format.lower() not in ["binary", "netcdf"]:
        raise ValueError(
            "'output_format' should be one of ['binary', 'netcdf']."
        )

    model = download_model(n_steps=n_steps)
    warnings.filterwarnings("ignore", module="torch")
    model = load_model(model).eval()
    n_steps = model.encoder.stages[0].projection.weight.shape[1]

    # Inference config
    inference_config = load_inference_config(model, device)

    # Input loader
    input_path = Path(input_path)
    if not input_path.exists():
        LOGGER.error(
            "Input path ('%s') must point to an existing file or directory.",
            input_path
        )
        return 1

    input_loader = SingleInputLoader(
        input_path,
        output_path,
        n_steps=n_steps,
        output_format=output_format,
        output_path=output_path,
    )
    torch.set_num_threads(n_threads)
    try:
        return run_inference(
            model,
            input_loader,
            inference_config,
            output_path=output_path,
            device=device,
            dtype=dtype,
            progress=False,
            robust=False
        )
    except Exception as exc:
        LOGGER.error(
            "The following error was encountered while running the retrieval: %s",
            exc

        )
        return 1


@click.argument("input_files", type=str, nargs=-1)
@click.argument("output_path", type=str)
@click.option(
    "--device",
    type=str,
    default="cpu",
    help=(
        "The device on which to perform inference."
    )
)
@click.option(
    "--dtype",
    type=str,
    default="float32",
    help=(
        "The floating point type to use for inference."
    )
)
@click.option(
    "--output_format",
    type=str,
    default="netcdf",
    help=(
        "The format used to store the retrieval results. Shoule be 'netcdf' for NetCDF4 format"
        " (default) or 'binary' for GPROF binary format."
    )
)
@click.option(
    "--n_threads",
    type=int,
    default=8,
    help="The number of threads to use for CPU processing."
)
def cli_single_clim(
        input_files: List[Path],
        output_path: Path,
        device: str = "cpu",
        dtype: str = "float32",
        variant: str = "gmi",
        output_format: str = "netcdf",
        n_threads: int = 8,
) -> None:
    """
    Run GPROF IR retrieval on INPUT_PATH.
    """
    if len(input_files) == 0:
        LOGGER.error(
            "Need at least one input file.",
        )
        return 1

    if 3 < len(input_files):
        LOGGER.error(
            "GPROF-IR support three input files at most.",
        )
        return 1
    input_path = input_files[0]

    res = run_retrieval_single(
        input_path=input_path,
        output_path=output_path,
        device=device,
        dtype=dtype,
        output_format=output_format,
        n_threads=n_threads,
    )
    # Return error code.
    if isinstance(res, int):
        sys.exit(res)
    sys.exit()
