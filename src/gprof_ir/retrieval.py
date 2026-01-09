"""
gprof_ir.retrieval
==================

Provides functionality to run the GPROF IR retrieval on GPM merged IR input files.
"""
from datetime import datetime, timedelta
from importlib.metadata import version
import gzip
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings

import click
from filelock import FileLock
from huggingface_hub import hf_hub_download
import numpy as np
from pytorch_retrieve.architectures import RetrievalModel
from pytorch_retrieve.inference import InferenceConfig, load_model, run_inference
from scipy.ndimage import binary_closing
import torch
import toml
import xarray as xr

from . import config


LOGGER = logging.getLogger(__name__)


def load_input_data(path: Path) -> np.ndarray:
    """
    Load brightness temperatures from NetCDF file.
    """
    path = Path(path)
    if path.suffix.startswith(".nc"):
        return xr.load_dataset(path)
    elif path.suffix == "":
        with open(path, "rb") as f:
            buf = f.read()
    elif path.suffix == "gz":
        with gzip.open(path, "rb") as f:
            buf = f.read()
    else:
        raise ValueError(
            "Encoutered input file with unsupported extension '%s'. Expected '.nc4', '.gz', or ''.",
            path.suffix
        )

    raw = np.frombuffer(buf, dtype="u1").reshape(9896, 3298, 2, order='F')
    mask = raw == 255
    tbs = raw + 75.0
    tbs[mask] = np.nan
    tbs = np.transpose(tbs, (2, 1, 0))
    tbs = np.flip(np.roll(tbs, 9896 // 2, -1), 1)

    lons = np.linspace(-179.98181, 179.98181, 9896)
    lats = np.linspace(-59.981808, 59.981808, 3298)
    time = np.datetime64(datetime.strptime(path.stem, "merg_%Y%m%d%H_4km-pixel"))
    times = np.array([time, time + np.timedelta64(30, "m")])

    return xr.Dataset({
        "time": (("time",), times),
        "lat": (("lat",), lats),
        "lon": (("lon",), lons),
        "Tb": (("time", "lat", "lon"), tbs.copy()),
    })


def download_model(variant: str = "gmi", n_steps: Optional[int] = None) -> Path:
    """
    Download GPROF-NN 3D model from hugging face.
    """
    repo_id = "simonpf/gprof_ir"
    if n_steps in [None, 1]:
        filename = f"gprof_ir_{variant}.pt"
    else:
        filename = f"gprof_ir_{variant}_{n_steps}.pt"
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
        device: "str"
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
    return inference_config


def get_previous_merged_ir_file(path: Path) -> Path:
    """
    Get path pointing to merged-IR file before the current input file.
    """
    path = Path(path)
    date = datetime.strptime(path.name.split("_")[1], "%Y%m%d%H")
    previous_date = date - timedelta(hours=1)
    fname = previous_date.strftime("merg_%Y%m%d%H_4km-pixel") + path.suffix
    return path.parent / fname


def load_ir_tbs_multi_step(path, n_steps: int) -> Path:
    """
    Get path pointing to merged-IR file before the current input file.
    """
    data = []
    curr_path = path
    files = [path]
    for _ in range((n_steps - 1) // 2):
        curr_path = get_previous_merged_ir_file(curr_path)
        files.append(curr_path)

    for path in files:
        if path.exists():
            data.append(load_input_data(path))
        else:
            LOGGER.warning(
                "Tried loading previous IR data from %s but the file doesn't exist.",
                path
            )
            dummy = data[-1].copy(deep=True)
            dummy.Tb.data[:] = np.nan
            data.append(dummy)

    data = xr.concat(data, dim="time").sortby("time")
    return data


class InputLoader:
    """
    Input loader for loading merged IR input observations.
    """
    def __init__(
            self,
            path: Path,
            variant: str,
            n_steps: int,
            start_time: Optional[np.datetime64] = None,
            end_time: Optional[np.datetime64] = None,
            input_format: Optional[str] = None,
            output_format: str = "netcdf",
            output_path: Optional[Path] = None
    ):
        """
        Args:
            path: A path object pointing to a directory containing the GPM merged IR files.
            n_steps: The numebr of input steps to load.
            variant: The model variant being run.
            start_time: If given, limits processing to files with timestamps at or after the given start time.
            start_time: If given, limites processing to files with timestamps earlier than the given end_time.
            input_format: String specifying the format of the input data.
        """
        path = Path(path)
        if path.is_dir():
            if input_format is None:
                files = sorted(list(path.glob("**/merg_??????????_4km-pixel*")))
            elif input_format.lower() in ["nc", "netcdf"]:
                files = sorted(list(path.glob("**/merg_??????????_4km-pixel.nc?")))
            else:
                raise ValueError(
                    "Unknown input data format it should either be 'bin' or 'binary' for binary format "
                    "or 'nc' or 'netcdf' for NetCDF4 format."
                )

            filtered = []
            for path in files:
                date = np.datetime64(datetime.strptime(path.name.split("_")[1], "%Y%m%d%H"))
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
        self.variant = variant
        self.n_steps = n_steps

        self.output_format = output_format
        if output_path is None:
            output_path = Path(".")
        else:
            output_path = Path(output_path)
        self.output_path = output_path

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
            input_data = load_input_data(path)
            inpt = {
                "merged_ir": torch.tensor(input_data.Tb.data[:, None])
            }
        else:
            input_data = load_ir_tbs_multi_step(path, n_steps=self.n_steps)
            n_times = input_data.time.size
            inpt = {
                "merged_ir": torch.stack([
                    torch.tensor(input_data.Tb.data[n_times - self.n_steps - 1: n_times - 1]),
                    torch.tensor(input_data.Tb.data[n_times - self.n_steps: n_times])
                ])
            }
            input_data = input_data[{"time": slice(n_times - 2, n_times)}]

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
        date_str = path.stem.split("_")[1]
        return inpt, aux, f"gprof_ir_{date_str}.nc"

    def __iter__(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any], str]:
        """
        Iterate over retrieval input files.
        """
        for path in self.files:
            yield self.load_input(path)

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
            output_path = self.output_path / (filename[:-3]  + ".bin")
            surface_precip.flatten(order='C').tofile(output_path)
            return None

        results = xr.Dataset({
            "latitude": (("latitude",), aux["latitude"]),
            "longitude": (("longitude",), aux["longitude"]),
            "time": (("time",), aux["time"]),
            "surface_precip": (("time", "latitude", "longitude"), surface_precip),
            "quality_flag": (("time", "latitude", "longitude"), quality)
        })

        results.surface_precip.encoding = {"dtype": "float32", "zlib": True}
        results.quality_flag.encoding = {"zlib": True}
        results.quality_flag.attrs["meaning"] = (
            "0: Good quality, 1: Missing input, 2: Invalid value returned from retrieval"
        )
        results.attrs["algorithm"] = f"gprof_ir, version {version('gprof_ir')}"
        results.attrs["variant"] = self.variant
        results.attrs["n_steps"] = self.n_steps
        return results, filename


def run_retrieval(
        input_path: Path,
        output_path: Optional[Path] = None,
        device: str = "cpu",
        dtype: str = "float32",
        variant: str = "gmi",
        n_steps: int = 1,
        output_format: str = "netcdf",
        start_time: Optional[np.datetime64] = None,
        end_time: Optional[np.datetime64] = None,
        n_threads: int = 8,
) -> List[xr.Dataset]:
    """
    Run GPROF-IR retrieval on given input data.

    Args:
        input_path: A path pointing to a single file or a folder containing merged IR input files.
        output_path: The path to which to write the output.
        device: The device to run the retrieval on. dtype: The dtype to use for running theretrieval.
        variant: String identifying the GPROF-IR variant to run.
        n_steps: The number of input steps to run.
        output_format: The format to use to write the output files.
        start_time: Optional start time to limit the input files being considered.
        end_time: Optional end time to limit the input files being considered.
        n_threads: The number of threads to use for CPU processing.

    Return:
        A list of xarray.Datasets containing the results for all input files.
    """

    # Load the model
    variant = variant.lower()
    valid = ["gmi", "cmb"]
    if variant not in valid:
        raise ValueError(
            f"'variant' must be one of {valid}."
        )
        return 1

    # Ensure that n_steps is not None only for GMI variant and that n_steps is either 3 or 5.
    if n_steps in [3, 5] and variant == "cmb":
        raise ValueError(
            f"Multiple input steps are only available for the 'gmi' variant."
        )
        return 1
    valid = [1, 3, 5]
    if n_steps not in valid:
        raise ValueError(
            f"'n_steps' must be one of {valid}."
        )
        return 1

    if output_format.lower() not in ["binary", "netcdf"]:
        raise ValueError(
            "'outpu_format' should be one of ['binary', 'netcdf']."
        )

    model = download_model(variant=variant, n_steps=n_steps)
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
    input_loader = InputLoader(
        input_path,
        n_steps=n_steps,
        variant=variant,
        output_format=output_format,
        output_path=output_path,
        start_time=start_time,
        end_time=end_time
    )
    torch.set_num_threads(n_threads)
    return run_inference(
        model,
        input_loader,
        inference_config,
        output_path=output_path,
        device=device,
        dtype=dtype,
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
    "--variant",
    type=str,
    default="gmi",
    help=(
        "The model variant: 'gmi' or 'cmb'."
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
def cli(
        input_path: Path,
        output_path: Optional[Path] = None,
        device: str = "cpu",
        dtype: str = "float32",
        variant: str = "gmi",
        n_steps: Optional[int] = None,
        output_format: str = "netcdf",
        start_time: Optional[np.datetime64] = None,
        end_time: Optional[np.datetime64] = None,
        n_threads: int = 8
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
            return 1

    if end_time is not None:
        try:
            end_time = np.datetime64(end_time)
        except ValueError as err:
            LOGGER.error(
                "Error parsing end time '%s'",
                end_time
            )
            return 1

    res = run_retrieval(
        input_path=input_path,
        output_path=output_path,
        device=device,
        dtype=dtype,
        variant=variant,
        n_steps=n_steps,
        output_format=output_format,
        start_time=start_time,
        end_time=end_time,
        n_threads=n_threads
    )
    # Return error code.
    if isinstance(res, int):
        return res
