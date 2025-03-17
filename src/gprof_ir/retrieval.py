"""
gprof_ir.retrieval
==================

Provides functionality to run the GPROF IR retrieval on GPM merged IR input files.
"""
from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import warnings

import click
from filelock import FileLock
from huggingface_hub import hf_hub_download
import numpy as np
from pytorch_retrieve.inference import InferenceConfig, load_model, run_inference
from scipy.ndimage import binary_closing
import torch
import toml
import xarray as xr

from . import config


LOGGER = logging.getLogger(__name__)


def download_model(model_name: str) -> Path:
    """
    Download GPROF-NN 3D model from hugging face.
    """
    repo_id = "simonpf/gprof_ir"
    filename = "gprof_ir_ss.pt"
    model_path = Path(config.CONFIG.get("model_path"))
    model_file = model_path / filename
    if not model_file.exists():
        lock = FileLock(model_file.with_suffix(".lock"))
        with lock:
            LOGGER.info("Downloading model to %s", str(model_path))
            model_file = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=model_path)
    return model_file


class InputLoader:
    """
    Input loader for loading merged IR input observations.
    """
    def __init__(
            self,
            path: Path,
            start_time: Optional[np.datetime64] = None,
            end_time: Optional[np.datetime64] = None
    ):
        """
        Args:
            path: A path object pointing to a directory containing the GPM merged IR files.
        """
        path = Path(path)
        if path.is_dir():
            files = sorted(list(path.glob("**/merg_*.nc?")))
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
        input_data = xr.load_dataset(path)
        inpt = {
        "merged_ir": torch.tensor(input_data.Tb.data[:, None])
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
        "valid_input": valid[..., ::2, ::2]
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
        quality[invalid] = 2

        valid_input = aux["valid_input"]
        surface_precip[~valid_input] = np.nan
        quality[~valid_input] = 1

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
        return results, filename


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
def cli(
        input_path: Path,
        output_path: Optional[Path] = None,
        device: str = "cpu",
        dtype: str = "float32",
) -> None:
    """
    Run GPROF IR retrieval on INPUT_PATH.
    """

    # Load the model
    model = download_model("gprof_ir_ss.pt")
    warnings.filterwarnings("ignore", module="torch")
    model = torch.compile(load_model(model))

    # Inference config

    config_path = Path(__file__).parent / "config_files" / "gprof_ir_ss_inference.toml"
    inference_config = toml.loads(open(config_path).read())
    inference_config = InferenceConfig.parse(
        model.output_config,
        inference_config
    )
    if device == "cpu":
        inference_config.batch_size = 1

    # Input loader
    input_path = Path(input_path)
    if not input_path.exists():
        LOGGER.error(
            "Input path ('%s') must point to an existing file or directory."
        )
    input_loader = InputLoader(input_path)

    # Output path
    if output_path is None:
        output_path = Path(".")

    run_inference(
        model,
        input_loader,
        inference_config,
        output_path=output_path,
        device=device,
        dtype=dtype,
    )
