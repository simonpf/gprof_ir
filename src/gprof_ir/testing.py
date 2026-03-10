"""
gprof_ir.testing
================

Provides testing functionality for GPROF-NN retrievals.
"""
from pathlib import Path
from typing import Dict, List, Optional

import click
import numpy as np
from pytorch_retrieve.inference import to_rec
from pytorch_retrieve import metrics
from pytorch_retrieve.metrics import ScalarMetric
from pytorch_retrieve.architectures import load_model
from pytorch_retrieve.tensors import MaskedTensor
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import xarray as xr

from chimp.data.training_data import SingleStepDataset

from .retrieval import download_model, load_model


def run_tests(
        model: nn.Module,
        test_dataset: DataLoader,
        scalar_metrics: Dict[str, List[ScalarMetric]],
        device: str = "cuda",
        dtype: str = "float32"
) -> xr.Dataset:
    """
    Evaluate retrieval model on test set.

    Args:
        model: A trained retrieval model.
        test_dataset: A dataset providing access to the test data.
        scalar_metrics: A dictionary mapping target names to corresponding
             metrics to evaluate.
        tile_size: A tile size to use for the evaluation.
        device: The device on which to perform the evaluation.
        dtype: The dtype to use.

    Return:
        A the xarray.Dataset containing the calculated error metrics.
    """
    model = model.to(device=device, dtype=dtype).eval()

    for x, y in tqdm(test_dataset):
        x = to_rec(x, device=device, dtype=dtype)

        y = to_rec(y, device=device, dtype=dtype)
        for key, target in y.items():
            mask = torch.isnan(target)
            if mask.any():
                y[key] = MaskedTensor(target, mask=mask)

        with torch.no_grad():
            pred = model(x)

        for key, pred_k in pred.items():
            mtrcs = scalar_metrics.get(key, [])

            pred_k = pred_k.expected_value()
            ref = y[key]

            for metric in mtrcs:
                metric = metric.to(device=device)
                metric.update(pred_k, ref)

    retrieval_results = {}
    dims = ("reference", "retrieved")
    for name, mtrcs in scalar_metrics.items():
        for metric in mtrcs:
            res_name = name + "_" + metric.name.lower()
            res = metric.compute().cpu().numpy()
            if 1 < res.ndim:
                retrieval_results[res_name] = (dims[:res.ndim], metric.compute().cpu().numpy())
            else:
                retrieval_results[res_name] = metric.compute().cpu().numpy()

    if len(retrieval_results) > 0:
        retrieval_results = xr.Dataset(retrieval_results)
    else:
        retrieval_results = None

    return retrieval_results


@click.argument("config")
@click.argument("test_data_path")
@click.argument("output_filename")
@click.option("--device", type=str, default="cuda")
@click.option("--dtype", type=str, default="bfloat16")
@click.option("--batch_size", type=int, default=32)
@click.option("--n_steps", type=int, default=1)
@click.option("--subsample", type=int, default=1)
@click.option("-v", "--verbose", count=True)
def cli(
        config: str,
        test_data_path: str,
        output_filename: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        batch_size: int = 32,
        n_steps: int = 1,
        subsample: int = 1,
        verbose: int = 0,
) -> int:
    """
    Calculate test data accuracy for a given GPROF-IR CONFIG using the test data located in TEST_DATA_PATH.
    """
    config = config.lower()
    path = download_model(config, n_steps)
    model = load_model(path).eval()

    test_data_path = Path(test_data_path)

    inpt_data = f"merged_ir_{n_steps}" if 1 < n_steps else "merged_ir"
    ref_data = "gpm" if config == "cmb" else "gpm_gmi"
    print(ref_data, config, path)
    test_dataset = SingleStepDataset(
        test_data_path,
        input_datasets=[inpt_data],
        reference_datasets=[ref_data],
        sample_rate=(1.0 / subsample),
        scene_size=256,
        augment=False,
        validation=True
    )
    data_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    scalar_metrics = {
        name: [
            metrics.RelativeBias(),
            metrics.MSE(),
            metrics.CorrelationCoef(),
            metrics.ScatterPlot(bins=np.logspace(-2, 2.5, 101), conditional={})
        ] for name in model.to_config_dict()["output"].keys()
    }

    device = torch.device(device)
    dtype = getattr(torch, dtype)

    retrieval_results = run_tests(
        model,
        data_loader,
        scalar_metrics=scalar_metrics,
        device=device,
        dtype=dtype,
    )

    if retrieval_results is not None:
        retrieval_results.to_netcdf(output_filename)
