"""
Tests for the gprof_ir.retrieval module.
"""
import subprocess

import numpy as np
import pytest
import xarray as xr

from gprof_ir.retrieval import InputLoader


def test_input_loader(retrieval_input_data):
    """
    Test loading of input data.
    """
    # Test loading of all files in folder.
    input_loader = InputLoader(
        retrieval_input_data,
        "gmi",
        1,
    )
    assert len(input_loader) == 1
    inpt, aux, fname = input_loader[0]
    inpt, aux, fname = next(iter(input_loader))

    # Test loading of a single file.
    input_loader = InputLoader(
        retrieval_input_data / "merg_2020010100_4km-pixel.nc4",
        "gmi",
        1,
    )
    assert len(input_loader) == 1
    inpt, aux, fname = input_loader[0]
    inpt, aux, fname = next(iter(input_loader))

    # Test filtering of files by date.
    input_loader = InputLoader(
        retrieval_input_data,
        "gmi",
        1,
        start_time=np.datetime64("2021-01-01"),
    )
    assert len(input_loader) == 0

    input_loader = InputLoader(
        retrieval_input_data,
        "gmi",
        1,
        end_time=np.datetime64("2019-12-31"),
    )
    assert len(input_loader) == 0


@pytest.mark.parametrize("variant", ["cmb", "gmi"])
def test_retrieval(
        retrieval_input_data,
        tmp_path,
        variant
):
    """
    Test running the GPROF IR retrieval.
    """
    args = [
        "gprof_ir",
        "retrieve",
        str(retrieval_input_data / "merg_2020010100_4km-pixel.nc4"),
        "--output_path",
        str(tmp_path),
        "--variant",
        variant,
        "--n_steps",
        "1"
    ]
    subprocess.run(args)
    assert (tmp_path / "gprof_ir_2020010100.nc").exists()
    with xr.open_dataset(tmp_path / "gprof_ir_2020010100.nc") as results:
        assert "algorithm" in results.attrs
        assert results.attrs["variant"] == variant
        assert results.attrs["n_steps"] == 1


@pytest.mark.parametrize("n_steps", ["3"])
def test_multi_step_retrieval(
        retrieval_input_data,
        tmp_path,
        n_steps
):
    """
    Test running the GPROF IR retrieval.
    """
    args = [
        "gprof_ir",
        "retrieve",
        str(retrieval_input_data / "merg_2020010100_4km-pixel.nc4"),
        "--output_path",
        str(tmp_path),
        "--variant",
        "gmi",
        "--n_steps",
        n_steps
    ]
    subprocess.run(args)
    assert (tmp_path / "gprof_ir_2020010100.nc").exists()
    with xr.open_dataset(tmp_path / "gprof_ir_2020010100.nc") as results:
        assert "algorithm" in results.attrs
        assert results.attrs["variant"] == "gmi"
        assert results.attrs["n_steps"] == int(n_steps)
