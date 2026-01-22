"""
Tests for the gprof_ir.retrieval module.
"""
import subprocess

import numpy as np
import pytest
import torch
import xarray as xr

from gprof_ir.retrieval import MultiInputLoader

HAS_CUDA = torch.cuda.is_available()


def test_input_loader(retrieval_input_data):
    """
    Test loading of input data.
    """
    # Test loading of all files in folder.
    input_loader = MultiInputLoader(
        retrieval_input_data,
        "gmi",
        1,
        input_format="netcdf"
    )
    assert len(input_loader) == 2
    inpt, aux, fname = input_loader[0]
    inpt, aux, fname = next(iter(input_loader))

    # Test loading of a single file.
    input_loader = MultiInputLoader(
        retrieval_input_data / "merg_2020010100_4km-pixel.nc4",
        "gmi",
        1,
        input_format="netcdf"
    )
    assert len(input_loader) == 1
    inpt, aux, fname = input_loader[0]
    inpt, aux, fname = next(iter(input_loader))

    # Test filtering of files by date.
    input_loader = MultiInputLoader(
        retrieval_input_data,
        "gmi",
        1,
        start_time=np.datetime64("2021-01-01"),
        input_format="netcdf"
    )
    assert len(input_loader) == 0

    input_loader = MultiInputLoader(
        retrieval_input_data,
        "gmi",
        1,
        end_time=np.datetime64("2019-12-31"),
        input_format="netcdf"
    )
    assert len(input_loader) == 0


@pytest.mark.parametrize("variant", ["cmb", "gmi"])
def test_retrieve(
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
        str(tmp_path / "output.nc4"),
        "--variant",
        variant
    ]
    subprocess.run(args)
    assert (tmp_path / "output.nc4").exists()
    with xr.open_dataset(tmp_path / "output.nc4") as results:
        assert "algorithm" in results.attrs
        assert results.attrs["variant"] == variant
        assert results.attrs["n_steps"] == 1


@pytest.mark.parametrize("variant", ["gmi"])
def test_retrieve_dummy_suffix(
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
        str(retrieval_input_data / "merg_2020010100_4km-pixel.dummy"),
        str(tmp_path / "output.nc4"),
        "--variant",
        variant,
        "--input_format",
        "netcdf"
    ]
    subprocess.run(args)
    assert (tmp_path / "output.nc4").exists()
    with xr.open_dataset(tmp_path / "output.nc4") as results:
        assert "algorithm" in results.attrs
        assert results.attrs["variant"] == variant
        assert results.attrs["n_steps"] == 1


@pytest.mark.parametrize("variant", ["gmi"])
def test_retrieve_binary_output(
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
        str(tmp_path / "results.nc"),
        "--variant",
        variant,
    ]
    subprocess.run(args)
    assert (tmp_path / "results.nc").exists()
    with xr.open_dataset(tmp_path / "results.nc") as results:
        assert "algorithm" in results.attrs
        assert results.attrs["variant"] == variant
        assert results.attrs["n_steps"] == 1
        sp_ref = results.surface_precip.data
    sp_ref = np.roll(np.flip(sp_ref, 1), sp_ref.shape[-1] // 2, -1)

    args = [
        "gprof_ir",
        "retrieve",
        str(retrieval_input_data / "merg_2020010100_4km-pixel.nc4"),
        str(tmp_path / "results.bin"),
        "--variant",
        variant,
        "--output_format",
        "binary"
    ]
    subprocess.run(args)
    assert (tmp_path / "results.bin").exists()
    sp_flat = np.fromfile(tmp_path / "results.bin", dtype="f4")
    assert np.isclose(sp_ref.flatten(), sp_flat, atol=1e-3).all()


def test_retrieve_multi_step(
        retrieval_input_data,
        tmp_path,
):
    """
    Test running the GPROF IR retrieval.
    """
    args = [
        "gprof_ir",
        "retrieve",
        str(retrieval_input_data / "merg_2020010100_4km-pixel.nc4"),
        str(retrieval_input_data / "merg_2020010101_4km-pixel.nc4"),
        str(tmp_path / "output.nc4"),
        "--variant",
        "gmi"
    ]
    subprocess.run(args)
    assert (tmp_path / "output.nc4").exists()
    with xr.open_dataset(tmp_path / "output.nc4") as results:
        assert "algorithm" in results.attrs
        assert results.attrs["variant"] == "gmi"
        assert results.attrs["n_steps"] == 3


@pytest.mark.skipif(not HAS_CUDA, reason="cuda not available")
def test_retrieve_binary(
        retrieval_input_data_binary,
        tmp_path,
):
    """
    Test running the GPROF IR retrieval.
    """
    for suffix in ["", "gz", "gzip"]:
        if suffix == "":
            fname = "merg_2018010100_4km-pixel"
        else:
            fname = f"merg_2018010100_4km-pixel.{suffix}"
        args = [
            "gprof_ir",
            "retrieve",
            str(retrieval_input_data_binary / fname),
            str(tmp_path / f"output_{suffix}.nc4"),
            "--device",
            "cuda"
        ]
        subprocess.run(args)
        assert (tmp_path / f"output_{suffix}.nc4").exists()
        with xr.open_dataset(tmp_path / f"output_{suffix}.nc4") as results:
            assert "algorithm" in results.attrs
            assert results.attrs["variant"] == "gmi"
            assert results.attrs["n_steps"] == 1

    args = [
        "gprof_ir",
        "retrieve",
        str(retrieval_input_data_binary / ("merg_2018010100_4km-pixel" + suffix)),
        str(tmp_path / "output.nc4"),
        "--input_format",
        "gz",
        "--device",
        "cuda"
    ]
    subprocess.run(args)
    assert not (tmp_path / "output.nc4").exists()


@pytest.mark.skipif(not HAS_CUDA, reason="cuda not available")
def test_retrieve_corrupted_binary(
        retrieval_input_data_binary_corrupted,
        tmp_path,
):
    """
    Test running the GPROF IR retrieval.
    """
    args = [
        "gprof_ir",
        "retrieve",
        str(retrieval_input_data_binary_corrupted / "merg_2018010100_4km-pixel_corr"),
        str(tmp_path / "output.nc4"),
    ]
    proc = subprocess.run(args)
    assert proc.returncode != 0


@pytest.mark.skipif(not HAS_CUDA, reason="cuda not available")
@pytest.mark.parametrize("variant", ["cmb", "gmi"])
def test_run_binary(
        retrieval_input_data_binary,
        tmp_path,
        variant
):
    """
    Test running the GPROF IR retrieval.
    """
    args = [
        "gprof_ir",
        "run",
        str(retrieval_input_data_binary / "merg_2018010100_4km-pixel"),
        "--output_path",
        str(tmp_path),
        "--variant",
        variant,
        "--n_steps",
        "1",
        "--device",
        "cuda:0"
    ]
    subprocess.run(args)
    assert (tmp_path / "gprof_ir_2018010100.nc").exists()
    with xr.open_dataset(tmp_path / "gprof_ir_2018010100.nc") as results:
        assert "algorithm" in results.attrs
        assert results.attrs["variant"] == variant
        assert results.attrs["n_steps"] == 1

@pytest.mark.parametrize("variant", ["cmb", "gmi"])
def test_run(
        retrieval_input_data,
        tmp_path,
        variant
):
    """
    Test running the GPROF IR retrieval.
    """
    args = [
        "gprof_ir",
        "run",
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


@pytest.mark.skipif(not HAS_CUDA, reason="cuda not available")
@pytest.mark.parametrize("variant", ["cmb", "gmi"])
def test_run_binary(
        retrieval_input_data_binary,
        tmp_path,
        variant
):
    """
    Test running the GPROF IR retrieval.
    """
    args = [
        "gprof_ir",
        "run",
        str(retrieval_input_data_binary / "merg_2018010100_4km-pixel"),
        "--output_path",
        str(tmp_path),
        "--variant",
        variant,
        "--n_steps",
        "1",
        "--device",
        "cuda:0"
    ]
    subprocess.run(args)
    assert (tmp_path / "gprof_ir_2018010100.nc").exists()
    with xr.open_dataset(tmp_path / "gprof_ir_2018010100.nc") as results:
        assert "algorithm" in results.attrs
        assert results.attrs["variant"] == variant
        assert results.attrs["n_steps"] == 1


@pytest.mark.parametrize("n_steps", ["3"])
def test_run_multi_step(
        retrieval_input_data,
        tmp_path,
        n_steps
):
    """
    Test running the GPROF IR retrieval.
    """
    args = [
        "gprof_ir",
        "run",
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


@pytest.mark.parametrize("variant", ["gmi"])
def test_run_binary_output(
        retrieval_input_data,
        tmp_path,
        variant
):
    """
    Test running the GPROF IR retrieval.
    """
    args = [
        "gprof_ir",
        "run",
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
        sp_ref = results.surface_precip.data
    sp_ref = np.roll(np.flip(sp_ref, 1), sp_ref.shape[-1] // 2, -1)

    args = [
        "gprof_ir",
        "run",
        str(retrieval_input_data / "merg_2020010100_4km-pixel.nc4"),
        "--output_path",
        str(tmp_path),
        "--variant",
        variant,
        "--n_steps",
        "1",
        "--output_format",
        "binary"
    ]
    subprocess.run(args)
    assert (tmp_path / "gprof_ir_2020010100.bin").exists()
    sp_flat = np.fromfile(tmp_path / "gprof_ir_2020010100.bin", dtype="f4")
    assert np.isclose(sp_ref.flatten(), sp_flat, atol=1e-3).all()
