import gzip
import os
from pathlib import Path
import shutil

import numpy as np
import pytest
import requests

from huggingface_hub import hf_hub_download

from gprof_ir.config import CONFIG

# Hugging Face repository details
GPROF_IR_HF = "https://huggingface.co/simonpf/gprof_ir"


@pytest.fixture(scope="session")
def retrieval_input_data():
    """
    Fixture to download and cache a test file from a Hugging Face repository.
    The file is stored in a session-scoped temp directory to avoid re-downloading.
    """
    model_path = Path(CONFIG.get("model_path"))
    file_path = hf_hub_download(
        repo_id="simonpf/gprof_ir",
        filename="test_data/merg_2020010100_4km-pixel.nc4",
        local_dir=model_path
    )
    other_file = Path(file_path).parent / "merg_2020010101_4km-pixel.nc4"
    shutil.copy(file_path, other_file)
    dummy_file = Path(file_path).parent / "merg_2020010100_4km-pixel.dummy"
    shutil.copy(file_path, dummy_file)
    return model_path / "test_data"


@pytest.fixture(scope="session")
def retrieval_input_data_binary():
    """
    Fixture to download and cache a test file from a Hugging Face repository.
    The file is stored in a session-scoped temp directory to avoid re-downloading.
    """
    model_path = Path(CONFIG.get("model_path"))
    file_path = hf_hub_download(
        repo_id="simonpf/gprof_ir",
        filename="test_data/merg_2018010100_4km-pixel",
        local_dir=model_path
    )
    file_path = Path(file_path)
    gz_file = file_path.with_suffix(".gz")
    with open(file_path, "rb") as f, gzip.open(gz_file, "wb") as gzfile:
        shutil.copyfileobj(f, gzfile)
    gzip_file = gz_file.with_suffix(".gzip")
    shutil.copy(gz_file, gzip_file)
    return model_path / "test_data"


@pytest.fixture(scope="session")
def retrieval_input_data_binary_corrupted():
    """
    Fixture to download and cache a test file from a Hugging Face repository.
    The file is stored in a session-scoped temp directory to avoid re-downloading.
    """
    model_path = Path(CONFIG.get("model_path"))
    input_file = model_path / "test_data/merg_2018010100_4km-pixel"
    output_file = model_path / "test_data/merg_2018010100_4km-pixel_corr"
    np.memmap(
        input_file, dtype=np.float32, mode="r"
    )[:(os.path.getsize(input_file) // np.dtype(np.float32).itemsize) // 2].tofile(output_file)
    return model_path / "test_data"
