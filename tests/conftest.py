import pytest
import requests
import hashlib
from pathlib import Path

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
    return model_path / "test_data"
