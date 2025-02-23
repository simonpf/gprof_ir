"""
Tests for the gprof_ir.data.gpm module.
"""
import os

import numpy as np
import pytest

from chimp.utils import get_date
from chimp import areas

from gprof_ir.data.gpm import gpm


HAS_PANSAT_PASSWORD = "PANSAT_PASSWORD" in os.environ
NEEDS_PANSAT_PASSWORD = pytest.mark.skipif(not HAS_PANSAT_PASSWORD, reason="Needs pansat password.")


@NEEDS_PANSAT_PASSWORD
def test_find_files():
    """
    Test that CPCIR input files for a given date are found.
    """
    recs = gpm.find_files(
        np.datetime64("2020-01-01"),
        np.datetime64("2020-01-01T23:43:00"),
        np.timedelta64(30, "m")
    )
    assert len(recs) > 10

@NEEDS_PANSAT_PASSWORD
def test_process_files(tmp_path):
    """
    Ensure that the expected input observation files are created.
    """
    recs = gpm.find_files(
        np.datetime64("2020-01-01"),
        np.datetime64("2020-01-01"),
        np.timedelta64(30, "m")
    )
    assert len(recs) == 1
    gpm.process_file(recs[0], areas.CPCIR, tmp_path, np.timedelta64(30, "m"))

    files = sorted(list((tmp_path / "gpm").glob("*.nc")))
    assert len(files) == 4

    dates = [get_date(path) for path in files]
    assert dates[2] == np.datetime64("2020-01-01T00:00:00")
