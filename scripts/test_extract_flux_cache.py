"""Tests of the flux cache reader."""

import os
import pickle

import numpy as np
import pytest
from extract_flux_cache import NAMES, read_cache


def write(path, obj):
    """Pickle ``obj`` to ``path`` and return the path."""
    path.write_bytes(pickle.dumps(obj))
    return path


def test_reads_lists_and_numpy_scalars(tmp_path):
    columns = tuple([np.float64(i), float(i) + 1.0] for i in range(len(NAMES)))
    data = read_cache(write(tmp_path / "flux_data.p", columns))
    assert list(data) == list(NAMES)
    assert data["top_day"].tolist() == [2.0, 3.0]


def test_refuses_objects(tmp_path):
    class Payload:
        def __reduce__(self):
            return (os.getcwd, ())

    columns = (Payload(),) + tuple([1.0] for _ in NAMES[1:])
    with pytest.raises(pickle.UnpicklingError, match="refused"):
        read_cache(write(tmp_path / "flux_data.p", columns))


def test_refuses_unequal_columns(tmp_path):
    columns = ([1.0, 2.0],) + tuple([1.0] for _ in NAMES[1:])
    with pytest.raises(ValueError, match="differ in length"):
        read_cache(write(tmp_path / "flux_data.p", columns))


def test_refuses_missing_column(tmp_path):
    columns = tuple([1.0] for _ in NAMES[1:])
    with pytest.raises(ValueError):
        read_cache(write(tmp_path / "flux_data.p", columns))
