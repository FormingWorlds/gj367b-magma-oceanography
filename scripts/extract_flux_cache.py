"""Convert the ``flux_data.p`` cache of each run to a compact file.

The caches are in ``testNN/flux_data.p`` of the raw-output archive. Each one holds
the median surface flux and the mean CMB flux at each ``hf`` output, in mW/m2.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
from numpy._core.multiarray import scalar as np_scalar
from stagyy_common import DATA_DIR

NAMES = (
    "time_gyr",
    "top_all",
    "top_day",
    "top_night",
    "bot_all",
    "bot_day",
    "bot_night",
)
RUNS = (1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16)
NUMPY_MODULES = ("numpy.core.multiarray", "numpy._core.multiarray")


def read_cache(path):
    """Read one cache and refuse any pickled object except NumPy numbers.

    Parameters
    ----------
    path : pathlib.Path
        Path of ``flux_data.p``.

    Returns
    -------
    dict of str to numpy.ndarray
        One float array per column of the cache.

    Raises
    ------
    ValueError
        If a column is not 1-D, is empty, or the columns differ in length.
    """

    class NumbersOnly(pickle.Unpickler):
        def find_class(self, module, name):
            if module in NUMPY_MODULES and name == "scalar":
                return np_scalar
            if module == "numpy" and name == "dtype":
                return np.dtype
            raise pickle.UnpicklingError(f"refused {module}.{name}")

    with path.open("rb") as stream:
        columns = NumbersOnly(stream).load()
    out = {
        name: np.asarray(col, dtype=float)
        for name, col in zip(NAMES, columns, strict=True)
    }
    if any(col.ndim != 1 or col.size == 0 for col in out.values()):
        raise ValueError(f"{path}: every column must be 1-D and non-empty")
    if len({col.size for col in out.values()}) != 1:
        raise ValueError(f"{path}: columns differ in length")
    return out


def main(source):
    """Write ``data/stagyy_flux/run_NN.npz`` for each run.

    Parameters
    ----------
    source : pathlib.Path
        Folder that holds ``testNN/flux_data.p``.
    """
    target = DATA_DIR / "stagyy_flux"
    target.mkdir(exist_ok=True)
    for run in RUNS:
        data = read_cache(source / f"test{run:02d}" / "flux_data.p")
        np.savez_compressed(target / f"run_{run:02d}.npz", **data)
        print(
            f"run {run}: {data['time_gyr'].size} rows to {data['time_gyr'][-1]:.3f} Gyr"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    main(parser.parse_args().source)
