"""Store surface bulk basalt and melt fraction against longitude and time for one run.

The output is one ``surface_map_runNN.npz`` per run in ``GJ367B_MAPS``, default
``figures/data/surface_maps``. Snapshots with a 0-byte field file are skipped and
counted, and their isnap values are stored in the output.
"""

import argparse
from pathlib import Path

import numpy as np
from stagpy.stagyydata import StagyyData
from stagyy_common import (
    MAPS_DIR,
    SEC_PER_GYR,
    bulk_basalt,
    check_output,
    longitude_deg,
    run_folder,
    sample_indices,
)


def main(run, max_rows):
    """Read the top cell row of each sampled snapshot and write the map.

    Parameters
    ----------
    run : int
        Run number of Table 1.
    max_rows : int
        Maximum number of sampled snapshots.
    """
    folder = run_folder(run)
    sdat = StagyyData(folder)
    n_snaps = len(sdat.snaps)
    sampled = sample_indices(n_snaps, max_rows)
    times, basalt, melt, kept, skipped = [], [], [], [], []
    stem = folder / "Output" / "GJ367b"
    step = None
    for isnap in sampled:
        paths = [Path(f"{stem}_{tag}{isnap:05d}") for tag in ("bs", "f", "fc")]
        if any(not p.is_file() or p.stat().st_size == 0 for p in paths):
            skipped.append(isnap)
            continue
        step = sdat.snaps[isnap]
        kept.append(isnap)
        times.append(step.time / SEC_PER_GYR)
        basalt.append(bulk_basalt(step))
        melt.append(np.asarray(step.fields["meltfrac"].values)[0, :, -1, 0])
    if step is None:
        raise SystemExit(f"run {run}: no readable snapshot under {stem}")
    lon = longitude_deg(step.geom.p_centers)
    order = np.argsort(lon)
    check_output(MAPS_DIR).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        MAPS_DIR / f"surface_map_run{run:02d}.npz",
        time_gyr=np.asarray(times, dtype=np.float64),
        lon_deg=lon[order].astype(np.float32),
        basalt=np.asarray(basalt, dtype=np.float32)[:, order],
        meltfrac=np.asarray(melt, dtype=np.float32)[:, order],
        isnap=np.asarray(kept, dtype=np.int64),
        n_snaps=n_snaps,
        stride=sampled[1] - sampled[0],
        skipped_isnap=np.asarray(skipped, dtype=np.int64),
    )
    print(
        f"run {run}: {len(times)} rows of {n_snaps} snapshots, skipped {len(skipped)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=int)
    parser.add_argument("--max-rows", type=int, default=2000)
    args = parser.parse_args()
    main(args.run, args.max_rows)
