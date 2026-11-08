"""Store time series, radial profiles and surface flux of one run as a compact file."""

import argparse

import numpy as np
from stagpy.error import InvalidSnapshotError, MissingDataError
from stagpy.stagyydata import StagyyData
from stagyy_common import (
    DATA_DIR,
    SEC_PER_GYR,
    bulk_basalt,
    check_output,
    conductive_flux,
    longitude_deg,
    run_folder,
    sample_indices,
    surface_forcing,
)

SERIES = (
    "F_top",
    "F_bot",
    "Tmean",
    "Tmin",
    "Tmax",
    "Vrms",
    "eta_mean",
    "erupt_rate",
    "Tcmb",
)


def main(run, max_flux_rows):
    """Read the run and write ``data/stagyy_1d/run_NN.npz``.

    Parameters
    ----------
    run : int
        Run number of Table 1.
    max_flux_rows : int
        Maximum number of snapshots in the flux series.
    """
    sdat = StagyyData(run_folder(run))
    out = {}
    t_substellar, t_night = surface_forcing(sdat)
    out["t_substellar"], out["t_night"] = t_substellar, t_night
    series = sdat.tseries
    picks = sample_indices(series.time.size, 5000)
    out["ts_time_gyr"] = (series.time / SEC_PER_GYR)[picks].astype(np.float32)
    for name in SERIES:
        out[f"ts_{name}"] = series[name].values[picks].astype(np.float32)
    step = sdat.snaps[-1]
    lon = longitude_deg(step.geom.p_centers)
    day = np.abs(lon) < 90.0
    out["last_time_gyr"] = step.time / SEC_PER_GYR
    out["r_centers_km"] = step.geom.r_centers / 1.0e3
    out["r_top_km"] = step.geom.r_walls[-1] / 1.0e3
    out["lon_deg"] = lon.astype(np.float32)
    for name in ("T", "eta"):
        field = np.asarray(step.fields[name].values)[0, :, :, 0]
        log = name == "eta"
        for key, mask in (("day", day), ("night", ~day)):
            profile = (
                np.log10(field[mask]).mean(axis=0) if log else field[mask].mean(axis=0)
            )
            out[f"prof_{name}_{key}"] = profile.astype(np.float32)
    out["prof_rprof_Tmean"] = np.asarray(step.rprofs["Tmean"].values, dtype=np.float32)
    out["surf_T"] = np.asarray(step.fields["T"].values)[0, :, -1, 0].astype(np.float32)
    out["surf_bulk_basalt"] = bulk_basalt(step).astype(np.float32)
    out["surf_melt"] = np.asarray(step.fields["meltfrac"].values)[0, :, -1, 0].astype(
        np.float32
    )
    columns = (
        "time_gyr",
        "near_day_med",
        "near_night_med",
        "near_day_mean",
        "near_night_mean",
        "half_day_med",
        "half_night_med",
        "half_day_mean",
        "half_night_mean",
    )
    rows, skipped = [], []
    for isnap in sample_indices(len(sdat.snaps), max_flux_rows):
        try:
            snap = sdat.snaps[isnap]
            flux = conductive_flux(snap, t_substellar, t_night)
        except (InvalidSnapshotError, MissingDataError):
            skipped.append(isnap)
            continue
        mask = np.abs(flux["lon_deg"]) < 90.0
        near, half = flux["near_surface"], flux["half_cell"]
        rows.append(
            (
                snap.time / SEC_PER_GYR,
                np.median(near[mask]),
                np.median(near[~mask]),
                np.mean(near[mask]),
                np.mean(near[~mask]),
                np.median(half[mask]),
                np.median(half[~mask]),
                np.mean(half[mask]),
                np.mean(half[~mask]),
            )
        )
    table = np.asarray(rows, dtype=np.float32)
    for k, name in enumerate(columns):
        out[f"flux_{name}"] = table[:, k]
    out["flux_skipped_isnap"] = np.asarray(skipped, dtype=np.int64)
    folder = DATA_DIR / "stagyy_1d"
    check_output(folder).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / f"run_{run:02d}.npz", **out)
    print(
        f"run {run}: {len(rows)} flux rows, skipped {len(skipped)}, T_sub {t_substellar}, T_night {t_night}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=int)
    parser.add_argument("--max-flux-rows", type=int, default=400)
    args = parser.parse_args()
    main(args.run, args.max_flux_rows)
