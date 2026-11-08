"""Store the last snapshot of every paper run for Figure 2 as a compact file.

Temperature and bulk basalt fraction are stored as float16 and viscosity as
float16 ``log10``, enough for a colour map. Runs 13 and 14 end before 8 Gyr;
their end time is stored with the fields.
"""

import numpy as np
from stagpy.stagyydata import StagyyData
from stagyy_common import (
    DATA_DIR,
    MODEL_NAMES,
    SEC_PER_GYR,
    bulk_basalt,
    check_output,
    run_folder,
)


def main():
    """Read the snapshots and write ``data/stagyy_fig02_annulus.npz``."""
    out = {}
    for run in sorted(MODEL_NAMES):
        step = StagyyData(run_folder(run)).snaps[-1]
        out[f"run{run}_time_gyr"] = step.time / SEC_PER_GYR
        out[f"run{run}_isnap"] = step.isnap
        out[f"run{run}_r_walls"] = step.geom.r_walls
        out[f"run{run}_p_walls"] = step.geom.p_walls
        temp = np.asarray(step.fields["T"].values)[0, :, :, 0]
        eta = np.asarray(step.fields["eta"].values)[0, :, :, 0]
        out[f"run{run}_T"] = temp.astype(np.float16)
        out[f"run{run}_log_eta"] = np.log10(eta).astype(np.float16)
        rows = range(step.geom.r_centers.size)
        out[f"run{run}_bulk_basalt"] = np.stack(
            [bulk_basalt(step, row) for row in rows], axis=1
        ).astype(np.float16)
        print(f"run {run}: snapshot {step.isnap}, {step.time / SEC_PER_GYR:.3f} Gyr")
    np.savez_compressed(check_output(DATA_DIR / "stagyy_fig02_annulus.npz"), **out)


if __name__ == "__main__":
    main()
