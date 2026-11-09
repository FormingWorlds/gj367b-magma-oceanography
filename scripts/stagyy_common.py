"""Shared reader and geometry for the GJ 367 b StagYY figure scripts.

The run folders hold StagYY binary output, read here with StagPy. Set the
environment variable ``GJ367B_DATA`` to the folder that holds ``test_NN``
(default ``raw/`` in the repository, where the raw-output archive is unpacked).
"""

import os
from pathlib import Path

import numpy as np

REPO_DIR = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("GJ367B_DATA", REPO_DIR / "raw"))
DATA_DIR = REPO_DIR / "data"
REPLOT_DIR = REPO_DIR / "figures"
MAPS_DIR = Path(os.environ.get("GJ367B_MAPS", DATA_DIR / "surface_maps"))
SEC_PER_GYR = 3600.0 * 24.0 * 365.25 * 1.0e9
SIGMA_SB = 5.670374419e-8
# Model name of each run number (folder test_NN): H or C start, N, E or L melt
# transport, "+" for the 1745 K dayside, "-core" and "-thick" for the variants.
MODEL_NAMES = {
    1: "HN",
    2: "CN",
    4: "CN-thick",
    5: "HE",
    6: "HL",
    7: "CE",
    8: "CL",
    9: "HE+",
    10: "CE+",
    12: "CN+",
    13: "HL+",
    14: "CL+",
    15: "CN-core",
    16: "HN-core",
}
# Models of each melt-transport group, in plotting order.
GROUPS = {
    "N": (1, 2, 4, 16, 15, 12),
    "E": (5, 7, 9, 10),
    "L": (6, 8, 13, 14),
}
GROUP_TITLES = {
    "N": "No eruption",
    "E": "Eruption mode \u201call\u201d",
    "L": "Eruption \u201clith\u201d, segregation",
}


def model_style(run):
    """Return the line style of one model, the same in every figure.

    Parameters
    ----------
    run : int
        Run number (folder ``test_NN``).

    Returns
    -------
    dict
        ``color``, ``ls`` and ``label`` for ``Axes.plot``. The colour gives the
        setup (base, 1745 K dayside, core cooling, thick mantle); a solid line is
        a hot start and a dashed line a cold start.
    """
    import fwl_mpl

    name = MODEL_NAMES[run]
    if name.endswith("-thick"):
        slot = "iris"
    elif name.endswith("-core"):
        slot = "fern"
    elif "+" in name:
        slot = "magma"
    else:
        slot = "signal"
    color = fwl_mpl.PALETTE["light"][slot]
    return {"color": color, "ls": "-" if name[0] == "H" else "--", "label": name}


def run_folder(run):
    """Return the output folder of a paper run.

    Parameters
    ----------
    run : int
        Run number of Table 1; run N is stored in the folder ``test_NN``.

    Returns
    -------
    pathlib.Path
        Folder with the ``par`` file and the ``Output`` folder.

    Raises
    ------
    FileNotFoundError
        If the folder does not exist under ``DATA_ROOT``.
    """
    folder = DATA_ROOT / f"test_{run:02d}"
    if not (folder / "par").is_file():
        raise FileNotFoundError(f"no par file in {folder}; set GJ367B_DATA")
    return folder


def longitude_deg(phi):
    """Convert the StagYY azimuth to longitude from the substellar point.

    Parameters
    ----------
    phi : numpy.ndarray
        Azimuth in radians. The substellar point is at ``phi = pi``.

    Returns
    -------
    numpy.ndarray
        Longitude ``phi - 180`` in degrees, in the range [-180, 180).
    """
    return (np.degrees(phi) - 180.0 + 180.0) % 360.0 - 180.0


def bulk_basalt(step, row=-1):
    """Return the bulk basalt fraction of a cell row, solid and melt together.

    Parameters
    ----------
    step : stagpy.step.Step
        Snapshot.
    row : int, optional
        Radial cell index. The default is the top cell.

    Returns
    -------
    numpy.ndarray
        ``(1 - f) * basalt + f * meltcompo``, with ``f`` the melt fraction.
    """
    melt = np.asarray(step.fields["meltfrac"].values)[0, :, row, 0]
    solid = np.asarray(step.fields["basalt"].values)[0, :, row, 0]
    liquid = np.asarray(step.fields["meltcompo"].values)[0, :, row, 0]
    return (1.0 - melt) * solid + melt * liquid


def annulus_mesh(r_walls, p_walls):
    """Return the cell-corner mesh of the annulus with the star on the left.

    Parameters
    ----------
    r_walls : numpy.ndarray
        Radial cell walls in metres.
    p_walls : numpy.ndarray
        Azimuthal cell walls in radians.

    Returns
    -------
    tuple of numpy.ndarray
        Coordinates ``x`` and ``y`` in km, with shape
        ``(p_walls.size, r_walls.size)``.
    """
    pmesh, rmesh = np.meshgrid(p_walls, r_walls, indexing="ij")
    return rmesh * np.cos(pmesh) / 1.0e3, rmesh * np.sin(pmesh) / 1.0e3


def check_output(path):
    """Refuse an output path inside the run data.

    Parameters
    ----------
    path : pathlib.Path
        File or folder that a script will write.

    Returns
    -------
    pathlib.Path
        The same path.

    Raises
    ------
    ValueError
        If the path lies under ``DATA_ROOT``.
    """
    if path.resolve().is_relative_to(DATA_ROOT.resolve()):
        raise ValueError(f"{path} lies in the run data {DATA_ROOT}")
    return path


def surface_forcing(sdat):
    """Return the substellar and nightside temperatures from the par file of a run.

    Parameters
    ----------
    sdat : stagpy.stagyydata.StagyyData
        Run.

    Returns
    -------
    tuple of float
        Substellar and nightside temperature in K.
    """
    return (
        float(sdat.par.get("boundaries", "topT_locked_subsolar", None)),
        float(sdat.par.get("boundaries", "topT_locked_farside", None)),
    )


def sample_indices(n_items, max_rows):
    """Return evenly spaced indices that always include the last item.

    The result holds at most ``max_rows + 1`` indices: ``max_rows`` evenly spaced
    ones and the last item.

    Parameters
    ----------
    n_items : int
        Number of items.
    max_rows : int
        Maximum number of indices before the last item is appended.

    Returns
    -------
    list of int
        Increasing indices.

    Raises
    ------
    ValueError
        If ``n_items`` is below 1.
    """
    if n_items < 1:
        raise ValueError("need at least 1 item")
    stride = max(1, int(np.ceil(n_items / max_rows)))
    indices = list(range(0, n_items, stride))
    if indices[-1] != n_items - 1:
        indices.append(n_items - 1)
    return indices


def time_cells(time, isnap, stride):
    """Return time edges and the rows after which output is missing.

    Parameters
    ----------
    time : numpy.ndarray
        Row times, increasing.
    isnap : numpy.ndarray
        Snapshot number of each row.
    stride : int
        Snapshot step of the sampling.

    Returns
    -------
    edges : numpy.ndarray
        Pairs ``(low, high)`` per row, with shape ``(time.size, 2)``.
    gaps : numpy.ndarray
        Indices ``i`` where snapshots are missing between rows ``i`` and ``i + 1``.

    Raises
    ------
    ValueError
        If there are fewer than 2 rows, the times do not increase, or every
        interval is a gap.
    """
    if time.size < 2:
        raise ValueError("need at least 2 rows")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("row times must increase")
    steps = np.diff(isnap)
    gaps = np.flatnonzero(steps > stride + 1)
    dt = np.diff(time)
    regular = np.ones(dt.size, dtype=bool)
    regular[gaps] = False
    if not regular.any():
        raise ValueError("every interval is a gap")
    half = 0.5 * dt
    low = np.concatenate(([0.0], time[1:] - half))
    high = np.concatenate((time[:-1] + half, [time[-1] + half[-1]]))
    for i in gaps:
        # Typical spacing from up to 5 regular intervals before the gap, else after it.
        before = dt[max(i - 5, 0) : i][regular[max(i - 5, 0) : i]]
        after = dt[i + 1 : i + 6][regular[i + 1 : i + 6]]
        step = np.median(
            before if before.size else after if after.size else dt[regular]
        )
        middle = 0.5 * (time[i] + time[i + 1])
        high[i] = min(time[i] + 0.5 * step, middle)
        low[i + 1] = max(time[i + 1] - 0.5 * step, middle)
        if i + 1 == time.size - 1:
            high[-1] = time[-1] + 0.5 * step
    return np.column_stack((low, high)), gaps


def conductive_flux(step, t_substellar, t_night):
    """Return the surface heat flux of a snapshot from temperature and conductivity.

    The runs hold no ``hf`` files, so this is a reconstruction. ``near_surface`` is
    the flux between the 2 top cell centres and needs no boundary condition.
    ``half_cell`` solves a radiative balance across the top half cell with the law
    ``T_rad^4 = t_substellar^4 * max(cos(lon), 0) + t_night^4``. The runs fix the
    surface temperature instead (Methods), so ``half_cell`` is a diagnostic that no
    figure or number of the manuscript uses.

    Parameters
    ----------
    step : stagpy.step.Step
        Snapshot.
    t_substellar, t_night : float
        Substellar and nightside temperature in K.

    Returns
    -------
    dict of numpy.ndarray
        Keys ``lon_deg``, ``near_surface`` and ``half_cell`` (W/m2, positive upward)
        and ``t_surface`` (K).
    """
    temp = np.asarray(step.fields["T"].values)[0, :, :, 0]
    cond = np.asarray(step.fields["Tcond"].values)[0, :, :, 0]
    r_c, r_w = step.geom.r_centers, step.geom.r_walls
    upper, lower = r_c[-1] - r_w[-2], r_w[-2] - r_c[-2]
    near = (temp[:, -2] - temp[:, -1]) / (lower / cond[:, -2] + upper / cond[:, -1])
    lon = longitude_deg(step.geom.p_centers)
    dz = r_w[-1] - r_c[-1]
    t_rad4 = t_substellar**4 * np.clip(np.cos(np.radians(lon)), 0.0, None) + t_night**4
    t_surf = temp[:, -1].copy()
    for _ in range(60):
        residual = cond[:, -1] * (temp[:, -1] - t_surf) / dz - SIGMA_SB * (
            t_surf**4 - t_rad4
        )
        t_surf -= residual / (-cond[:, -1] / dz - 4.0 * SIGMA_SB * t_surf**3)
    half = cond[:, -1] * (temp[:, -1] - t_surf) / dz
    return {
        "lon_deg": lon,
        "near_surface": near,
        "half_cell": half,
        "t_surface": t_surf,
    }
