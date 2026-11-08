"""Plot the overview figures of the GJ 367 b StagYY models.

The figures group the 14 models by melt transport (no eruption, eruption mode
"all", eruption mode "lith" with melt segregation):

- ``flux``: dayside heat flux of the groups without segregation, without the
  mostly molten outputs, and nightside heat flux of all models.
- ``surface``: surface bulk basalt fraction against time and longitude, with
  the melt zone as contour lines, and below each map the mean basalt fraction
  in 4 longitude regions on the same time axis.
- ``profiles``: surface basalt and melt fraction against longitude, averaged
  over the last quarter of each run.
- ``interior``: mean mantle temperature and core-mantle boundary temperature.
"""

import argparse
from pathlib import Path

import fwl_mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from scipy.ndimage import median_filter, uniform_filter
from stagyy_common import (
    DATA_DIR,
    GROUP_TITLES,
    GROUPS,
    MAPS_DIR,
    MODEL_NAMES,
    REPLOT_DIR,
    model_style,
    time_cells,
)

REGIONS = (
    (r"Substellar, $|\lambda| < 30\degree$", lambda lon: np.abs(lon) < 30.0),
    (
        r"Dayside, $30\degree < |\lambda| < 90\degree$",
        lambda lon: (np.abs(lon) > 30.0) & (np.abs(lon) < 90.0),
    ),
    (r"Nightside west, $\lambda < -90\degree$", lambda lon: lon < -90.0),
    (r"Nightside east, $\lambda > 90\degree$", lambda lon: lon > 90.0),
)
REGION_COLORS = ("magma", "ochre", "signal", "iris")
# The east nightside is dashed so that the two nightside halves differ in grey print too.
REGION_STYLES = ("-", "-", "-", "--")
MELT_LEVELS = (0.05, 0.5)
LATE_FRACTION = 0.75
# Running-mean width of the basalt profiles in longitude cells (0.70 deg each).
PROFILE_SMOOTH = 9
# HN and CN: the dayside ftop output disagrees with their temperature field and
# with the global surface flux of time.dat, so their dayside flux is not plotted.
DAY_FLUX_EXCLUDED = (1, 2)
# Dayside outputs where at least MOLTEN_SHARE of the dayside surface cells have a
# melt fraction above MOLTEN_MELT are not plotted: with a fixed surface temperature
# the flux written for molten top cells does not close the energy budget.
MOLTEN_MELT = 0.3
MOLTEN_SHARE = 0.5


def load_map(run):
    """Return the stored surface map of one run, sorted by longitude.

    Parameters
    ----------
    run : int
        Run number (folder ``test_NN``).

    Returns
    -------
    dict
        ``time_gyr``, ``isnap``, ``stride``, ``lon_deg`` (increasing),
        ``basalt`` and ``meltfrac`` (time by longitude).
    """
    path = MAPS_DIR / f"surface_map_run{run:02d}.npz"
    if not path.is_file():
        raise SystemExit(
            f"{path} is missing: set GJ367B_MAPS or run extract_surface_maps.py {run}"
        )
    data = np.load(path)
    order = np.argsort(data["lon_deg"])
    return {
        "time_gyr": data["time_gyr"],
        "isnap": data["isnap"],
        "stride": int(data["stride"]),
        "lon_deg": data["lon_deg"][order].astype(float),
        "basalt": data["basalt"][:, order],
        "meltfrac": data["meltfrac"][:, order],
    }


def dayside_molten_share(surface_map, time_gyr):
    """Return the share of dayside surface cells above MOLTEN_MELT at given times.

    Parameters
    ----------
    surface_map : dict
        Output of :func:`load_map`.
    time_gyr : numpy.ndarray
        Times (Gyr) at which the share is wanted; each takes the next map row
        at or after it, or the last row.

    Returns
    -------
    numpy.ndarray
        Share of the cells with ``abs(lon) < 90`` deg, one value per time.
    """
    day = np.abs(surface_map["lon_deg"]) < 90.0
    share = (surface_map["meltfrac"][:, day] > MOLTEN_MELT).mean(axis=1)
    rows = np.searchsorted(surface_map["time_gyr"], time_gyr)
    return share[np.clip(rows, 0, len(share) - 1)]


def despike(values, factor=2.0):
    """Replace single outputs that differ from their neighbours by more than ``factor``.

    Parameters
    ----------
    values : numpy.ndarray
        Time series.
    factor : float, optional
        An output is replaced by the median of itself and its 2 neighbours when it
        is more than ``factor`` times above or below that median.

    Returns
    -------
    numpy.ndarray
        Copy of ``values`` with the spikes replaced.
    """
    med = median_filter(values, size=3, mode="nearest")
    spike = (values > factor * med) | (values < med / factor)
    return np.where(spike, med, values)


def flux_figure(out):
    """Draw the dayside heat flux of the groups without segregation and the nightside flux.

    Dayside outputs with a mostly molten dayside surface are left out
    (``dayside_molten_share``); in the segregation group that is every output, so
    the group has no dayside panel. The nightside panel has logarithmic axes, shows
    only outputs after t = 0, and replaces single-output spikes (``despike``).

    Parameters
    ----------
    out : pathlib.Path
        Output file.
    """
    fig = plt.figure(figsize=(7.1, 2.7), layout="constrained")
    grid = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 1.1))
    day_groups = ("N", "E")
    day_axes = [fig.add_subplot(grid[0, 0])]
    day_axes.append(fig.add_subplot(grid[0, 1], sharey=day_axes[0]))
    for col, (ax, group) in enumerate(zip(day_axes, day_groups, strict=True)):
        for run in GROUPS[group]:
            if run in DAY_FLUX_EXCLUDED:
                continue
            data = np.load(DATA_DIR / "stagyy_flux" / f"run_{run:02d}.npz")
            share = dayside_molten_share(load_map(run), data["time_gyr"])
            day = np.where(share < MOLTEN_SHARE, data["top_day"], np.nan)
            ax.plot(data["time_gyr"], day, lw=1.0, **model_style(run))
        # The first cold-start outputs are negative and lie below the axis.
        ax.axhline(0.0, color="0.6", lw=0.5, zorder=0)
        ax.set_ylim(-20.0, 100.0)
        ax.set_yticks(range(-20, 101, 20))
        ax.set_title(GROUP_TITLES[group], loc="left", fontsize=8)
        ax.set_xlim(0.0, 8.0)
        ax.set_xticks(range(0, 9, 2))
        ax.set_xlabel("Time (Gyr)")
        if col > 0:
            ax.tick_params(labelleft=False)
    day_axes[0].set_ylabel(r"Dayside heat flux (mW/m$^2$)")
    night = fig.add_subplot(grid[0, 2])
    for runs in GROUPS.values():
        for run in runs:
            data = np.load(DATA_DIR / "stagyy_flux" / f"run_{run:02d}.npz")
            later = data["time_gyr"] > 0.0
            style = model_style(run)
            night.plot(
                data["time_gyr"][later],
                despike(data["top_night"][later]),
                lw=0.8,
                color=style["color"],
                ls=style["ls"],
            )
    night.set_xscale("log")
    night.set_yscale("log")
    night.set_xlim(1.0e-3, 8.0)
    night.set_ylim(20.0, 2000.0)
    night_ticks = (20, 50, 100, 200, 500, 1000, 2000)
    night.set_yticks(night_ticks)
    night.set_yticklabels([str(tick) for tick in night_ticks])
    night.yaxis.set_minor_formatter(plt.NullFormatter())
    night.set_title("Nightside, all models", loc="left", fontsize=8)
    night.set_ylabel(r"Nightside heat flux (mW/m$^2$)")
    night.set_xlabel("Time (Gyr)")
    for ax, label in zip([*day_axes, night], "abc", strict=True):
        fwl_mpl.panel_label(ax, label)
    fig.legend(
        handles=setup_key(),
        loc="outside lower center",
        ncols=6,
        frameon=False,
        fontsize=7,
        handlelength=2.0,
        columnspacing=1.2,
    )
    save(fig, out)


def fill_gaps(y, rows, values, n_sub=20):
    """Replace missing output intervals by a linear interpolation in time.

    Parameters
    ----------
    y : numpy.ndarray
        Time edges of the rows (Gyr), one more than ``rows``.
    rows : numpy.ndarray
        Index into ``values`` for each row; -1 marks a missing interval.
    values : numpy.ndarray
        Stored rows (time by longitude).

    Returns
    -------
    tuple of numpy.ndarray
        New time edges and rows, with each missing interval split into
        ``n_sub`` rows that blend linearly from the row before to the row after.
    """
    edges, out = [y[0]], []
    for k, row in enumerate(rows):
        if row >= 0:
            out.append(values[row])
            edges.append(y[k + 1])
            continue
        before, after = values[rows[k - 1]], values[rows[k + 1]]
        sub = np.linspace(y[k], y[k + 1], n_sub + 1)
        for j in range(n_sub):
            w = (j + 0.5) / n_sub
            out.append((1.0 - w) * before + w * after)
            edges.append(sub[j + 1])
    return np.array(edges), np.array(out)


def map_ticks(t_end):
    """Return the upper time limit and the time ticks of a map with time across.

    Parameters
    ----------
    t_end : float
        End of the last output cell (Gyr).

    Returns
    -------
    tuple of (float, list of float)
        The end rounded to 0.01 Gyr, and ticks from 0 that end on it, spaced so
        that the labels of a narrow panel do not touch.
    """
    top = round(float(t_end), 2)
    step = 4.0 if top > 4.0 else 1.0 if top > 1.5 else 0.5
    ticks = [t for t in np.arange(0.0, top, step) if top - t > 0.4 * step]
    return top, ticks + [top]


def surface_figure(out):
    """Draw the surface basalt maps with the regional basalt series below each.

    Each model has a map of bulk basalt fraction against time and longitude,
    with melt contours, and below it the mean basalt fraction of 4 longitude
    regions on the same time axis.

    Parameters
    ----------
    out : pathlib.Path
        Output file.
    """
    fig = plt.figure(figsize=(7.1, 6.6), layout="constrained")
    grid = fig.add_gridspec(6, 6, height_ratios=[1.25, 0.75] * 3)
    ink = fwl_mpl.SURFACES["light"]["ink"]
    for r, runs in enumerate(GROUPS.values()):
        for k, run in enumerate(runs):
            ax_map = fig.add_subplot(grid[2 * r, k])
            ax_series = fig.add_subplot(grid[2 * r + 1, k], sharex=ax_map)
            data = load_map(run)
            lon = data["lon_deg"]
            lon_edges = np.concatenate(([-180.0], 0.5 * (lon[1:] + lon[:-1]), [180.0]))
            cells, gaps = time_cells(data["time_gyr"], data["isnap"], data["stride"])
            t_edges = np.append(cells[:, 0], cells[-1, 1])
            rows = np.arange(cells.shape[0])
            for i in gaps[::-1]:
                t_edges = np.insert(t_edges, i + 1, cells[i, 1])
                rows = np.insert(rows, i + 1, -1)
            t_edges, basalt = fill_gaps(t_edges, rows, data["basalt"])
            mesh = ax_map.pcolormesh(
                t_edges,
                lon_edges,
                basalt.T,
                cmap="cmc.bam_r",
                vmin=0.0,
                vmax=1.0,
                rasterized=True,
            )
            centres = 0.5 * (cells[:, 0] + cells[:, 1])
            # Smooth over 25 outputs and 9 cells so the contours show the melt zone, not cell noise.
            melt = uniform_filter(
                data["meltfrac"].astype(float), size=(25, 9), mode="nearest"
            )
            contours = ax_map.contour(
                centres,
                lon,
                melt.T,
                levels=MELT_LEVELS,
                colors="white",
                linewidths=(0.8, 1.6),
            )
            contours.set_path_effects(
                [patheffects.withStroke(linewidth=2.6, foreground=ink)]
            )
            for edge in (-90.0, 90.0):
                ax_map.axhline(edge, color=ink, lw=0.4, ls=":")
            top, ticks = map_ticks(t_edges[-1])
            ax_map.set_xlim(0.0, top)
            ax_map.set_xticks(ticks)
            ax_map.set_ylim(-180.0, 180.0)
            ax_map.set_yticks((-180, 0, 180))
            ax_map.tick_params(labelbottom=False)
            ax_map.set_title(MODEL_NAMES[run], loc="left", fontweight="bold", pad=2)
            for (label, select), slot, style in zip(
                REGIONS, REGION_COLORS, REGION_STYLES, strict=True
            ):
                values = data["basalt"][:, select(lon)].mean(axis=1)
                ax_series.plot(
                    data["time_gyr"],
                    values,
                    lw=0.8,
                    ls=style,
                    color=fwl_mpl.PALETTE["light"][slot],
                    label=label,
                )
            ax_series.set_ylim(0.0, 1.0)
            ax_series.set_yticks((0.0, 0.5, 1.0))
            if k == 0:
                ax_map.set_ylabel(r"Longitude ($\degree$)")
                ax_series.set_ylabel("Basalt")
            else:
                ax_map.tick_params(labelleft=False)
                ax_series.tick_params(labelleft=False)
    handles, labels = ax_series.get_legend_handles_labels()
    ax_legend = fig.add_subplot(grid[2:4, 4:])
    ax_legend.set_axis_off()
    ax_legend.legend(
        handles,
        labels,
        loc="center",
        frameon=False,
        title="Lower panels: mean basalt fraction",
    )
    ax_bar = fig.add_subplot(grid[4, 4:])
    ax_bar.set_axis_off()
    fig.colorbar(
        mesh,
        cax=ax_bar.inset_axes([0.05, 0.35, 0.9, 0.15]),
        orientation="horizontal",
        label="Bulk basalt fraction (maps)",
    )
    fig.supxlabel("Time (Gyr)")
    save(fig, out, dpi=400)


def setup_key():
    """Return legend handles for the setup colours and the start line styles."""
    pal = fwl_mpl.PALETTE["light"]
    entries = (
        (pal["signal"], "-", r"$T_\mathrm{day}$ 1597 K"),
        (pal["magma"], "-", r"$T_\mathrm{day}$ 1745 K (+)"),
        (pal["fern"], "-", "core cooling"),
        (pal["iris"], "-", "thick mantle"),
        ("0.3", "-", "hot start"),
        ("0.3", "--", "cold start"),
    )
    return [
        plt.Line2D([], [], color=c, ls=ls, lw=1.4, label=lab) for c, ls, lab in entries
    ]


def profiles_figure(out):
    """Draw surface basalt and melt fraction against longitude, late in each run.

    The basalt curves are smoothed with a running mean over ``PROFILE_SMOOTH``
    longitude cells, wrapping at 180 degrees; the melt curves are not smoothed.

    Parameters
    ----------
    out : pathlib.Path
        Output file.
    """
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(7.1, 3.1),
        sharex=True,
        sharey="row",
        layout="constrained",
        gridspec_kw={"height_ratios": (1.0, 0.38)},
    )
    for col, (group, runs) in enumerate(GROUPS.items()):
        for run in runs:
            data = load_map(run)
            late = data["time_gyr"] >= LATE_FRACTION * data["time_gyr"][-1]
            style = model_style(run)
            basalt = uniform_filter(
                data["basalt"][late].mean(axis=0), size=PROFILE_SMOOTH, mode="wrap"
            )
            axes[0, col].plot(data["lon_deg"], basalt, lw=1.0, **style)
            axes[1, col].plot(
                data["lon_deg"],
                data["meltfrac"][late].mean(axis=0),
                lw=1.0,
                color=style["color"],
                ls=style["ls"],
            )
        axes[0, col].set_title(GROUP_TITLES[group], loc="left", fontsize=8)
    for ax in axes.flat:
        ax.set_xlim(-180.0, 180.0)
        ax.set_xticks((-180, -90, 0, 90, 180))
        for lon in (-90.0, 90.0):
            ax.axvline(lon, color="0.75", lw=0.5, ls=":", zorder=0)
    axes[0, 0].set_ylim(0.0, 1.0)
    axes[1, 0].set_ylim(0.0, 1.0)
    axes[1, 0].set_yticks((0.0, 0.5, 1.0))
    axes[0, 0].set_ylabel("Bulk basalt fraction")
    axes[1, 0].set_ylabel("Melt\nfraction")
    for ax in axes[1]:
        ax.set_xlabel(r"Longitude ($\degree$)")
    for ax, label in zip(axes.flat, "abcdef", strict=True):
        fwl_mpl.panel_label(ax, label)
    fig.legend(
        handles=setup_key(),
        loc="outside lower center",
        ncols=6,
        frameon=False,
        fontsize=7,
        handlelength=2.0,
        columnspacing=1.2,
    )
    save(fig, out)


def interior_figure(out):
    """Draw the mean mantle temperature and the core-mantle boundary temperature.

    Parameters
    ----------
    out : pathlib.Path
        Output file.
    """
    fig, axes = plt.subplots(
        1, 2, figsize=(7.1, 2.9), sharex=True, layout="constrained"
    )
    for runs in GROUPS.values():
        for run in runs:
            data = np.load(DATA_DIR / "stagyy_1d" / f"run_{run:02d}.npz")
            style = model_style(run)
            axes[0].plot(data["ts_time_gyr"], data["ts_Tmean"], lw=0.8, **style)
            axes[1].plot(
                data["ts_time_gyr"],
                data["ts_Tcmb"],
                lw=0.8,
                color=style["color"],
                ls=style["ls"],
            )
    axes[0].set_ylabel("Mean mantle temperature (K)")
    axes[1].set_ylabel("Core-mantle boundary temperature (K)")
    for ax, label in zip(axes, "ab", strict=True):
        ax.set_xlabel("Time (Gyr)")
        ax.set_xlim(0.0, 8.0)
        fwl_mpl.panel_label(ax, label)
    # Name the 2 segregation models that the text discusses; their colour repeats.
    for run, name, va in ((6, "HL", "bottom"), (8, "CL", "top")):
        data = np.load(DATA_DIR / "stagyy_1d" / f"run_{run:02d}.npz")
        for ax, key in zip(axes, ("ts_Tmean", "ts_Tcmb"), strict=True):
            ax.text(
                8.1,
                data[key][-1],
                name,
                va=va,
                ha="left",
                fontsize=6,
                color=model_style(run)["color"],
                clip_on=False,
            )
    axes[0].set_ylim(1300.0, 2500.0)
    axes[0].set_yticks(range(1300, 2600, 400))
    axes[1].set_ylim(1500.0, 2500.0)
    axes[1].set_yticks(range(1500, 2600, 250))
    fig.legend(
        handles=setup_key(),
        loc="outside lower center",
        ncols=6,
        frameon=False,
        fontsize=7,
        handlelength=2.0,
        columnspacing=1.2,
    )
    save(fig, out)


def save(fig, out, dpi=None):
    """Apply the layout rules, print what remains wrong, and save.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    out : pathlib.Path
    dpi : float, optional
        Resolution of rasterized parts.
    """
    for line in fwl_mpl.finish(fig, dpi=dpi):
        print(f"layout {out.name}:", line)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print("wrote", out)


FIGURES = {
    "flux": flux_figure,
    "surface": surface_figure,
    "profiles": profiles_figure,
    "interior": interior_figure,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "names", nargs="*", choices=sorted(FIGURES), default=sorted(FIGURES)
    )
    parser.add_argument("--out-dir", type=Path, default=REPLOT_DIR)
    args = parser.parse_args()
    fwl_mpl.use()
    for name in args.names or sorted(FIGURES):
        FIGURES[name](args.out_dir / f"fig_{name}.pdf")
