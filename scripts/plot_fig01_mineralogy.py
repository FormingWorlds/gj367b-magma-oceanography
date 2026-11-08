"""Build Figure 1 of the GJ 367 b paper: rock type and mineralogy.

The figure has a cross layout. The central panel shows the IUGS rock-type
classification as a function of bulk basalt fraction (0 = harzburgite,
1 = basalt) and temperature. The four arms show the volume fractions of the
individual mineral phases along the edges of that map:

- top    : mineralogy versus basalt fraction at the high-temperature slice
- bottom : mineralogy versus basalt fraction at the low-temperature slice
- left   : mineralogy versus temperature for the pure harzburgite end-member
- right  : mineralogy versus temperature for the pure basalt end-member

Mineral assemblages come from a Perple_X phase-equilibrium grid supplied as a
single CSV (columns: ``f_bas``, ``T``, one column per mineral volume percent,
and ``Rock type``).

Usage
-----
python plot_fig01_mineralogy.py [--csv PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import fwl_mpl
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.gridspec import GridSpec

# Mineral phases in stacking order (mafic phases first, then the feldspar,
# then the minor aluminous low-temperature phases). The order is shared by
# every mineralogy panel so the colours read consistently across the figure.
MINERALS = [
    "Orthopyroxene",
    "Olivine",
    "Clinopyroxene",
    "Plagioclase",
    "Corundum",
    "Cordierite",
    "Kyanite",
    "Quartz",
]

# Lab palette, one colour per mineral, distinct under colour-vision deficiency.
MINERAL_COLORS = dict(zip(MINERALS, fwl_mpl.CYCLE, strict=True))

# Rock types ordered along the harzburgite -> basalt progression, each with a
# fill colour for the classification map.
ROCK_TYPES = ["Harzburgite", "Lherzolite", "Norite", "Gabbro"]
# Sequential map sampled in order, since the rock types follow the basalt fraction.
ROCK_COLORS = [mpl.colormaps["cmc.batlow"](v) for v in (0.92, 0.68, 0.42, 0.18)]


def load_grid(csv_path: Path) -> pd.DataFrame:
    """Read the Perple_X mineralogy grid and drop all-zero mineral columns.

    Parameters
    ----------
    csv_path : Path
        Path to the StagYY/Perple_X mineralogy CSV.

    Returns
    -------
    pandas.DataFrame
        The grid with ``f_bas`` and ``T`` as floats. Mineral columns that are
        identically zero across the whole grid (e.g. Spinel) are removed.
    """
    df = pd.read_csv(csv_path)
    df["f_bas"] = df["f_bas"].astype(float)
    df["T"] = df["T"].astype(float)
    for m in list(MINERALS):
        if m in df.columns and float(df[m].abs().max()) == 0.0:
            MINERALS.remove(m)
    return df


def _present_minerals(df: pd.DataFrame) -> list[str]:
    """Return the stacking-ordered minerals that exist as columns in ``df``."""
    return [m for m in MINERALS if m in df.columns]


def stack_vertical(ax, x, df_slice, minerals, label):
    """Draw a vertically stacked area plot (volume fraction on the y-axis).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    x : array_like
        Abscissa values (basalt fraction).
    df_slice : pandas.DataFrame
        Rows for the slice, already sorted by ``x``.
    minerals : list of str
        Mineral columns to stack, in order.
    label : str
        Panel title.
    """
    ys = [df_slice[m].to_numpy() for m in minerals]
    colors = [MINERAL_COLORS[m] for m in minerals]
    ax.stackplot(
        x,
        *ys,
        colors=colors,
        edgecolor=fwl_mpl.SURFACES["light"]["background"],
        linewidth=0.3,
    )
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title(label, pad=3)


def stack_horizontal(ax, y, df_slice, minerals, label):
    """Draw a horizontally stacked area plot (volume fraction on the x-axis).

    Used for the temperature-dependent side panels, where temperature runs on
    the shared vertical axis and the mineral fractions stack left to right.
    """
    cum = np.zeros(len(y))
    for m in minerals:
        v = df_slice[m].to_numpy()
        ax.fill_betweenx(
            y,
            cum,
            cum + v,
            facecolor=MINERAL_COLORS[m],
            edgecolor=fwl_mpl.SURFACES["light"]["background"],
            linewidth=0.3,
        )
        cum = cum + v
    ax.set_ylim(y.min(), y.max())
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_title(label, pad=3)


def sublabel(ax, text):
    """Place the panel label ``text``, without brackets, above ``ax``."""
    fwl_mpl.panel_label(ax, text.strip("()"))


def build_figure(df: pd.DataFrame) -> mpl.figure.Figure:
    """Assemble the figure: rock-type map (a) beside four mineralogy panels (b-e).

    Parameters
    ----------
    df : pandas.DataFrame
        Mineralogy grid from :func:`load_grid`.

    Returns
    -------
    matplotlib.figure.Figure
        The figure, 7.1 in wide for the full text width.
    """
    minerals = _present_minerals(df)

    f_vals = np.sort(df["f_bas"].unique())
    T_vals = np.sort(df["T"].unique())
    T_lo, T_hi = T_vals.min(), T_vals.max()

    # Rock-type map: pivot rock type into a 2D (T, f_bas) integer grid.
    rock_to_int = {r: i for i, r in enumerate(ROCK_TYPES)}
    pivot = df.pivot(index="T", columns="f_bas", values="Rock type")
    Z = np.vectorize(lambda r: rock_to_int.get(r, np.nan))(pivot.to_numpy())
    cmap = ListedColormap(ROCK_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, len(ROCK_TYPES) + 0.5), cmap.N)

    fig = plt.figure(figsize=(7.1, 3.7), layout="none")
    gs = GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=[1.45, 1, 1],
        wspace=0.30,
        hspace=0.95,
        left=0.075,
        right=0.985,
        top=0.92,
        bottom=0.23,
    )
    ax_map = fig.add_subplot(gs[:, 0])
    ax_f0 = fig.add_subplot(gs[0, 1])
    ax_f1 = fig.add_subplot(gs[0, 2], sharey=ax_f0)
    ax_hi = fig.add_subplot(gs[1, 1])
    ax_lo = fig.add_subplot(gs[1, 2], sharey=ax_hi)

    # --- rock-type map ---
    ax_map.pcolormesh(
        f_vals, T_vals, Z, cmap=cmap, norm=norm, shading="nearest", rasterized=True
    )
    # Crisp black boundaries between rock-type fields.
    Tg, Fg = np.meshgrid(T_vals, f_vals, indexing="ij")
    ax_map.contour(
        Fg,
        Tg,
        Z,
        levels=np.arange(0.5, len(ROCK_TYPES) - 0.5 + 0.1),
        colors=fwl_mpl.SURFACES["light"]["ink"],
        linewidths=0.7,
    )
    ax_map.set_xlim(0, 1)
    ax_map.set_ylim(T_lo, T_hi)
    ax_map.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_map.set_yticks(np.arange(300, 1300, 100))
    ax_map.set_xlabel("Basalt fraction")
    ax_map.set_ylabel(r"Temperature, $T$ (K)")
    sublabel(ax_map, "(a)")

    # Rock-type labels at the field centroid, or at a manual anchor where the
    # centroid of an L-shaped or thin field falls near a boundary.
    anchors = {
        "Gabbro": (0.90, 430, 0),
        "Harzburgite": (0.042, 760, 90),
    }
    for r, i in rock_to_int.items():
        mask = Z == i
        if not mask.any():
            continue
        if r in anchors:
            fc, tc, rot = anchors[r]
        else:
            fc, tc, rot = Fg[mask].mean(), Tg[mask].mean(), 0
        dark = mpl.colors.rgb_to_hsv(ROCK_COLORS[i][:3])[2] < 0.6
        ax_map.text(
            fc,
            tc,
            r,
            ha="center",
            va="center",
            fontweight="bold",
            rotation=rot,
            color=fwl_mpl.SURFACES["light"]["background" if dark else "ink"],
        )

    # --- end members against temperature ---
    f0 = df[df["f_bas"] == f_vals.min()].sort_values("T")
    f1 = df[df["f_bas"] == f_vals.max()].sort_values("T")
    stack_horizontal(
        ax_f0, f0["T"].to_numpy(), f0, minerals, r"Harzburgite ($f_\mathrm{bas}=0$)"
    )
    stack_horizontal(
        ax_f1, f1["T"].to_numpy(), f1, minerals, r"Basalt ($f_\mathrm{bas}=1$)"
    )
    for ax in (ax_f0, ax_f1):
        ax.set_yticks([300, 600, 900, 1200])
        ax.set_xlabel("Vol. (%)")
    ax_f0.set_ylabel(r"$T$ (K)")
    ax_f1.tick_params(labelleft=False)
    sublabel(ax_f0, "(b)")
    sublabel(ax_f1, "(c)")

    # --- temperature slices against basalt fraction ---
    t_hi = df[df["T"] == T_hi].sort_values("f_bas")
    t_lo = df[df["T"] == T_lo].sort_values("f_bas")
    stack_vertical(
        ax_hi, t_hi["f_bas"].to_numpy(), t_hi, minerals, rf"$T = {int(T_hi)}$ K"
    )
    stack_vertical(
        ax_lo, t_lo["f_bas"].to_numpy(), t_lo, minerals, rf"$T = {int(T_lo)}$ K"
    )
    for ax in (ax_hi, ax_lo):
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xlabel("Basalt fraction")
    ax_hi.set_ylabel("Vol. (%)")
    ax_lo.tick_params(labelleft=False)
    sublabel(ax_hi, "(d)")
    sublabel(ax_lo, "(e)")

    # --- mineral legend, one row along the bottom ---
    handles = [
        mpl.patches.Patch(
            facecolor=MINERAL_COLORS[m],
            label=m,
            edgecolor=fwl_mpl.SURFACES["light"]["axis"],
            linewidth=0.4,
        )
        for m in minerals
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(minerals),
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
        handlelength=1.0,
        handleheight=1.0,
        columnspacing=1.0,
    )

    return fig


def main() -> None:
    """Parse arguments, build the figure, and save it as a vector PDF."""
    here = Path(__file__).resolve().parent
    default_csv = here.parent / "data" / "StagYY_mineralogy.csv"
    default_out = here.parent / "figures" / "fig01.pdf"

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=default_csv)
    p.add_argument("--out", type=Path, default=default_out)
    args = p.parse_args()

    fwl_mpl.use()
    df = load_grid(args.csv)
    fig = build_figure(df)
    problems = fwl_mpl.finish(fig)
    for line in problems:
        print(f"layout: {line}")
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
