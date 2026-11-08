"""Plot Figure 2: mantle cross-sections of all paper runs at their last snapshot."""

import argparse
from pathlib import Path

import fwl_mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import LogFormatterSciNotation
from stagyy_common import DATA_DIR, MODEL_NAMES, REPLOT_DIR

# Plot thickness of the mantle relative to its true thickness.
STRETCH = 2.6
STRETCH_THICK = 1.3
PANELS = (
    ("T", "Temperature (K)", "cmc.lajolla", Normalize(0.0, 2400.0)),
    ("eta", r"Viscosity (Pa$\,$s)", "cmc.oslo_r", LogNorm(1.0e17, 1.0e26)),
    ("bulk_basalt", "Bulk basalt fraction", "cmc.bam_r", Normalize(0.0, 1.0)),
)
TICKS = (
    [0, 600, 1200, 1800, 2400],
    np.logspace(17.0, 26.0, 4),
    [0, 0.25, 0.5, 0.75, 1],
)
# Run number: T0 (K), T_day (K), eruption mode, melt segregation, core cooling.
SETUP = {
    1: (3000, 1597, None, False, False),
    2: (1600, 1597, None, False, False),
    4: (1600, 1597, None, False, False),
    5: (3000, 1597, "all", False, False),
    6: (3000, 1597, "lith", True, False),
    7: (1600, 1597, "all", False, False),
    8: (1600, 1597, "lith", True, False),
    9: (3000, 1745, "all", False, False),
    10: (1600, 1745, "all", False, False),
    12: (1600, 1745, None, False, False),
    13: (3000, 1745, "lith", True, False),
    14: (1600, 1745, "lith", True, False),
    15: (1600, 1597, None, False, True),
    16: (3000, 1597, None, False, True),
}
# Grid slot (row, column) of each run; column 0 hot start, column 1 cold start.
SLOTS = {
    1: (0, 0),
    2: (0, 1),
    5: (1, 0),
    7: (1, 1),
    6: (2, 0),
    8: (2, 1),
    16: (3, 0),
    15: (3, 1),
    9: (4, 0),
    10: (4, 1),
    13: (5, 0),
    14: (5, 1),
    4: (6, 0),
    12: (6, 1),
}


def block(run, t_gyr, thick):
    """Return the parameter text of one run.

    Parameters
    ----------
    run : int
        Run number.
    t_gyr : float
        Time of the snapshot in Gyr.
    thick : float
        Mantle thickness in km.

    Returns
    -------
    str
        Lines with T0, T_day, melt transport, and the variant, if any.
    """
    t0, tday, mode, seg, core = SETUP[run]
    lines = [rf"$T_0$ {t0} K", rf"$T_\mathrm{{day}}$ {tday} K"]
    if seg:
        lines.append(f"{mode} + segregation")
    else:
        lines.append(f"eruption: {mode}" if mode else "no eruption")
    if core:
        lines.append("core cooling")
    if abs(thick - 640.0) > 1.0:
        lines.append(f"mantle {thick:.0f} km")
    if t_gyr < 7.99:
        lines.append(rf"$t$ = {t_gyr:.3g} Gyr")
    return "\n".join(lines)


def ring(ax, data, run, field, cmap, norm, stretch, r_max):
    """Draw one annulus with the star on the left and the mantle stretched.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    data : numpy.lib.npyio.NpzFile
        Content of ``stagyy_fig02_annulus.npz``.
    run : int
        Run number.
    field : numpy.ndarray
        Cell values, shape ``(azimuth, radius)``.
    cmap : str
        Colour map.
    norm : matplotlib.colors.Normalize
        Colour normalisation.
    stretch : float
        Plot thickness of the mantle over its true thickness.
    r_max : float
        Largest surface radius of all runs in km, for a common scale.

    Returns
    -------
    tuple
        The mesh and the surface radius in km.
    """
    r_w = data[f"run{run}_r_walls"] / 1.0e3
    r_top = r_w[-1]
    r_plot = r_top - (r_top - r_w) * stretch
    pm, rm = np.meshgrid(data[f"run{run}_p_walls"], r_plot, indexing="ij")
    mesh = ax.pcolormesh(
        rm * np.cos(pm), rm * np.sin(pm), field, cmap=cmap, norm=norm, rasterized=True
    )
    phi = np.linspace(0.0, 2.0 * np.pi, 721)
    for rr in (r_plot[0], r_top):
        ax.plot(rr * np.cos(phi), rr * np.sin(phi), color="0.35", lw=0.3)
    for ph in np.radians([0.0, 90.0, 180.0, 270.0]):
        cos, sin = np.cos(ph), np.sin(ph)
        ax.plot(
            [r_top * cos, 1.07 * r_top * cos],
            [r_top * sin, 1.07 * r_top * sin],
            color="0.35",
            lw=0.5,
        )
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_xlim(-1.1 * r_max, 1.1 * r_max)
    ax.set_ylim(-1.1 * r_max, 1.1 * r_max)
    return mesh, r_top


def main(mode, out):
    """Draw the figure from ``data/stagyy_fig02_annulus.npz``.

    Parameters
    ----------
    mode : str
        Style mode of ``fwl_mpl``, ``light`` or ``dark``.
    out : pathlib.Path
        Output file.
    """
    fwl_mpl.use(mode)
    data = np.load(DATA_DIR / "stagyy_fig02_annulus.npz")
    r_max = max(data[f"run{r}_r_walls"][-1] for r in SLOTS) / 1.0e3
    fig = plt.figure(figsize=(7.1, 9.0), layout="none")
    outer = GridSpec(
        8,
        2,
        height_ratios=[0.25] + [1] * 7,
        wspace=0.06,
        hspace=0.04,
        left=0.035,
        right=0.995,
        top=0.985,
        bottom=0.06,
    )
    for col, head in enumerate(("Hot start", "Cold start")):
        hax = fig.add_subplot(outer[0, col])
        hax.set_axis_off()
        hax.text(0.5, 0.75, head, ha="center", va="center", fontsize=9, weight="bold")
        for j, panel in enumerate(PANELS):
            hax.text(
                (j + 0.5) / 3,
                0.12,
                panel[1].split(" (")[0],
                ha="center",
                va="center",
                fontsize=7,
            )
    meshes = {}
    for run, (row, col) in SLOTS.items():
        sub = outer[row + 1, col].subgridspec(1, 3, wspace=0.0)
        r_w = data[f"run{run}_r_walls"] / 1.0e3
        thick = r_w[-1] - r_w[0]
        stretch = STRETCH if abs(thick - 640.0) < 1.0 else STRETCH_THICK
        fields = (
            data[f"run{run}_T"].astype(float),
            10.0 ** data[f"run{run}_log_eta"].astype(float),
            data[f"run{run}_bulk_basalt"].astype(float),
        )
        for j, ((_, _, cmap, norm), field) in enumerate(zip(PANELS, fields)):
            ax = fig.add_subplot(sub[0, j])
            mesh, r_top = ring(ax, data, run, field, cmap, norm, stretch, r_max)
            meshes[j] = mesh
            name = MODEL_NAMES[run]
            if j == 0:
                ax.text(
                    -1.1 * r_max,
                    1.1 * r_max,
                    name,
                    ha="left",
                    va="top",
                    fontsize=6.5,
                    weight="bold",
                )
                text = block(run, float(data[f"run{run}_time_gyr"]), thick)
                ax.text(
                    0.0,
                    0.0,
                    text,
                    ha="center",
                    va="center",
                    fontsize=4.6,
                    linespacing=1.1,
                )
            else:
                ax.text(
                    0.0, 0.0, name, ha="center", va="center", fontsize=6, weight="bold"
                )
            if row == 0 and j == 0:
                ax.annotate(
                    "",
                    xy=(-1.0 * r_top, 0.0),
                    xytext=(-1.35 * r_top, 0.0),
                    arrowprops={"arrowstyle": "-|>", "color": "C2", "lw": 1.0},
                    annotation_clip=False,
                )
                ax.text(
                    -1.33 * r_top,
                    0.12 * r_top,
                    "Star",
                    ha="left",
                    va="bottom",
                    fontsize=5.5,
                )
    for j, (_, label, _, norm) in enumerate(PANELS):
        cax = fig.add_axes([0.05 + j * 0.33, 0.027, 0.26, 0.010])
        cb = fig.colorbar(meshes[j], cax=cax, orientation="horizontal", ticks=TICKS[j])
        cb.set_label(label, fontsize=7, labelpad=1)
        cb.ax.tick_params(labelsize=6)
        if isinstance(norm, LogNorm):
            cax.xaxis.set_major_formatter(LogFormatterSciNotation())
        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position("bottom")
    for line in fwl_mpl.finish(fig, dpi=400):
        print("layout:", line)
    fig.savefig(out, dpi=400)
    print("wrote", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="light", choices=("light", "dark"))
    parser.add_argument("--out", type=Path, default=REPLOT_DIR / "fig02.pdf")
    args = parser.parse_args()
    main(args.mode, args.out)
