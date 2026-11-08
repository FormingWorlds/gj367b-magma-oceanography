"""Plot the observation figures: surfaces, atmosphere evidence and MIRI photometry.

Reads ``data/vayu_surface_spectra.npz``, ``data/miri_photometry.csv``,
``data/bayes_model_comparison.csv`` and ``data/bayes_redistribution.csv``
(written by ``extract_vayu_obs.py`` and VAYU) and writes
``replots/fig_obs_surfaces.pdf``, ``fig_obs_atmospheres.pdf`` and
``fig_obs_photometry.pdf``.
"""

import argparse
import csv
import json
from pathlib import Path

import fwl_mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from plot_fig12_redistribution import FLOOR, MODES, PRESSURE_LABELS, PRESSURES
from plot_fig12_redistribution import cell_text, grid, read_table

FIG_DIR = Path(__file__).resolve().parents[1]
DATA = FIG_DIR / "data"
REPLOT_DIR = FIG_DIR / "figures"
THRESHOLD = -3.0
# Ratio of observed to predicted 1-sigma (ppm) of the TRAPPIST-1c F1500W eclipse.
NOISE_FACTOR = json.loads((DATA / "miri_noise_calibration.json").read_text())[
    "noise_factor"
]
# Rock types that the StagYY and Perple_X models predict at the surface.
PREDICTED = ("gabbro", "norite", "lherzolite", "harzburgite")
GAS4 = (("CO2", r"CO$_2$"), ("SO2", r"SO$_2$"), ("H2O", r"H$_2$O"), ("CH4", r"CH$_4$"))
MIXES = (
    ("", "pure"),
    ("N2_1ppm_", "1 ppm"),
    ("N2_1000ppm_", r"$10^3$ ppm"),
)
# Pressure label, value in bar and the name suffix of the Koll (2022) redistribution.
KOLL = (
    ("001", 0.01, ""),
    ("01", 0.1, ""),
    ("1", 1.0, "_less"),
    ("10", 10.0, "_more"),
    ("100", 100.0, "_full"),
)


def surface_evidence(path=DATA / "bayes_model_comparison.csv"):
    """Return the delta ln Z of every bare-rock surface.

    Parameters
    ----------
    path : Path
        VAYU evidence table.

    Returns
    -------
    dict
        Delta ln Z relative to pyrite by surface name.
    """
    with open(path, newline="") as f:
        return {
            r["surface"]: float(r["ΔlnZ"])
            for r in csv.DictReader(f)
            if r["atmosphere"] == "bare_rock"
        }


def read_photometry(path=DATA / "miri_photometry.csv"):
    """Return the relative emission and the calibrated 1-sigma by model name.

    The raw uncertainty is 1/SNR, a fraction of the eclipse depth, so the
    absolute 1-sigma of the relative emission is raw x emission x NOISE_FACTOR.

    Parameters
    ----------
    path : Path
        Output of ``extract_vayu_obs.py``.

    Returns
    -------
    dict
        ``name -> {"kind", "F1280W", "F1500W", "F1280W_sigma", "F1500W_sigma"}``.
    """
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            row = {"kind": r["kind"]}
            for band in ("F1280W", "F1500W"):
                row[band] = float(r[band])
                row[f"{band}_sigma"] = (
                    float(r[f"{band}_sigma_raw"]) * row[band] * NOISE_FACTOR
                )
            out[r["name"]] = row
    return out


def consistent_range(phot, evidence):
    """Return the lowest and highest relative emission of the consistent surfaces.

    Parameters
    ----------
    phot : dict
        Output of :func:`read_photometry`.
    evidence : dict
        Output of :func:`surface_evidence`.

    Returns
    -------
    tuple of float
        Minimum and maximum over both filters.
    """
    vals = [
        phot[name][band]
        for name, row in phot.items()
        if row["kind"] == "surface" and evidence[name] >= THRESHOLD
        for band in ("F1280W", "F1500W")
    ]
    return min(vals), max(vals)


def surfaces_figure(out):
    """Draw the surface spectra (a) and the ranked surface evidence (b)."""
    spec = np.load(DATA / "vayu_surface_spectra.npz")
    evidence = surface_evidence()
    pal = fwl_mpl.PALETTE["light"]
    names = list(spec["names"])
    labels = dict(zip(names, spec["labels"], strict=True))
    wl = spec["wavelength_um"]
    contrast = dict(zip(names, spec["contrast_ppm"], strict=True))
    ok = [n for n in names if evidence[n] >= THRESHOLD]
    rejected = [n for n in names if evidence[n] < THRESHOLD]
    fig = plt.figure(figsize=(7.1, 3.5), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=(1.6, 1.0))
    ax = fig.add_subplot(gs[0])
    stack = np.array([contrast[n] for n in ok])
    ax.fill_between(
        wl,
        stack.min(axis=0),
        stack.max(axis=0),
        color=pal["signal"],
        alpha=0.18,
        lw=0,
        label=f"{len(ok)} consistent surfaces",
    )
    lines = ("gabbro", "lherzolite", "harzburgite")
    for name, color in zip(
        lines, (pal["magma"], pal["fern"], pal["iris"]), strict=True
    ):
        ax.plot(
            wl,
            contrast[name],
            color=color,
            lw=0.9,
            label=f"{labels[name]} ({evidence[name]:.1f})",
        )
    ax.plot(wl, contrast["pyrite"], color=pal["signal"], lw=1.5, label="Pyrite (0)")
    for name in rejected:
        ax.plot(wl, contrast[name], color=pal["garnet"], lw=0.6, ls=":")
    ax.plot(
        [], [], color=pal["garnet"], lw=0.6, ls=":", label=f"{len(rejected)} rejected"
    )
    ax.plot(
        wl,
        spec["blackbody_ppm"],
        color="0.4",
        ls="--",
        lw=0.8,
        label=f"Blackbody {float(spec['blackbody_k']):.0f} K",
    )
    ax.errorbar(
        spec["obs_um"],
        spec["obs_ppm"],
        xerr=spec["obs_um_err"],
        yerr=spec["obs_ppm_err"],
        fmt="o",
        ms=3,
        color="black",
        elinewidth=0.8,
        capsize=0,
        zorder=10,
        label="Zhang et al. (2024)",
    )
    ax.set_xlim(5.0, 12.0)
    ax.set_ylim(0.0, 200.0)
    ax.set_yticks(range(0, 201, 50))
    ax.set_xlabel(r"Wavelength ($\mu$m)")
    ax.set_ylabel("Eclipse depth (ppm)")
    ax.legend(
        fontsize=5.5,
        ncols=3,
        loc="upper left",
        frameon=False,
        columnspacing=0.8,
        handlelength=1.6,
    )
    bx = fig.add_subplot(gs[1])
    order = sorted(names, key=lambda n: evidence[n])
    y = np.arange(len(order))
    values = [evidence[n] for n in order]
    bx.barh(
        y,
        values,
        height=0.7,
        color=[pal["signal"] if v >= THRESHOLD else pal["garnet"] for v in values],
    )
    bx.plot([0.0], [y[-1]], "o", ms=3, color=pal["signal"], clip_on=False)
    bx.axvline(THRESHOLD, color="0.3", ls="--", lw=0.7)
    bx.set_yticks(y, [labels[n] for n in order], fontsize=5.5)
    for tick, name in zip(bx.get_yticklabels(), order, strict=True):
        if name in PREDICTED:
            tick.set_fontweight("bold")
    bx.tick_params(axis="y", length=0)
    bx.minorticks_off()
    bx.set_xlim(-10.0, 0.0)
    bx.set_xticks(range(-10, 1, 2))
    bx.set_xlabel(r"$\Delta \ln Z$ vs pyrite")
    fwl_mpl.panel_label(ax, "a")
    fwl_mpl.panel_label(bx, "b")
    save(fig, out)


def gas_colours():
    """Return the colour of each gas, shared by the atmosphere and photometry figures."""
    pal = fwl_mpl.PALETTE["light"]
    return {
        "CO2": pal["garnet"],
        "SO2": pal["signal"],
        "H2O": pal["ochre"],
        "CH4": pal["fern"],
    }


def shade_cmap(colour):
    """Return a colour map from white (delta ln Z at the floor) to ``colour`` (0)."""
    return LinearSegmentedColormap.from_list("shade", [(1.0, 1.0, 1.0), to_rgb(colour)])


def heat(ax, values, xlabels, cmap):
    """Draw one evidence matrix with the values in the cells.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    values : ndarray
        Delta ln Z, shape ``(pressures, columns)``; row 0 is the lowest pressure.
    xlabels : list of str
    cmap : matplotlib.colors.Colormap
        Colour map over delta ln Z from ``FLOOR`` to 0.
    """
    norm = Normalize(FLOOR, 0.0)
    ax.imshow(
        np.clip(values, FLOOR, 0.0), cmap=cmap, norm=norm, origin="lower", aspect="auto"
    )
    for i, j in np.ndindex(values.shape):
        v = values[i, j]
        r, g, b = cmap(norm(max(v, FLOOR)))[:3]
        dark = 0.299 * r + 0.587 * g + 0.114 * b < 0.5
        ax.text(
            j,
            i,
            cell_text(v),
            ha="center",
            va="center",
            fontsize=5,
            color="white" if dark else "black",
            fontweight="bold" if v >= THRESHOLD else "normal",
        )
        if v >= THRESHOLD:
            ax.add_patch(
                plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="black", lw=1.0)
            )
    ax.set_xticks(range(len(xlabels)), xlabels, fontsize=5.5, rotation=45)
    ax.minorticks_off()


def gas_matrix(table, gas):
    """Return the f = 2/3 evidence of one gas, pure and in N2, shape ``(pressures, 3)``."""
    return np.array(
        [[table[f"{p}bar_{pre}{gas}"] for pre, _ in MIXES] for p in PRESSURES]
    )


def atmospheres_figure(out):
    """Draw the f = 2/3 evidence per gas (a) and the redistribution panels (b, c)."""
    colours = gas_colours()
    table = read_table(DATA / "bayes_model_comparison.csv")
    red = read_table(DATA / "bayes_redistribution.csv")
    fig = plt.figure(figsize=(7.1, 2.8), layout="constrained")
    gs = fig.add_gridspec(1, 6, width_ratios=(3, 3, 3, 3, 5, 5))
    axes = []
    for k, (gas, label) in enumerate(GAS4):
        ax = fig.add_subplot(gs[k], sharey=axes[0] if axes else None)
        heat(
            ax, gas_matrix(table, gas), [m for _, m in MIXES], shade_cmap(colours[gas])
        )
        ax.set_title(
            ("(a) " if k == 0 else "") + label,
            loc="left",
            fontsize=8,
            color=colours[gas],
            fontweight="bold",
        )
        ax.set_xlabel("$f = 2/3$")
        if k:
            ax.tick_params(labelleft=False)
        axes.append(ax)
    axes[0].set_yticks(range(len(PRESSURES)), PRESSURE_LABELS)
    axes[0].set_ylabel("Surface pressure (bar)")
    for k, (gas, colour, title) in enumerate(
        (
            ("H2O", colours["H2O"], r"(b) H$_2$O"),
            ("N2_1000ppm_CO2", colours["CO2"], r"(c) 1000 ppm CO$_2$ in N$_2$"),
        )
    ):
        bx = fig.add_subplot(gs[4 + k], sharey=axes[0])
        heat(bx, grid(red, gas), [label for _, label in MODES], shade_cmap(colour))
        bx.tick_params(labelleft=False)
        bx.set_xlabel("$f$")
        bx.set_title(title, loc="left", fontsize=8, color=colour, fontweight="bold")
        axes.append(bx)
    bar = fig.colorbar(
        plt.cm.ScalarMappable(Normalize(FLOOR, 0.0), shade_cmap("0.25")),
        ax=axes,
        shrink=0.9,
        pad=0.01,
    )
    bar.set_label(r"$\Delta \ln Z$ vs pyrite (shade)")
    save(fig, out)


def photometry_figure(out):
    """Draw the relative F1280W and F1500W emission against surface pressure."""
    phot = read_photometry()
    lo, hi = consistent_range(phot, surface_evidence())
    gases = tuple(gas_colours().items())
    fig, axes = plt.subplots(
        2, 2, figsize=(7.1, 4.2), sharex=True, sharey=True, layout="constrained"
    )
    for r, (pre, mix) in enumerate(
        (("N2_1000ppm_", r"1000 ppm in N$_2$"), ("", "pure"))
    ):
        for c, koll in enumerate((False, True)):
            ax = axes[r, c]
            ax.axhspan(lo, hi, color="0.85", lw=0, zorder=0)
            ax.axhline(1.0, color="0.4", ls="--", lw=0.6)
            for k, (gas, color) in enumerate(gases):
                rows = [
                    phot[f"{p}bar_{pre}{gas}{suf if koll else ''}"]
                    for p, _, suf in KOLL
                ]
                x = np.array([v for _, v, _ in KOLL])
                for band, ls, marker, shift in (
                    ("F1280W", "-", "o", 0.94),
                    ("F1500W", "--", "s", 1.06),
                ):
                    ax.errorbar(
                        x * shift * (1.0 + 0.03 * (k - 1.5)),
                        [w[band] for w in rows],
                        yerr=[w[f"{band}_sigma"] for w in rows],
                        color=color,
                        ls=ls,
                        marker=marker,
                        ms=2.5,
                        lw=0.8,
                        elinewidth=0.6,
                        capsize=0,
                        mfc="white" if band == "F1500W" else color,
                    )
            ax.set_xscale("log")
            kind = "Koll scaling" if koll else "$f = 2/3$"
            ax.set_title(f"{mix}, {kind}", loc="left", fontsize=8)
            if r == 1:
                ax.set_xlabel("Surface pressure (bar)")
            if c == 0:
                ax.set_ylabel("Emission / blackbody")
            fwl_mpl.panel_label(ax, "abcd"[2 * r + c])
    axes[1, 1].set_xticks(
        [v for _, v, _ in KOLL],
        ["0.01\n2/3", "0.1\n2/3", "1\n9/16", "10\n17/48", "100\n1/4"],
    )
    axes[0, 0].set_ylim(0.2, 1.4)
    axes[0, 0].set_yticks(np.arange(0.2, 1.41, 0.2))
    handles = [
        plt.Line2D([], [], color=c, lw=1.2, label=lab)
        for (_, c), lab in zip(
            gases, (r"CO$_2$", r"SO$_2$", r"H$_2$O", r"CH$_4$"), strict=True
        )
    ]
    handles += [
        plt.Line2D([], [], color="0.3", marker="o", ms=3, ls="-", label="F1280W"),
        plt.Line2D(
            [], [], color="0.3", marker="s", mfc="white", ms=3, ls="--", label="F1500W"
        ),
        plt.Rectangle((0, 0), 1, 1, color="0.85", label="consistent surfaces"),
    ]
    fig.legend(
        handles=handles, loc="outside lower center", ncols=7, fontsize=7, frameon=False
    )
    save(fig, out)


def save(fig, out):
    """Apply the layout rules, print what remains wrong, and save ``fig`` to ``out``."""
    for line in fwl_mpl.finish(fig):
        print(f"layout {out.name}:", line)
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


FIGURES = {
    "surfaces": surfaces_figure,
    "atmospheres": atmospheres_figure,
    "photometry": photometry_figure,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "names", nargs="*", choices=sorted(FIGURES), default=sorted(FIGURES)
    )
    parser.add_argument("--out-dir", type=Path, default=REPLOT_DIR)
    args = parser.parse_args()
    fwl_mpl.use()
    for name in args.names or sorted(FIGURES):
        FIGURES[name](args.out_dir / f"fig_obs_{name}.pdf")
