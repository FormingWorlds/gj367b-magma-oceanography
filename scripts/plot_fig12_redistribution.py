"""Plot Figure 12: Bayesian evidence against heat redistribution and surface pressure.

Reads ``figures/data/bayes_redistribution.csv``, the VAYU nested-sampling output for
the greybody surface with 5 atmospheres, 5 surface pressures and 5 redistribution
factors, and writes ``figures/replots/fig12.pdf``.
"""

import argparse
import csv
from pathlib import Path

import fwl_mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

FIG_DIR = Path(__file__).resolve().parents[1]
DATA = FIG_DIR / "data" / "bayes_redistribution.csv"
OUT = FIG_DIR / "figures" / "fig12.pdf"

PRESSURES = ("001", "01", "1", "10", "100")
PRESSURE_LABELS = ("0.01", "0.1", "1", "10", "100")
# Name suffix and tick label for each f, from no redistribution to full redistribution.
MODES = (
    ("", "2/3"),
    ("_less", "9/16"),
    ("_half", "11/24"),
    ("_more", "17/48"),
    ("_full", "1/4"),
)
GASES = (
    ("CH4", r"CH$_4$"),
    ("CO2", r"CO$_2$"),
    ("H2O", r"H$_2$O"),
    ("SO2", r"SO$_2$"),
    ("N2_1000ppm_CO2", r"N$_2$ + 1000 ppm CO$_2$"),
)
THRESHOLD = -3.0
FLOOR = -10.0


def read_table(path):
    """Read ``atmosphere -> delta ln Z`` for the greybody surface.

    Parameters
    ----------
    path : Path
        CSV with the columns ``surface``, ``atmosphere`` and ``ΔlnZ``.

    Returns
    -------
    dict
        Delta ln Z by atmosphere name.
    """
    with open(path, newline="") as f:
        return {
            r["atmosphere"]: float(r["ΔlnZ"])
            for r in csv.DictReader(f)
            if r["surface"] == "greybody"
        }


def grid(table, gas):
    """Arrange the delta ln Z of one gas as pressure by redistribution factor.

    Parameters
    ----------
    table : dict
        Output of :func:`read_table`.
    gas : str
        Atmosphere base name, for example ``"H2O"``.

    Returns
    -------
    ndarray
        Shape ``(len(PRESSURES), len(MODES))``; row 0 is the lowest pressure.

    Raises
    ------
    KeyError
        If a model of the grid is missing from the table.
    """
    return np.array(
        [[table[f"{p}bar_{gas}{suffix}"] for suffix, _ in MODES] for p in PRESSURES]
    )


def cell_text(value):
    """Return the cell label: 1 decimal, or ``<-10`` below the colour floor."""
    if value < FLOOR:
        return "<−10"
    text = f"{value:.1f}"
    return "0.0" if text == "-0.0" else text.replace("-", "−")


def main(out):
    """Draw the 5 panels and the shared colour bar into ``out``."""
    fwl_mpl.use()
    table = read_table(DATA)
    norm = Normalize(vmin=FLOOR, vmax=0.0)
    cmap = plt.get_cmap("Reds_r")
    fig, axes = plt.subplots(
        1, 5, figsize=(7.1, 2.3), sharey=True, layout="constrained"
    )
    for ax, (gas, title) in zip(axes, GASES, strict=True):
        values = grid(table, gas)
        ax.imshow(
            np.clip(values, FLOOR, 0.0),
            cmap=cmap,
            norm=norm,
            origin="lower",
            aspect="auto",
        )
        for i, j in np.ndindex(values.shape):
            v = values[i, j]
            ax.text(
                j,
                i,
                cell_text(v),
                ha="center",
                va="center",
                fontsize=6,
                color="white" if v < -6.0 else "black",
                fontweight="bold" if v >= THRESHOLD else "normal",
            )
            if v >= THRESHOLD:
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", lw=1.2
                    )
                )
        ax.set_xticks(
            range(len(MODES)), [label for _, label in MODES], fontsize=6, rotation=45
        )
        ax.minorticks_off()
        ax.set_title(
            f"({'abcde'[axes.tolist().index(ax)]}) {title}", fontsize=8, loc="left"
        )
        ax.set_xlabel("$f$")
    axes[0].set_yticks(range(len(PRESSURES)), PRESSURE_LABELS)
    axes[0].set_ylabel("Surface pressure (bar)")
    bar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, shrink=0.9, pad=0.01
    )
    bar.set_label(r"$\Delta \ln Z$ vs pyrite")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT)
    main(parser.parse_args().out)
