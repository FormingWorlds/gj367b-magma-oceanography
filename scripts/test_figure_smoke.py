"""Smoke tests: each figure script writes its PDF into a folder that does not exist yet."""

import sys

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")


def test_fig01_main_writes_into_a_new_folder(tmp_path, monkeypatch):
    import plot_fig01_mineralogy as fig01

    out = tmp_path / "new" / "fig01.pdf"
    monkeypatch.setattr(sys, "argv", ["plot_fig01_mineralogy.py", "--out", str(out)])
    fig01.main()
    assert out.stat().st_size > 0


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([(0.0, 300.0, "Harzburgite"), (0.1, 300.0, "Dunite")], "unknown rock types"),
        ([(0.0, 300.0, "Harzburgite"), (0.1, 400.0, "Norite")], "misses"),
    ],
)
def test_fig01_refuses_a_broken_grid(rows, message):
    import plot_fig01_mineralogy as fig01

    df = pd.DataFrame(rows, columns=["f_bas", "T", "Rock type"])
    df["Olivine"] = 50.0
    with pytest.raises(ValueError, match=message):
        fig01.build_figure(df)


def test_fig02_main_writes_into_a_new_folder(tmp_path):
    import plot_fig02_annulus as fig02

    out = tmp_path / "new" / "fig02.pdf"
    fig02.main("light", out)
    assert out.stat().st_size > 0


def test_fig12_main_writes_into_a_new_folder(tmp_path):
    import plot_fig12_redistribution as fig12

    out = tmp_path / "new" / "fig12.pdf"
    fig12.main(out)
    assert out.stat().st_size > 0


def test_overview_save_writes_into_a_new_folder(tmp_path):
    import matplotlib.pyplot as plt
    import plot_overview_figures as pof

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    out = tmp_path / "new" / "fig.pdf"
    pof.save(fig, out)
    assert out.stat().st_size > 0
