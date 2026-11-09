"""Tests for the helpers of the overview figures."""

import numpy as np
import pytest


def test_model_names_cover_every_plotted_run():
    """Every run in the flux cache and the model groups has a model name."""
    from extract_flux_cache import RUNS
    from stagyy_common import GROUPS, MODEL_NAMES

    plotted = set(RUNS) | {run for runs in GROUPS.values() for run in runs}
    assert plotted <= set(MODEL_NAMES)
    assert len(set(MODEL_NAMES.values())) == len(MODEL_NAMES)


def test_dayside_molten_share_counts_dayside_cells_only():
    """Only |lon| < 90 counts, the threshold is strict, and times map to later rows."""
    from plot_overview_figures import MOLTEN_MELT, dayside_molten_share

    lon = np.array([-170.0, -60.0, -10.0, 10.0, 60.0, 170.0])
    melt = np.array(
        [
            [1.0, 0.0, MOLTEN_MELT, 0.9, 0.9, 1.0],
            [0.0, 0.9, 0.9, 0.9, 0.9, 0.0],
        ]
    )
    surface_map = {"lon_deg": lon, "meltfrac": melt, "time_gyr": np.array([0.0, 1.0])}
    share = dayside_molten_share(surface_map, np.array([0.0, 0.5, 1.0, 9.0]))
    np.testing.assert_allclose(share, [0.5, 1.0, 1.0, 1.0])


def test_model_style_encodes_start_and_setup():
    """Hot starts are solid and cold starts dashed; '+' models share 1 colour."""
    from stagyy_common import model_style

    assert model_style(1)["ls"] == "-"
    assert model_style(2)["ls"] == "--"
    assert (
        model_style(9)["color"] == model_style(10)["color"] != model_style(5)["color"]
    )
    assert model_style(15)["color"] == model_style(16)["color"]


def test_fill_gaps_blends_linearly_across_a_missing_interval():
    """A missing interval becomes rows that move linearly from before to after."""
    from plot_overview_figures import fill_gaps

    values = np.array([[0.0, 10.0], [1.0, 20.0]])
    y = np.array([0.0, 1.0, 3.0, 4.0])
    rows = np.array([0, -1, 1])
    edges, out = fill_gaps(y, rows, values, n_sub=4)
    assert edges[0] == 0.0 and edges[-1] == 4.0 and len(edges) == len(out) + 1
    np.testing.assert_allclose(out[1:5, 0], [0.125, 0.375, 0.625, 0.875])
    np.testing.assert_allclose(out[[0, -1]], values)


def test_map_ticks_keep_narrow_labels_apart():
    """Ticks of a map with time across end on the run end and stay sparse."""
    from plot_overview_figures import map_ticks

    assert map_ticks(8.00161) == (8.0, [0.0, 4.0, 8.0])
    top, ticks = map_ticks(1.1302)
    assert top == 1.13 and ticks == [0.0, 0.5, 1.13]
    top, ticks = map_ticks(2.2611)
    assert top == 2.26 and ticks == [0.0, 1.0, 2.26]


@pytest.mark.parametrize(
    "rows", [np.array([-1, 0, 1]), np.array([0, 1, -1]), np.array([0, -1, -1, 1])]
)
def test_fill_gaps_needs_a_stored_row_on_both_sides(rows):
    from plot_overview_figures import fill_gaps

    values = np.array([[0.0], [1.0]])
    y = np.arange(rows.size + 1, dtype=float)
    with pytest.raises(ValueError, match="stored row on both sides"):
        fill_gaps(y, rows, values)


def test_despike_replaces_one_spike_and_keeps_the_ends():
    """A 10x output is replaced; the first and last outputs stay."""
    from plot_overview_figures import despike

    values = np.array([300.0, 30.0, 32.0, 300.0, 31.0, 33.0, 300.0])
    out = despike(values)
    np.testing.assert_allclose(out, [300.0, 30.0, 32.0, 32.0, 31.0, 33.0, 300.0])


def test_despike_refuses_values_at_or_below_zero():
    from plot_overview_figures import despike

    with pytest.raises(ValueError, match="positive values"):
        despike(np.array([30.0, -50.0, 31.0]))


def test_load_grid_drops_empty_minerals_without_changing_the_module_list(tmp_path):
    import plot_fig01_mineralogy as fig01

    before = list(fig01.MINERALS)
    path = tmp_path / "grid.csv"
    path.write_text(
        "f_bas,T,Olivine,Kyanite,Rock type\n0.0,300,60.0,0.0,Harzburgite\n"
        "0.1,300,50.0,0.0,Harzburgite\n"
    )
    df = fig01.load_grid(path)
    assert "Kyanite" not in df.columns and "Olivine" in df.columns
    assert fig01.MINERALS == before


def test_shoreline_angle_and_domain():
    """1274 K at T_day 1597 K lies at 67.8 deg; a melting T above T_day raises."""
    import manuscript_numbers as mn

    assert mn.shoreline_deg(1597.0) == pytest.approx(67.8, abs=0.05)
    with pytest.raises(ValueError, match="between T_NIGHT and t_day"):
        mn.shoreline_deg(1597.0, t_melt=1700.0)
