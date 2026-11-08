"""Tests for the helpers of the overview figures."""

import numpy as np


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
