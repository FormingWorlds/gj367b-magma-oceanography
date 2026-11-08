"""Tests for the StagYY helper functions on synthetic input."""

from types import SimpleNamespace

import numpy as np
import pytest
import stagyy_common as sc


def fake_step(temp, cond, r_walls, n_phi=8):
    """Return an object with the attributes of a StagPy step that the flux reads."""
    r_walls = np.asarray(r_walls, dtype=float)
    p_walls = np.linspace(0.0, 2.0 * np.pi, n_phi + 1)
    fields = {
        "T": SimpleNamespace(
            values=np.broadcast_to(temp, (n_phi, r_walls.size - 1))[None, :, :, None]
        ),
        "Tcond": SimpleNamespace(
            values=np.broadcast_to(cond, (n_phi, r_walls.size - 1))[None, :, :, None]
        ),
    }
    geom = SimpleNamespace(
        r_walls=r_walls,
        r_centers=0.5 * (r_walls[1:] + r_walls[:-1]),
        p_centers=0.5 * (p_walls[1:] + p_walls[:-1]),
    )
    return SimpleNamespace(fields=fields, geom=geom)


def test_longitude_places_substellar_at_zero_and_keeps_sign():
    phi = np.radians([90.0, 180.0, 270.0, 359.0])
    assert np.allclose(sc.longitude_deg(phi), [-90.0, 0.0, 90.0, 179.0])


def test_near_surface_flux_is_series_resistance_on_unequal_cells():
    # Cell centres at 10 and 40 km with k = 2 and 8: a width-free harmonic mean gives another value.
    step = fake_step([500.0, 200.0], [2.0, 8.0], [0.0, 20.0e3, 60.0e3])
    flux = sc.conductive_flux(step, 1597.0, 100.0)["near_surface"]
    expected = 300.0 / (10.0e3 / 2.0 + 20.0e3 / 8.0)
    assert flux == pytest.approx(np.full(8, expected))
    assert expected != pytest.approx(2.0 * 2.0 * 8.0 / 10.0 * 300.0 / 30.0e3)


def test_half_cell_flux_solves_the_radiative_balance_at_night():
    step = fake_step([500.0, 200.0], [3.0, 3.0], [0.0, 20.0e3, 40.0e3])
    out = sc.conductive_flux(step, 1597.0, 100.0)
    night = np.abs(out["lon_deg"]) > 90.0
    balance = sc.SIGMA_SB * (out["t_surface"][night] ** 4 - 100.0**4)
    assert out["half_cell"][night] == pytest.approx(balance)
    assert out["half_cell"][night] == pytest.approx(
        3.0 * (200.0 - out["t_surface"][night]) / 10.0e3
    )
    assert out["half_cell"][~night][0] != pytest.approx(out["half_cell"][night][0])


@pytest.mark.parametrize(
    ("n_items", "max_rows"), [(799, 400), (800, 400), (5, 400), (95839, 2000)]
)
def test_sample_indices_respect_the_limit_and_keep_the_last(n_items, max_rows):
    picks = sc.sample_indices(n_items, max_rows)
    assert picks[0] == 0 and picks[-1] == n_items - 1
    assert len(picks) <= max_rows + 1


def test_time_cells_leave_the_gap_open():
    time = np.array([1.0, 2.0, 3.0, 4.0, 10.0, 11.0])
    isnap = np.array([0, 1, 2, 3, 9, 10])
    cells, gaps = sc.time_cells(time, isnap, 1)
    assert gaps.tolist() == [3]
    assert cells[3, 1] == pytest.approx(4.5)
    assert cells[4, 0] == pytest.approx(9.5)
    assert cells[0, 0] == 0.0
    assert cells[-1, 1] == pytest.approx(11.5)


def test_time_cells_refuse_decreasing_time():
    with pytest.raises(ValueError, match="increase"):
        sc.time_cells(np.array([1.0, 0.5]), np.array([0, 1]), 1)


def test_check_output_refuses_the_run_data(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "DATA_ROOT", tmp_path)
    with pytest.raises(ValueError, match="run data"):
        sc.check_output(tmp_path / "test_01" / "x.npz")
    assert sc.check_output(tmp_path.parent / "elsewhere.npz").name == "elsewhere.npz"


def test_time_cells_keep_edges_increasing_when_output_gets_denser_after_a_gap():
    time = np.array([0.0, 1.0, 2.0, 3.0, 3.2, 3.4])
    cells, gaps = sc.time_cells(time, np.array([0, 1, 2, 3, 9, 10]), 1)
    assert gaps.tolist() == [3]
    assert np.all(cells[:, 1] > cells[:, 0])
    assert cells[3, 1] <= cells[4, 0]


def test_time_cells_treat_one_extra_snapshot_as_no_gap_and_two_as_a_gap():
    time = np.array([1.0, 2.0, 3.0])
    assert sc.time_cells(time, np.array([0, 5, 11]), 5)[1].size == 0
    assert sc.time_cells(time, np.array([0, 5, 12]), 5)[1].tolist() == [1]


def test_time_cells_need_two_rows():
    with pytest.raises(ValueError, match="at least 2"):
        sc.time_cells(np.array([1.0]), np.array([0]), 1)


def test_surface_temperature_follows_the_irradiation_law_for_small_conductivity():
    step = fake_step([500.0, 200.0], [1.0e-9, 1.0e-9], [0.0, 20.0e3, 40.0e3])
    out = sc.conductive_flux(step, 1597.0, 100.0)
    expected = (
        1597.0**4 * np.clip(np.cos(np.radians(out["lon_deg"])), 0.0, None) + 100.0**4
    ) ** 0.25
    assert out["t_surface"] == pytest.approx(expected, rel=1.0e-6)
