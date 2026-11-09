"""Tests for the table helpers of the Figure 12 plot."""

import numpy as np
import pytest
from plot_fig12_redistribution import (
    DATA,
    MODES,
    PRESSURES,
    cell_text,
    grid,
    read_table,
)


def _table(gas, value=-1.0):
    """Return a complete grid for one gas with every value equal to ``value``."""
    return {f"{p}bar_{gas}{s}": value for p in PRESSURES for s, _ in MODES}


def test_grid_rows_are_pressures_and_columns_are_f():
    """Row 0 is 0.01 bar, column 0 is f = 2/3, column 4 is f = 1/4."""
    table = _table("H2O")
    table["001bar_H2O"] = -0.5
    table["100bar_H2O_full"] = -20.0
    out = grid(table, "H2O")
    table[f"{PRESSURES[0]}bar_H2O{MODES[4][0]}"] = -3.0
    table[f"{PRESSURES[4]}bar_H2O{MODES[0][0]}"] = -7.0
    out = grid(table, "H2O")
    assert out.shape == (5, 5)
    assert out[0, 0] == -0.5
    assert out[4, 4] == -20.0
    assert out[0, 4] == -3.0
    assert out[4, 0] == -7.0


def test_grid_missing_model_raises():
    """A missing grid point is an error, not a silent gap."""
    table = _table("CO2")
    del table["10bar_CO2_half"]
    with pytest.raises(KeyError):
        grid(table, "CO2")


@pytest.mark.parametrize(
    ("value", "text"),
    [(-2.34, "−2.3"), (-10.0, "−10.0"), (-10.01, "<−10"), (0.0, "0.0")],
)
def test_cell_text(value, text):
    """Values at or above the floor get 1 decimal; values below it get ``<-10``."""
    assert cell_text(value) == text


def test_stored_table_is_complete():
    """The stored table holds all 125 models of the 5 panels."""
    table = read_table(DATA)
    for gas in ("CH4", "CO2", "H2O", "SO2", "N2_1000ppm_CO2"):
        assert np.isfinite(grid(table, gas)).all()


def test_cell_text_signs_and_floor():
    """No negative zero, a real minus sign, and the floor label."""
    assert cell_text(-0.04) == "0.0"
    assert cell_text(-1.26) == "−1.3"
    assert cell_text(2.0) == "2.0"
    assert cell_text(-50.0) == "<−10"
