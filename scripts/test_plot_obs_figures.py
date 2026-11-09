"""Tests for the observation figures and their data files."""

import csv
import json

import numpy as np
import plot_obs_figures as obs
import pytest
from plot_fig12_redistribution import grid, read_table


def test_surface_evidence_counts():
    """27 surfaces, 23 with delta ln Z >= -3, and pyrite is the reference."""
    ev = obs.surface_evidence()
    names = list(np.load(obs.DATA / "vayu_surface_spectra.npz")["names"])
    assert len(names) == 27
    assert sum(ev[n] >= obs.THRESHOLD for n in names) == 23
    assert ev["pyrite"] == 0.0
    assert all(name in ev for name in obs.PREDICTED)


def test_photometry_applies_noise_factor(tmp_path):
    """The calibrated sigma is raw sigma x relative emission x NOISE_FACTOR."""
    path = tmp_path / "phot.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["kind", "name", "F1280W", "F1500W", "F1280W_sigma_raw", "F1500W_sigma_raw"]
        )
        w.writerow(["surface", "x", "0.9", "0.8", "0.01", "0.02"])
    row = obs.read_photometry(path)["x"]
    assert row["F1280W_sigma"] == pytest.approx(0.01 * 0.9 * obs.NOISE_FACTOR)
    assert row["F1500W_sigma"] == pytest.approx(0.02 * 0.8 * obs.NOISE_FACTOR)
    assert row["F1500W"] == 0.8


def test_consistent_range_excludes_rejected_surfaces():
    """Magnesium sulphate (rejected) is far below the consistent-surface band."""
    phot = obs.read_photometry()
    ev = obs.surface_evidence()
    lo, hi = obs.consistent_range(phot, ev)
    assert 0.9 < lo < hi < 1.01
    assert lo == pytest.approx(0.919364, abs=1e-6)
    assert hi == pytest.approx(1.00542, abs=1e-6)
    assert phot["magnesium_sulphate"]["F1280W"] < lo - 0.1


def test_photometry_has_every_plotted_model():
    """Each gas, mixture and Koll pressure step of the photometry figure exists."""
    phot = obs.read_photometry()
    for pre in ("", "N2_1000ppm_"):
        for gas in ("CO2", "SO2", "H2O", "CH4"):
            for p, _, suffix in obs.KOLL:
                assert f"{p}bar_{pre}{gas}" in phot
                assert f"{p}bar_{pre}{gas}{suffix}" in phot


def test_gas_matrix_matches_redistribution_table():
    """The pure H2O column and the 1000 ppm CO2 column agree with f = 2/3 of Fig. b, c."""
    table = read_table(obs.DATA / "bayes_model_comparison.csv")
    red = read_table(obs.DATA / "bayes_redistribution.csv")
    h2o = obs.gas_matrix(table, "H2O")
    assert h2o.shape == (5, 3)
    np.testing.assert_allclose(h2o[:, 0], grid(red, "H2O")[:, 0])
    np.testing.assert_allclose(
        obs.gas_matrix(table, "CO2")[:, 2], grid(red, "N2_1000ppm_CO2")[:, 0]
    )


def test_shade_cmap_ends():
    """The shade map is white at the floor and the gas colour at 0."""
    cmap = obs.shade_cmap("#ff0000")
    np.testing.assert_allclose(cmap(0.0)[:3], (1.0, 1.0, 1.0))
    np.testing.assert_allclose(cmap(1.0)[:3], (1.0, 0.0, 0.0))


@pytest.mark.parametrize("name", sorted(obs.FIGURES))
def test_figures_write(tmp_path, name):
    """Each figure function writes a non-empty PDF."""
    out = tmp_path / f"{name}.pdf"
    obs.fwl_mpl.use()
    obs.FIGURES[name](out)
    assert out.stat().st_size > 1000


def test_noise_calibration_is_consistent_and_physical():
    """The factor is observed / predicted sigma, and the predicted depth is possible.

    The bound is a zero-albedo bare rock without redistribution on TRAPPIST-1c
    (T_irr = 480 K, Teske et al. 2025) relative to the observed 380 K (Zieba et
    al. 2023); a pi too many in the planet solid angle gives about 1250 ppm.
    """
    cal = json.loads((obs.DATA / "miri_noise_calibration.json").read_text())
    assert cal["noise_factor"] == pytest.approx(
        cal["observed_sigma_ppm"] / cal["model_sigma_ppm"]
    )

    def planck_15um(temperature):
        return 1.0 / np.expm1(14387.77 / (15.0 * temperature))

    upper = (
        cal["observed_ppm"] * planck_15um(480.0 * (2 / 3) ** 0.25) / planck_15um(380.0)
    )
    assert 0.0 < cal["model_depth_ppm"] < upper


def test_calibrated_sigma_of_surfaces_is_about_0p08():
    """The stored photometry gives a calibrated 1-sigma near 0.08 for every surface.

    A pi too many in the planet flux of ``surface_photometry`` makes the raw
    1/SNR pi times smaller and this sigma about 0.025.
    """
    phot = obs.read_photometry()
    sig = [
        row[f"{band}_sigma"]
        for row in phot.values()
        if row["kind"] == "surface"
        for band in ("F1280W", "F1500W")
    ]
    assert 0.06 < min(sig) and max(sig) < 0.10


def test_reduced_chi2_matches_hand_value(tmp_path):
    """Two points, model linear in wavelength: chi2 / (N - 1) by hand."""
    path = tmp_path / "spec.npz"
    np.savez(
        path,
        wavelength_um=np.array([5.0, 7.0, 9.0]),
        names=np.array(["a", "b"]),
        contrast_ppm=np.array([[100.0, 200.0, 300.0], [0.0, 0.0, 0.0]]),
        obs_um=np.array([6.0, 8.0]),
        obs_ppm=np.array([160, 250]),
        obs_ppm_err=np.array([10, 25]),
    )
    chi2 = obs.reduced_chi2(path)
    # a: model 150 and 250 -> (1^2 + 0^2) / 1; b: (16^2 + 10^2) / 1
    assert chi2["a"] == pytest.approx(1.0)
    assert chi2["b"] == pytest.approx(356.0)


def test_reduced_chi2_refuses_extrapolation(tmp_path):
    """An observed point outside the model bands raises."""
    path = tmp_path / "spec.npz"
    np.savez(
        path,
        wavelength_um=np.array([5.0, 7.0]),
        names=np.array(["a"]),
        contrast_ppm=np.array([[1.0, 2.0]]),
        obs_um=np.array([8.0]),
        obs_ppm=np.array([1]),
        obs_ppm_err=np.array([1]),
    )
    with pytest.raises(ValueError, match="outside the model bands"):
        obs.reduced_chi2(path)


def test_reduced_chi2_of_the_stored_spectra():
    """Pyrite has the lowest value; the values the text quotes."""
    chi2 = obs.reduced_chi2()
    assert min(chi2, key=chi2.get) == "pyrite"
    assert chi2["pyrite"] == pytest.approx(1.74, abs=0.005)
    assert chi2["norite"] == pytest.approx(1.86, abs=0.005)
    assert chi2["gabbro"] == pytest.approx(1.87, abs=0.005)
    assert chi2["basalt_glass"] == pytest.approx(1.86, abs=0.005)
