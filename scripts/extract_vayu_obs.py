"""Store the VAYU surface spectra and the MIRI photometry for the observation figures.

Run from the root of a VAYU working copy (the folder with ``agni_config.toml``,
``src``, ``plt`` and ``out/gj367b``) with the VAYU Python environment, which has
pandeia and its reference data::

    cd /path/to/vayu && python /path/to/extract_vayu_obs.py

Writes, in ``figures/data``:

``vayu_surface_spectra.npz``
    Eclipse depth (ppm) of every bare-rock surface, the blackbody at the
    dayside temperature for f = 2/3, and the MIRI points of Zhang et al. (2024).
``miri_photometry.csv``
    Emission relative to that blackbody in F1280W and F1500W for every surface
    and every greybody atmosphere, with the photon-noise uncertainty of VAYU
    (planet, star and thermal background for 4 eclipses; filter throughputs
    from pandeia) before the calibration factor is applied. The uncertainty
    is 1/SNR, a fraction of the eclipse depth, not an absolute value.
``miri_noise_calibration.json``
    The calibration factor: the observed 1-sigma of the TRAPPIST-1c F1500W
    eclipse of Zieba et al. (2023), 94 ppm, divided by the 1-sigma in ppm that
    the same noise model predicts for that eclipse.
``bayes_model_comparison.csv``
    Copy of the VAYU evidence table (reference: pyrite).
"""

import glob
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "plt"))

import plot_emission_miri_surfaces as miri  # noqa: E402
from plot_contrast import (  # noqa: E402
    CONFIG,
    get_planet_data,
    get_star_spectrum_path,
    load_contrast_data,
    load_model_contrasts,
    r_earth,
    r_sun,
)
from src import constants  # noqa: E402
from src.surface_labels import surface_labels  # noqa: E402
from src.utils import compute_dayside_brightness_temperature, contrast_ppm, planck  # noqa: E402

PLANET = "gj367b"
OUT = Path(__file__).resolve().parents[1] / "data"
OBS_SIGMA_PPM = 94.0  # Zieba et al. (2023), TRAPPIST-1c F1500W


def spectra():
    """Write the surface eclipse spectra, the blackbody and the observed points."""
    p = get_planet_data(PLANET)
    t_star, r_star = p["star_temp"], p["star_radius"] * r_sun
    r_planet = p["planet_radius"] * r_earth
    star = get_star_spectrum_path(PLANET)
    names = [k for k in surface_labels if k != "greybody"]
    contrasts, band_nm = load_model_contrasts(
        PLANET, "surface", "bare_rock", names, t_star, r_planet, r_star, star
    )
    t_bb = compute_dayside_brightness_temperature(
        t_star, p["star_radius"], p["planet_a"], 0.0, 2 / 3
    )
    bb = contrast_ppm(
        wavelength_nm=band_nm,
        T_star=t_star,
        R_planet_m=r_planet,
        R_star_m=r_star,
        planet_flux=planck(band_nm, t_bb),
        stellar_spectrum=star,
    )
    obs = load_contrast_data(os.path.join(CONFIG["obs_data_dir"], f"{PLANET}_data.csv"))
    keep = (band_nm > 4000.0) & (band_nm < 13000.0)
    np.savez_compressed(
        OUT / "vayu_surface_spectra.npz",
        wavelength_um=band_nm[keep] / 1000.0,
        names=np.array(names),
        labels=np.array([surface_labels[k] for k in names]),
        contrast_ppm=np.array([contrasts[k][keep] for k in names]),
        blackbody_ppm=bb[keep],
        blackbody_k=t_bb,
        obs_um=obs["X"].to_numpy(),
        obs_um_err=obs["ΔX"].to_numpy(),
        obs_ppm=obs["Y"].to_numpy(),
        obs_ppm_err=obs["ΔY"].to_numpy(),
    )
    print(f"{len(names)} surface spectra, blackbody {t_bb:.1f} K")


def noise_calibration(config, wave):
    """Return the predicted F1500W depth and 1-sigma of the TRAPPIST-1c eclipse.

    Parameters
    ----------
    config : dict
        VAYU configuration from ``miri.load_config()``.
    wave : numpy.ndarray
        Wavelength grid in micron.

    Returns
    -------
    dict
        ``depth_ppm`` and ``sigma_ppm`` for 4 eclipses of the granite bare rock.

    Notes
    -----
    The noise is set mainly by the star, so the predicted sigma in ppm does not
    depend on the planet model; the predicted depth does.
    """
    tp = miri.get_throughput(wave, "f1500w")
    t = get_planet_data("trappist-1c")
    d_m = t["planet_d"] * constants.pc
    rp_m = t["planet_radius"] * constants.r_earth
    star = (
        miri.load_stellar_flux(
            os.path.join(config["stellar_spectra_dir"], "trappist-1_SPHINX.txt"), wave
        )
        * (constants.au / d_m) ** 2
    )
    data = miri.load_agni_output("out/trappist-1c/granite/bare_rock/atm.nc")
    flux = np.interp(wave, data["bandcenter"] / 1000, data["ba_U_total"] * 1000)
    snr = miri.compute_snr(wave, flux * (rp_m / d_m) ** 2, star, tp)
    depth_ppm = snr["model_photons"] / snr["star_photons"] * 1e6
    return {"depth_ppm": depth_ppm, "sigma_ppm": depth_ppm / snr["snr"]}


def surface_photometry(nc_path, wave, tp, t_planet, star, rp_m, d_m):
    """Return the relative emission and 1/SNR of one AGNI model in each filter.

    Parameters
    ----------
    nc_path : str
        AGNI ``atm.nc`` output.
    wave : numpy.ndarray
        Wavelength grid in micron.
    tp : dict
        Filter name to throughput on ``wave``.
    t_planet : float
        Blackbody temperature (K) of the reference dayside.
    star : numpy.ndarray
        Stellar flux at the observer (W/m2/micron).
    rp_m, d_m : float
        Planet radius and distance (m).

    Returns
    -------
    dict or None
        ``F1280W``, ``F1500W`` and ``<filter>_uncert`` (1/SNR, a fraction of the
        eclipse depth), or None if the model is missing.

    Notes
    -----
    AGNI's upward flux and ``planck`` (which includes pi) are hemispheric
    fluxes, so the flux at the observer is F (Rp/d)^2, not F pi (Rp/d)^2 as in
    ``plot_emission_miri_surfaces.process_surface``.
    """
    if not os.path.exists(nc_path):
        print(f"[MISSING] {nc_path}")
        return None
    data = miri.load_agni_output(nc_path)
    dilution = (rp_m / d_m) ** 2
    flux = (
        np.interp(wave, data["bandcenter"] / 1000, data["ba_U_total"] * 1000) * dilution
    )
    bb = planck(wave * 1000, t_planet) * 1000 * dilution
    row = {}
    for filt, thr in tp.items():
        row[filt] = miri.integrate_flux(wave, flux, thr) / miri.integrate_flux(
            wave, bb, thr
        )
        row[f"{filt}_uncert"] = miri.compute_snr(wave, flux, star, thr)["uncertainty"]
    return row


def photometry():
    """Write the relative MIRI emission of all surfaces and greybody atmospheres."""
    config = miri.load_config()
    wave = np.linspace(10.0, 20.0, 1000)
    calib = noise_calibration(config, wave)
    (OUT / "miri_noise_calibration.json").write_text(
        json.dumps(
            {
                "noise_factor": OBS_SIGMA_PPM / calib["sigma_ppm"],
                "reference": "TRAPPIST-1c granite bare rock, F1500W",
                "observed_ppm": 421.0,
                "observed_sigma_ppm": OBS_SIGMA_PPM,
                "model_depth_ppm": calib["depth_ppm"],
                "model_sigma_ppm": calib["sigma_ppm"],
            },
            indent=1,
        )
        + "\n"
    )
    tp = {
        "F1280W": miri.get_throughput(wave, "f1280w"),
        "F1500W": miri.get_throughput(wave, "f1500w"),
    }
    p = get_planet_data(PLANET)
    d_m = p["planet_d"] * constants.pc
    rp_m = p["planet_radius"] * constants.r_earth
    t_planet = compute_dayside_brightness_temperature(
        p["star_temp"], p["star_radius"], p["planet_a"], 0.0, 2 / 3
    )
    star = (
        miri.load_stellar_flux(
            os.path.join(config["stellar_spectra_dir"], "gj367_SPHINX.txt"), wave
        )
        * (constants.au / d_m) ** 2
    )
    cases = [(s, "bare_rock", "surface", s) for s in surface_labels if s != "greybody"]
    for nc in sorted(glob.glob(f"out/{PLANET}/greybody/*/atm.nc")):
        name = Path(nc).parent.name
        cases.append(("greybody", name, "atmosphere", name))
    rows = []
    for surface, atmosphere, kind, name in cases:
        row = surface_photometry(
            os.path.join("out", PLANET, surface, atmosphere, "atm.nc"),
            wave,
            tp,
            t_planet,
            star,
            rp_m,
            d_m,
        )
        if row is None:
            continue
        rows.append(
            {
                "kind": kind,
                "name": name,
                "F1280W": row["F1280W"],
                "F1500W": row["F1500W"],
                "F1280W_sigma_raw": row["F1280W_uncert"],
                "F1500W_sigma_raw": row["F1500W_uncert"],
            }
        )
    pd.DataFrame(rows).to_csv(
        OUT / "miri_photometry.csv", index=False, float_format="%.6g"
    )
    print(f"{len(rows)} photometry rows")


if __name__ == "__main__":
    shutil.copyfile(
        f"out/{PLANET}/bayes_model_comparison.csv", OUT / "bayes_model_comparison.csv"
    )
    spectra()
    photometry()
