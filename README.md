# Magma oceanography of the sub-Earth exoplanet GJ 367 b: data and scripts

This repository holds the data and code that produce every figure and the derived numbers in the text of Lichtenberg et al., "Magma oceanography of the sub-Earth exoplanet GJ 367 b" (in preparation). Running the plot scripts on the data in `data/` reproduces the figures in `figures/`.

## Layout

| Folder | Content |
|---|---|
| `data/` | Compact extracts of the StagYY runs, the Perple_X mineralogy grid, the VAYU surface spectra, photometry and Bayesian evidence tables |
| `data/surface_maps/` | Surface basalt and melt fraction against longitude and time for each run |
| `figures/` | The figures of the manuscript, as produced by the scripts |
| `runs/test_NN/` | StagYY parameter files of each run: `par_GJ367b` (shared template) and `par` (run settings) |
| `scripts/` | Extraction scripts (`extract_*.py`), plot scripts (`plot_*.py`), shared helpers and tests |

## Quick start

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cd scripts
../.venv/bin/python -m pytest
../.venv/bin/python plot_fig01_mineralogy.py
../.venv/bin/python plot_fig02_annulus.py
../.venv/bin/python plot_overview_figures.py
../.venv/bin/python plot_obs_figures.py
```

The plot scripts read only `data/` and write to `figures/`. `manuscript_numbers.py` prints the derived numbers quoted in the text (photometric separations, effective albedos, shoreline positions, energy budget, and the outputs left out of Fig. 5).

## Figures

| Manuscript figure | File | Plot script | Data | Extraction |
|---|---|---|---|---|
| 1 | `fig01.pdf` | `plot_fig01_mineralogy.py` | `StagYY_mineralogy.csv` | Perple_X grid |
| 2 | `fig02.pdf` | `plot_fig02_annulus.py` | `stagyy_fig02_annulus.npz` | `extract_fig02_annulus.py` |
| 3 | `fig_surface.pdf` | `plot_overview_figures.py surface` | `surface_maps/` | `extract_surface_maps.py` |
| 4 | `fig_profiles.pdf` | `plot_overview_figures.py profiles` | `surface_maps/` | `extract_surface_maps.py` |
| 5 | `fig_flux.pdf` | `plot_overview_figures.py flux` | `stagyy_flux/`, `surface_maps/` | `extract_flux_cache.py` |
| 6 | `fig_interior.pdf` | `plot_overview_figures.py interior` | `stagyy_1d/` | `extract_1d.py` |
| 7 | `fig_obs_surfaces.pdf` | `plot_obs_figures.py surfaces` | `vayu_surface_spectra.npz`, `bayes_model_comparison.csv` | `extract_vayu_obs.py` |
| 8 | `fig_obs_atmospheres.pdf` | `plot_obs_figures.py atmospheres` | `bayes_model_comparison.csv`, `bayes_redistribution.csv` | `extract_vayu_obs.py` |
| 9 | `fig_obs_photometry.pdf` | `plot_obs_figures.py photometry` | `miri_photometry.csv`, `miri_noise_calibration.json` | `extract_vayu_obs.py` |

## Models

Run `test_NN` in `runs/` and the files `run_NN` in `data/` are the models of Table 1 of the manuscript.

| Run | Model | Run | Model |
|---|---|---|---|
| 01 | HN | 09 | HE+ |
| 02 | CN | 10 | CE+ |
| 04 | CN-thick | 12 | CN+ |
| 05 | HE | 13 | HL+ |
| 06 | HL | 14 | CL+ |
| 07 | CE | 15 | CN-core |
| 08 | CL | 16 | HN-core |

## From the raw output

The `extract_*.py` scripts read the raw output and write the files in `data/`. The raw output (StagYY field, `hf` and `time.dat` files, the `flux_data.p` caches, and the AGNI models) is archived separately on Zenodo; the record is linked here at publication. Unpack it into `raw/` or set `GJ367B_DATA` to the folder that holds `test_NN`.

The mantle convection models were run with StagYY (P. J. Tackley), which is not public; the parameter files in `runs/` give every setting of each run. The observation models use VAYU (https://github.com/stuitje/VAYU, commit `565f99702e7da5bb930105a334d39d2b909b8aa3`) and AGNI. Run `extract_vayu_obs.py` from the root of a VAYU working copy that holds the AGNI output in `out/`. It computes the photometry with the planet solid angle (Rp/d)^2, because AGNI's upward flux is a hemispheric flux.

The photometric uncertainty is the VAYU photon-noise estimate (1/SNR, a fraction of the eclipse depth) times the relative emission times a calibration factor. The factor is the observed 1-sigma of the TRAPPIST-1 c F1500W eclipse (94 ppm; Zieba et al. 2023) divided by the 1-sigma that the same noise model predicts for it (`data/miri_noise_calibration.json`).

## Licence

Code: Apache License 2.0 (`LICENSE`). Data and figures: Creative Commons Attribution 4.0 International (CC BY 4.0).
