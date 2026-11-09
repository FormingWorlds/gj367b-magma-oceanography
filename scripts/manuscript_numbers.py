"""Print the derived numbers quoted in the text of the GJ 367 b manuscript.

Every value comes from the files in ``data/`` or from the stated formula, so the
printed lines can be compared with the manuscript sentence by sentence.
"""

import math

import numpy as np
import plot_obs_figures as obs
from plot_overview_figures import MOLTEN_SHARE, dayside_molten_share, load_map
from stagyy_common import DATA_DIR, GROUPS, MODEL_NAMES, SEC_PER_GYR

# Planet and run settings (Lam et al. 2021 structure, par files in runs/).
R_PLANET = 4574.0e3
R_CMB = 3934.0e3
# Reference mantle density of the runs (par_GJ367b dens_dimensional).
RHO_MANTLE = 3300.0
CP_MANTLE = 1200.0
H_RADIOGENIC = 5.2e-12
T_NIGHT = 100.0
T_DAY = {"1597": 1597.0, "1745": 1745.0}
# Zero-depth melting temperature of basalt-enriched material in StagYY's
# Cf-dependent law: 2 T_sol - T_bo with T_sol 1390 K, T_liq 1970 K, T_bo 1506 K.
T_MELT_ENRICHED = 1274.0
A_B_MEASURED = 0.11
T_DAY_MEASURED = 1728.0
T_DAY_MEASURED_ERR = 90.0


def mantle_mass():
    """Return the mantle mass (kg) of the 640 km mantle at RHO_MANTLE."""
    return 4.0 / 3.0 * math.pi * (R_PLANET**3 - R_CMB**3) * RHO_MANTLE


def surface_temperature(theta_deg, t_day):
    """Return the fixed surface temperature at angle theta from the substellar point."""
    c = np.cos(np.radians(theta_deg))
    return T_NIGHT + (t_day - T_NIGHT) * np.clip(c, 0.0, None) ** 0.25


def shoreline_deg(t_day, t_melt=T_MELT_ENRICHED):
    """Return the angle where the fixed surface temperature equals ``t_melt``."""
    return math.degrees(math.acos(((t_melt - T_NIGHT) / (t_day - T_NIGHT)) ** 4))


def separation(row, lo, hi):
    """Return the larger of the two filter separations (in sigma) from [lo, hi]."""
    best = 0.0
    for band in ("F1280W", "F1500W"):
        v, s = row[band], row[f"{band}_sigma"]
        d = lo - v if v < lo else (v - hi if v > hi else 0.0)
        best = max(best, d / s)
    return best


def photometry():
    """Print the separations of Section 3.6.3 and Fig. 9."""
    phot = obs.read_photometry()
    lo, hi = obs.consistent_range(phot, obs.surface_evidence())
    sig = [w[f"{b}_sigma"] for w in phot.values() for b in ("F1280W", "F1500W")]
    print(
        f"consistent surfaces {lo:.3f}-{hi:.3f}; 1-sigma {min(sig):.3f}-{max(sig):.3f}"
    )
    names = {
        f"{p}bar_{pre}{gas}{suf if koll else ''}"
        for koll in (False, True)
        for pre in ("N2_1000ppm_", "")
        for gas in ("CO2", "SO2", "H2O", "CH4")
        for p, _, suf in obs.KOLL
    }
    n3 = sum(separation(phot[n], lo, hi) >= 3.0 for n in names)
    print(f"Fig. 9 cases >= 3 sigma: {n3} of {len(names)}")
    for gas in ("CO2", "SO2", "H2O", "CH4"):
        seps = [separation(phot[f"{p}bar_{gas}"], lo, hi) for p, _, _ in obs.KOLL]
        print(f"  pure {gas}, f = 2/3: {min(seps):.1f}-{max(seps):.1f} sigma")


def temperatures():
    """Print the albedo and temperature values of the Methods and Discussion."""
    t_ss0 = T_DAY["1745"] * 1.5**0.25
    print(f"substellar T for A_B = 0: {t_ss0:.0f} K")
    for label, t in T_DAY.items():
        a_eff = 1.0 - (t / t_ss0) ** 4
        t_eff = t * (2.0 / 3.0) ** 0.25
        offset = (T_DAY_MEASURED - t_eff) / T_DAY_MEASURED_ERR
        print(
            f"  T_day {label} K: effective A_B {a_eff:.2f}, dayside {t_eff:.0f} K ({offset:.1f} sigma)"
        )
    t_meas = t_ss0 * (1.0 - A_B_MEASURED) ** 0.25
    print(f"substellar T for A_B = {A_B_MEASURED}: {t_meas:.0f} K")
    for t in (*T_DAY.values(), t_meas):
        print(f"  shoreline at {t:.0f} K: {shoreline_deg(t):.1f} deg")


def shoreline_series(surface_map):
    """Return the melt-fraction 0.5 edge (deg) on the dayside for every map row."""
    lon = np.abs(surface_map["lon_deg"])
    edges = []
    for row in surface_map["meltfrac"]:
        molten = lon[(row > 0.5) & (lon < 90.0)]
        edges.append(molten.max() if molten.size else 0.0)
    return np.asarray(edges)


def shorelines():
    """Print the melt-fraction 0.5 edge of the segregation runs, late and at equal time."""
    for run in GROUPS["L"]:
        m = load_map(run)
        t = m["time_gyr"]
        edges = shoreline_series(m)
        edge = float(np.median(edges[t >= 0.75 * t.max()]))
        same = float(np.median(edges[(t >= 0.6) & (t < 1.13)]))
        t_day = T_DAY["1745"] if "+" in MODEL_NAMES[run] else T_DAY["1597"]
        print(
            f"  {MODEL_NAMES[run]}: last quarter {edge:.1f} deg "
            f"(T_s {surface_temperature(edge, t_day):.0f} K), 0.6-1.13 Gyr {same:.1f} deg"
        )


def fluxes():
    """Print the radiogenic flux, the energy budget and the Fig. 5 mask counts."""
    area = 4.0 * math.pi * R_PLANET**2
    mass = mantle_mass()
    print(
        f"mantle mass {mass:.2e} kg; radiogenic flux {H_RADIOGENIC * mass / area * 1e3:.1f} mW/m2"
    )
    for run in (6, 7):
        d = np.load(DATA_DIR / "stagyy_1d" / f"run_{run:02d}.npz")
        t, f, tm = d["ts_time_gyr"], d["ts_F_top"], d["ts_Tmean"]
        order = np.argsort(t, kind="stable")
        t, f, tm = t[order], f[order], tm[order]
        keep = t >= 0.5
        t, f, tm = t[keep], f[keep], tm[keep]
        dt = np.diff(t) * SEC_PER_GYR
        f_mean = (f[1:] * dt).sum() / dt.sum()
        lost = f_mean * area * dt.sum()
        supplied = (tm[0] - tm[-1]) * mass * CP_MANTLE + H_RADIOGENIC * mass * dt.sum()
        print(
            f"  {MODEL_NAMES[run]}: time-weighted F_top {f_mean:.2f} W/m2, lost/supplied {lost / supplied:.0f}"
        )
    for run in (*GROUPS["N"], *GROUPS["E"], *GROUPS["L"]):
        d = np.load(DATA_DIR / "stagyy_flux" / f"run_{run:02d}.npz")
        share = dayside_molten_share(load_map(run), d["time_gyr"])
        masked = share >= MOLTEN_SHARE
        big = d["top_day"] > 1.0e3
        print(
            f"  {MODEL_NAMES[run]}: masked {masked.sum()} of {masked.size}, "
            f"outputs above 1 W/m2 not masked: {(big & ~masked).sum()}"
        )


def chi_squared():
    """Print the reduced chi-squared of the surfaces named in the text."""
    chi2 = obs.reduced_chi2()
    best = min(chi2, key=chi2.get)
    print(f"lowest reduced chi2: {best} {chi2[best]:.2f}")
    for name in ("pyrite", "norite", "gabbro", "basalt_glass"):
        print(f"  {name}: {chi2[name]:.2f}")


if __name__ == "__main__":
    for part in (photometry, chi_squared, temperatures, shorelines, fluxes):
        print(f"== {part.__name__}")
        part()
