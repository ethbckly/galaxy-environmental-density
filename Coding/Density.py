print("Running....")

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table, hstack
from astropy.coordinates import SkyCoord
import astropy.units as u
import os
from astropy.cosmology import Planck18 as cosmo 
import sys
from scipy.stats import ks_2samp
import matplotlib.ticker as ticker

os.chdir(r"C:\Users\ethan\Desktop\Master's Work")

##############################################################################
##############################################################################

def import_files(file_path):
    table = Table.read(file_path)

    ra = np.asarray(table['ALPHA_J2000'], dtype=float)
    dec = np.asarray(table['DELTA_J2000'], dtype=float)
    z = np.asarray(table['lp_zBEST'], dtype=float) # Z_TYPE column, phot, spec
    cosmos_ids = table['ID']
    mass = np.asarray(table['lp_mass_best'], dtype=float)

    return ra, dec, z, cosmos_ids, mass, table

def sky(ra, dec):
    coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)

    return coords

def redshift_slicing(z, z_i, delta_z):
    return np.abs(z-z_i) < delta_z

def nearest_neighbour_density_monte(coords, z, n=10):
    N = len(coords)
    densities = np.full(N, np.nan)

    r_n_deg_all = np.full(N, np.nan)
    f_area_all = np.full(N, np.nan)

    print_every = max(N // 100, 1)

    for i in range(N):


        if i % print_every == 0:
            percent = 100 * i / N
            print(f"{percent:.1f}% complete")


        delta_z = k * (1+z[i])
        mask = redshift_slicing(z, z[i], delta_z)

        if np.sum(mask) <= n:
            densities[i] = np.nan
            continue
    
        coords_slice = coords[mask]
    
        sep = coords[i].separation(coords_slice).deg
        sep = sep[sep > 0]

        r_n_deg = np.sort(sep)[n-1]
        r_n_deg_all[i] = r_n_deg

        f_area = area_fraction(
            coords[i].ra.deg,
            coords[i].dec.deg,
            r_n_deg,
            ra_min, ra_max,
            dec_min, dec_max)
        

        '''f_area = area_fraction_masked(
            coords[i].ra.deg,
            coords[i].dec.deg,
            r_n_deg,
            mask_map, ra_edges, dec_edges
        '''

        if i < 10:
            print("f_area:", f_area)

        if f_area < 0.5:
            densities[i] = np.nan
            continue

        Distance_mpc_deg = cosmo.angular_diameter_distance(z[i])* (np.pi / 180)

        r_n_mpc = r_n_deg * Distance_mpc_deg
        densities[i] = (n / (f_area * (np.pi * r_n_mpc**2))).value

    return densities

def nearest_neighbour_density_old(coords, z, n=10):
    N = len(coords)
    densities = np.full(N, np.nan)

    r_n_deg_all = np.full(N, np.nan)

    for i in range(N):
        delta_z = k * (1+z[i])
        mask = redshift_slicing(z, z[i], delta_z)

        if np.sum(mask) <= n:
            densities[i] = np.nan
            continue
    
        coords_slice = coords[mask]
    
        sep = coords[i].separation(coords_slice).deg
        sep = sep[sep > 0]

        r_n_deg = np.sort(sep)[n-1]

        r_n_deg_all[i] = r_n_deg

        Distance_mpc_deg = cosmo.angular_diameter_distance(z[i]) * (np.pi / 180)

        r_n_mpc = r_n_deg * Distance_mpc_deg

        densities[i] = (n / (np.pi * r_n_mpc**2)).value
    
    r_safe_deg = np.nanmedian(r_n_deg_all)

    edge_mask = (
       (coords.ra.deg > ra_min + r_safe_deg) &
        (coords.ra.deg < ra_max - r_safe_deg) &
        (coords.dec.deg > dec_min + r_safe_deg) &
        (coords.dec.deg < dec_max - r_safe_deg)
    )

    densities[~edge_mask] = np.nan
    
    return densities

def nearest_neighbour_density_edges(coords, z, n=10):
    N = len(coords)
    densities = np.full(N, np.nan)

    r_n_deg_all = np.full(N, np.nan)

    for i in range(N):
        delta_z = k * (1+z[i])
        mask = redshift_slicing(z, z[i], delta_z)

        if np.sum(mask) <= n:
            densities[i] = np.nan
            continue
    
        coords_slice = coords[mask]
    
        sep = coords[i].separation(coords_slice).deg
        sep = sep[sep > 0]

        r_n_deg = np.sort(sep)[n-1]
#----------------------------------------------
        f_area = area_fraction_circle(
            coords[i].ra.deg,
            coords[i].dec.deg,
            r_n_deg,
            ra_min, ra_max,
            dec_min, dec_max
            )

        if f_area < 0.5:
            densities[i] = np.nan
            continue
#-------------------------------------------------
        r_n_deg_all[i] = r_n_deg

        Distance_mpc_deg = cosmo.angular_diameter_distance(z[i]) * (np.pi / 180)

        r_n_mpc = r_n_deg * Distance_mpc_deg

        densities[i] = (n / (np.pi * r_n_mpc**2)).value
    
    return densities

def area_fraction(ra0, dec0, r_n_deg, ra_min, ra_max, dec_min, dec_max, n_samples=2000):
    theta = np.random.uniform(0, 2*np.pi, n_samples)
    u = np.random.uniform(0, 1, n_samples)
    r = r_n_deg * np.sqrt(u)

    ra_samp = ra0 + r * np.cos(theta)
    dec_samp = dec0 + r * np.sin(theta)

    inside = (
        (ra_samp  >= ra_min) & (ra_samp  <= ra_max) &
        (dec_samp >= dec_min) & (dec_samp <= dec_max)
    )

    return np.mean(inside)

def distance_to_edges(ra, dec, ra_min, ra_max, dec_min, dec_max):
    return np.array([
        ra - ra_min,
        ra_max - ra,
        dec - dec_min,
        dec_max - dec
    ])

def area_fraction_circle(ra, dec, r, ra_min, ra_max, dec_min, dec_max):
    dist = distance_to_edges(ra, dec, ra_min, ra_max, dec_min, dec_max)
    distmin = np.min(dist)

    if r <= distmin:
        return 1.0
    
    f = distmin / r
    return np.clip(f, 0.0, 1.0)


####################################################################################
####################################################################################

file_path = "COSMOS_SMOLCIC"

ra, dec, z, cosmo_ids, mass, table = import_files(file_path)
coords = sky(ra, dec)

ra_min, ra_max = np.nanmin(ra), np.nanmax(ra)
dec_min, dec_max = np.nanmin(dec), np.nanmax(dec)

k = 0.05

xray = np.asarray(table["XrayAGN"] == "T")
mir = np.asarray(table["MIRAGN"] == "T")
sed = np.asarray(table["SEDAGN"] == "T")
hlagn = np.asarray(table["HLAGN"] == "T")
mlagn = np.asarray(table["MLAGN"] == "T")
qml = np.asarray(table["QMLAGN"] == "T")

agn_full = (xray | mir | sed | hlagn | mlagn | qml)
nonagn_full = ~agn_full

mask = (
    (mass > 7.0) &
    (mass < 12.5)
)

ra = ra[mask]
dec = dec[mask]
z = z[mask]
mass = mass[mask]
xray = xray[mask]
hlagn = hlagn[mask]
mlagn = mlagn[mask]
agn = agn_full[mask]
nonagn = ~agn

coords = sky(ra, dec)

print("Samples size after cuts:", len(mass))
print("Number of X-Ray AGN:", np.sum(xray))

density = nearest_neighbour_density_monte(coords, z, n=5)
print("Length coords:", len(coords))
print("Length density:", len(density))
log_density = np.log10(density)

valid = np.isfinite(density)

log_density = log_density[valid]
mass = mass[valid]
agn = agn[valid]
nonagn = nonagn[valid]
hlagn = hlagn[valid]
xray = xray[valid]
mlagn = mlagn[valid]
z = z[valid]

# ----- Define bins -----

z_bins = [
    (0.0,1.0),
    (1.0,2.0),
    (2.0,3.0)
]

mass_bins = [
    (9.5,10),
    (10,11),
    (11,12.5)
]

bins = np.linspace(-1.4, 0.6, 9)
'''
#------ Plotting AGN vs Non-AGN ----------#

fig, axes = plt.subplots(
    len(z_bins),
    len(mass_bins),
    figsize=(18,12),
    sharex=True,
    sharey=True
)

for zi, (zmin, zmax) in enumerate(z_bins):

    z_mask = (z >= zmin) & (z < zmax)

    for mi, (mmin, mmax) in enumerate(mass_bins):

        ax = axes[zi, mi]

        mass_mask = (mass >= mmin) & (mass < mmax)

        mask_total = z_mask & mass_mask

        # ----- Extract densities -----

        agn_density = log_density[mask_total & agn]
        nonagn_density = log_density[mask_total & nonagn]


        # ----- Histogram plots -----

        ax.set_ylim(0,3)
        ax.set_xlim(-1.4,0.6)

        if len(nonagn_density) > 0:
            ax.hist(
                nonagn_density,
                bins=bins,
                density=True,
                histtype='step',
                linewidth=1.4,
                linestyle='-',
                color='orange',
                alpha=0.9
            )

        if len(agn_density) > 0:
            ax.hist(
                agn_density,
                bins=bins,
                density=True,
                histtype='step',
                linewidth=1.4,
                linestyle='--',
                color='black',
                alpha=0.9
            )


        # ----- Axis labels -----

        if zi == 0:
            ax.set_title(f"log(M*) {mmin}-{mmax}", fontsize=12)

        if mi == 0:
            ax.set_ylabel(f"z {zmin}-{zmax}\nPDF")

        if zi == len(z_bins) - 1:
            ax.set_xlabel("log Σ")


        # ----- Tick styling -----

        ax.tick_params(axis='both', which='major', length=6)

        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

        ax.tick_params(axis='both', which='minor', length=3)



# ----- Global legend -----

fig.legend(
    ["Non-AGN", "AGN"],
    loc="upper center",
    bbox_to_anchor=(0.5,1.02),
    ncol=2,
    frameon=False,
    fontsize=12
)

plt.tight_layout()
plt.show()



# ----- Plotting HLAGN,MLAGN vs Non-AGN -----

fig, axes = plt.subplots(
    len(z_bins),
    len(mass_bins),
    figsize=(18,12),
    sharex=True,
    sharey=True
)

ks_results = []

for zi, (zmin, zmax) in enumerate(z_bins):

    z_mask = (z >= zmin) & (z < zmax)

    for mi, (mmin, mmax) in enumerate(mass_bins):

        ax = axes[zi, mi]
        ypos = 0.92

        mass_mask = (mass >= mmin) & (mass < mmax)
        mask_total = z_mask & mass_mask

        # ----- Extract densities -----

        nonagn_density = log_density[mask_total & nonagn]
        mlagn_density = log_density[mask_total & mlagn]
        hlagn_density = log_density[mask_total & hlagn]

        N_nonagn = len(nonagn_density)
        N_mlagn = len(mlagn_density)
        N_hlagn = len(hlagn_density)

        # ----- Plot histograms -----

        ax.set_ylim(0,3)
        ax.set_xlim(-1.4,0.6)

        if len(nonagn_density) > 0:
            ax.hist(
                nonagn_density,
                bins=bins,
                density=True,
                histtype='step',
                linewidth=1.2,
                linestyle='-',
                color='orange',
                alpha=0.9
            )


        if len(mlagn_density) > 0:
            ax.hist(
                mlagn_density,
                bins=bins,
                density=True,
                histtype='step',
                linewidth=1.2,
                linestyle='--',
                color='blue',
                alpha=0.9
            )

        if len(hlagn_density) > 0:
            ax.hist(
                hlagn_density,
                bins=bins,
                density=True,
                histtype='step',
                linewidth=1.2,
                linestyle=':',
                color='purple',
                alpha=0.9
            )

        # ----- Axis labels -----

        if zi == 0:
            ax.set_title(f"log(M*) {mmin}-{mmax}", fontsize=12)

        if mi == 0:
            ax.set_ylabel(f"z {zmin}-{zmax}\nPDF")

        if zi == len(z_bins) - 1:
            ax.set_xlabel("log Σ")

        ax.text(
            0.97, 0.97,
            f"N_nonAGN = {N_nonagn}\nN_MLAGN = {N_mlagn}\nN_HLAGN = {N_hlagn}",
            transform=ax.transAxes,
            ha='right',
            va='top',
            fontsize=9
        )

        # ----- Tick styling -----

        ax.tick_params(axis='both', which='major', length=6)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(axis='both', which='minor', length=3)

        # =========================================================
        # KS TESTS
        # =========================================================

        if len(hlagn_density) > 5 and len(nonagn_density) > 5:
            ks_stat, p = ks_2samp(hlagn_density, nonagn_density)

            ks_results.append(
                ["HLAGN vs NonAGN", zmin, zmax, mmin, mmax, ks_stat, p]
            )

            print(f"HLAGN vs NonAGN | z {zmin}-{zmax} | M {mmin}-{mmax}")
            print(f"KS={ks_stat:.3f}  p={p:.4f}\n")

            ax.text(
                0.05,
                ypos,
                f"p = {p:.3f}",
                transform=ax.transAxes,
                fontsize=10
            )

            ypos -= 0.08

        if len(mlagn_density) > 5 and len(nonagn_density) > 5:
            ks_stat, p = ks_2samp(mlagn_density, nonagn_density)

            ks_results.append(
                ["MLAGN vs NonAGN", zmin, zmax, mmin, mmax, ks_stat, p]
            )

            print(f"MLAGN vs NonAGN | z {zmin}-{zmax} | M {mmin}-{mmax}")
            print(f"KS={ks_stat:.3f}  p={p:.4f}\n")

            ax.text(
                0.05,
                ypos,
                f"p = {p:.3f}",
                transform=ax.transAxes,
                fontsize=10
            )

            ypos -= 0.08



# ----- Global legend -----

fig.legend(
    ["Non-AGN", "MLAGN", "HLAGN"],
    loc="upper center",
    bbox_to_anchor=(0.5,1.02),
    ncol=3,
    frameon=False,
    fontsize=12
)

plt.tight_layout()
plt.show()
'''


#------------OVERDENSITY CALCULATIONS + PLOTTING ---------------#

def bootstrap_error(values, n_boot=1000):

    values = values[~np.isnan(values)]

    if len(values) < 5:
        return np.nan

    medians = []

    for i in range(n_boot):
        sample = np.random.choice(values, size=len(values), replace=True)
        medians.append(np.nanmedian(sample))

    return np.std(medians)


# ---------- CALCULATE OVERDENSITY ---------- #

density = density[valid]

log_overdensity = np.full(len(density), np.nan)

for (zmin, zmax) in z_bins:

    z_mask = (z >= zmin) & (z < zmax)

    for (mmin, mmax) in mass_bins:

        mass_mask = (mass >= mmin) & (mass < mmax)

        mask = z_mask & mass_mask

    if np.sum(mask & nonagn) < 10:
        continue

    mean_density = np.nanmedian(density[z_mask & nonagn & mass_mask])

    log_overdensity[z_mask] = np.log10(density[z_mask] / mean_density)


# ---------- STORAGE ARRAYS ---------- #

hlagn_mean = []
mlagn_mean = []
nonagn_mean = []

hlagn_err = []
mlagn_err = []
nonagn_err = []

z_centers = []

# optional but useful diagnostic
hl_counts = []
ml_counts = []
na_counts = []


# ---------- COMPUTE STATS PER BIN ---------- #

for (zmin, zmax) in z_bins:

    z_mask = (z >= zmin) & (z < zmax)

    for (mmin, mmax) in mass_bins:

        mass_mask = (mass >= mmin) & (mass < mmax)


        hl = z_mask & hlagn & mass_mask
        ml = z_mask & mlagn & mass_mask
        na = z_mask & nonagn & mass_mask

    hl_vals = log_overdensity[hl]
    ml_vals = log_overdensity[ml]
    na_vals = log_overdensity[na]

    # medians
    hlagn_mean.append(np.nanmedian(hl_vals))
    mlagn_mean.append(np.nanmedian(ml_vals))
    nonagn_mean.append(np.nanmedian(na_vals))

    # bootstrap errors
    hlagn_err.append(bootstrap_error(hl_vals))
    mlagn_err.append(bootstrap_error(ml_vals))
    nonagn_err.append(bootstrap_error(na_vals))

    z_centers.append((zmin + zmax)/2)

    # counts (diagnostic)
    hl_counts.append(np.sum(~np.isnan(hl_vals)))
    ml_counts.append(np.sum(~np.isnan(ml_vals)))
    na_counts.append(np.sum(~np.isnan(na_vals)))


# ---------- PRINT SAMPLE SIZES ---------- #

print("HLAGN per bin:", hl_counts)
print("MLAGN per bin:", ml_counts)
print("NonAGN per bin:", na_counts)


# ---------- PLOT ---------- #

plt.figure(figsize=(7,5))

plt.errorbar(
    z_centers,
    nonagn_mean,
    yerr=nonagn_err,
    marker='s',
    label="Non-AGN",
    color="orange",
    capsize=4
)

plt.errorbar(
    z_centers,
    mlagn_mean,
    yerr=mlagn_err,
    marker='o',
    label="MLAGN",
    color="blue",
    capsize=4
)

plt.errorbar(
    z_centers,
    hlagn_mean,
    yerr=hlagn_err,
    marker='^',
    label="HLAGN",
    color="purple",
    capsize=4
)

plt.axhline(0, linestyle='--', color='gray')

plt.xlabel("Redshift")
plt.ylabel("log(1 + δ)")
plt.title("Galaxy Environment vs Redshift")

plt.legend()
plt.tight_layout()

plt.show()

#-----------------------MASS VS REDSHIFT PLOT--------------------------------------#

plt.figure(figsize=(7,6))

plt.hist2d(z, mass, bins=100, cmap="viridis")

plt.colorbar(label="Number of galaxies")

plt.xlabel("Redshift")
plt.ylabel("log(Stellar Mass / M☉)")

plt.title("Mass–Redshift Distribution")

plt.show()

plt.figure(figsize=(7,6))

plt.hexbin(z, mass, gridsize=60, cmap="plasma", mincnt=1)

plt.colorbar(label="Galaxy count")

plt.xlabel("Redshift")
plt.ylabel("log(Stellar Mass / M☉)")

plt.title("Mass vs Redshift")

plt.show()


z_grid = np.linspace(0,3,15)
mass_grid = np.linspace(9,12.5,15)

delta_map = np.full((len(z_grid)-1, len(mass_grid)-1), np.nan)

for i in range(len(z_grid)-1):

    z_mask = (z >= z_grid[i]) & (z < z_grid[i+1])

    for j in range(len(mass_grid)-1):

        mass_mask = (mass >= mass_grid[j]) & (mass < mass_grid[j+1])

        mask = z_mask & mass_mask

        agn_vals = log_overdensity[mask & agn]
        nonagn_vals = log_overdensity[mask & nonagn]

        if len(agn_vals) < 5 or len(nonagn_vals) < 5:
            continue

        delta_map[i,j] = (
            np.nanmedian(agn_vals) -
            np.nanmedian(nonagn_vals)
        )

plt.figure(figsize=(8,6))

plt.imshow(
    delta_map,
    origin='lower',
    aspect='auto',
    extent=[mass_grid[0], mass_grid[-1], z_grid[0], z_grid[-1]],
    cmap='coolwarm',
    vmin=-0.3,
    vmax=0.3
)

plt.colorbar(label="AGN − NonAGN overdensity")

plt.xlabel("log(Stellar Mass)")
plt.ylabel("Redshift")

plt.title("AGN Environmental Difference Map")

plt.show()

#-----------------------------------MASS DISTRIBUTIONS FOR MLAGN AND HLAGN---------------------------------#
bins_massdist = np.linspace(9, 12.5, 40)

plt.figure(figsize=(7, 5))

plt.hist(mass[hlagn], bins=bins_massdist, density=True, histtype='step',
         linewidth=2, color='purple', label='HLAGN')

plt.hist(mass[mlagn], bins=bins_massdist, density=True, histtype='step',
         linewidth=2, color='blue', label='MLAGN')

plt.hist(mass[nonagn], bins=bins_massdist, density=True, histtype='step',
         linewidth=2, color='orange', label='NonAGN')

plt.xlabel(r'$\log(M_{\star}/M_{\odot})$')
plt.ylabel('Normalised Count')
plt.title("Stellar mass distribution of galaxy populations")

plt.legend()
plt.tight_layout()
plt.show()