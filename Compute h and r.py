"""
Multi-scale terrain index generator (GeoTIFF)
============================================
Computes, for one or more analysis neighbourhoods:

  H (hypsometric position) : fraction of cells within the neighbourhood that
                             are higher than the centre cell; 1 at valley
                             bottoms, 0 at summits
  R (elevation range)      : maximum minus minimum elevation within the
                             neighbourhood, in metres

Both are written as GeoTIFFs carrying the coordinate reference system and
affine transform of the input DEM.

The input DEM must be in a PROJECTED coordinate system with cell sizes in
metres, so that a neighbourhood radius given in kilometres corresponds to a
well-defined number of cells.
"""

import os
import numpy as np
import rasterio
from scipy.ndimage import minimum_filter, maximum_filter, gaussian_filter, uniform_filter
from rasterio.fill import fillnodata

# ============================================================
# 0. User configuration
# ============================================================
# Path to the projected DEM (metres), and the output folder.
DEM_PATH = r'PATH\TO\DEM_projected.tif'
OUT_DIR  = r'PATH\TO\OUTPUT_FOLDER'

# Prefix of the output file names. Set this to the region being processed,
# so that outputs from different regions cannot be confused.
PREFIX = 'Qilian'

# Analysis neighbourhood radii, in km.
# The values used in the paper are H = 80 km for both regions, and
# R = 30 km (Qilian Mountains) or R = 5 km (Alps).
# Additional radii can be listed here to explore other scales.
H_RADII_KM = [80]
R_RADII_KM = [30]

# Gaussian smoothing applied before the min/max filters used for R, in cells.
# Suppresses single-cell artefacts; set to 0 to disable.
SMOOTH_SIGMA_PX = 2.12

# Number of elevation thresholds used to evaluate H.
N_BINS = 200

# Maximum search distance for NoData extrapolation, in cells. Should exceed
# the largest analysis window, otherwise windows near the edge of the domain
# will still reach unfilled cells.
FILL_DISTANCE = 2000

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

# ============================================================
# 1. Read the DEM and initialise
# ============================================================
print("=" * 60)
print("Reading DEM and spatial reference ...")
with rasterio.open(DEM_PATH) as src:
    ele = src.read(1).astype(np.float32)
    profile = src.profile          # CRS, affine transform, etc.
    nodata = src.nodata

    # Cell size in metres, taken directly from the affine transform
    pixel_m = abs(src.transform[0])

print(f"  DEM shape   : {ele.shape}  (ny x nx)")
print(f"  cell size   : {pixel_m:.1f} m")
print(f"  CRS         : {profile.get('crs')}")

# NoData and NaN handling
nan_mask = np.isnan(ele)
if nodata is not None:
    nan_mask = nan_mask | (ele == nodata)

# Extrapolate across NoData rather than filling with a constant: a constant
# fill would create an artificial cliff at the edge of the domain, which the
# moving windows would then see as relief.
print("  extrapolating across NoData (removing the edge cliff) ...")
valid_mask = (~nan_mask).astype('uint8')   # 1 = valid, 0 = NoData

ele_filled = fillnodata(
    ele.copy(),
    mask=valid_mask,
    max_search_distance=FILL_DISTANCE,
    smoothing_iterations=0
)

# Output metadata
profile.update(
    dtype=rasterio.float32,
    nodata=-9999.0,
    compress='lzw'      # LZW compression keeps the set of output TIFs small
)


def save_tif(out_array, filename, meta):
    """Write a GeoTIFF, restoring the original NoData mask."""
    out_array[nan_mask] = -9999.0
    with rasterio.open(filename, 'w', **meta) as dst:
        dst.write(out_array.astype(np.float32), 1)
    print(f"  [written] -> {os.path.basename(filename)}")


# ============================================================
# 2. Elevation range R
# ============================================================
if R_RADII_KM:
    print("\n" + "=" * 60)
    print("Elevation range R")

    if SMOOTH_SIGMA_PX > 0:
        ele_smooth = gaussian_filter(ele_filled, sigma=SMOOTH_SIGMA_PX)
    else:
        ele_smooth = ele_filled

    for radius_km in R_RADII_KM:
        print(f"\n>> R at {radius_km} km ...")
        r_win = int(round(radius_km * 1000.0 / pixel_m))
        r_win = r_win + 1 if r_win % 2 == 0 else r_win   # force an odd window
        print(f"   moving window: {r_win} x {r_win} cells")

        min_ele = minimum_filter(ele_smooth, size=r_win)
        max_ele = maximum_filter(ele_smooth, size=r_win)
        R = (max_ele - min_ele).astype(np.float32)

        out_name = os.path.join(OUT_DIR, f'{PREFIX}_R_{radius_km}km.tif')
        save_tif(R, out_name, profile)

# ============================================================
# 3. Hypsometric position H
# ============================================================
def compute_H_fast(dem_array, window_size, n_bins=200):
    """
    Hypsometric position by multi-threshold binarisation.

    For each of n_bins elevation thresholds spanning the DEM range, a uniform
    (box) filter gives the local fraction of cells above that threshold. This
    yields an exceedance-probability profile at every cell, from which H is
    read off at the elevation of the cell itself by linear interpolation.

    The intermediate probability table has shape (n_bins, ny, nx) and is held
    as float16 to limit peak memory use.
    """
    z_min, z_max = np.nanmin(dem_array), np.nanmax(dem_array)
    thresholds = np.linspace(z_min, z_max, n_bins)
    ny, nx = dem_array.shape

    P_table = np.empty((n_bins, ny, nx), dtype=np.float16)

    for k, t in enumerate(thresholds):
        binary = (dem_array > t).astype(np.float32)
        P_table[k] = uniform_filter(binary, size=window_size,
                                    mode='nearest').astype(np.float16)
        if (k + 1) % 50 == 0:
            print(f"   exceedance table: {k+1}/{n_bins}")

    P_table_t = P_table.transpose(1, 2, 0).astype(np.float32)
    z_flat = dem_array.ravel()
    P_flat = P_table_t.reshape(-1, n_bins)

    H_flat = np.array([
        np.interp(z_flat[i], thresholds, P_flat[i])
        for i in range(len(z_flat))
    ], dtype=np.float32)

    return H_flat.reshape(ny, nx)


if H_RADII_KM:
    print("\n" + "=" * 60)
    print("Hypsometric position H")

    for radius_km in H_RADII_KM:
        print(f"\n>> H at {radius_km} km ...")
        h_win = int(round(radius_km * 1000.0 / pixel_m))
        h_win = h_win + 1 if h_win % 2 == 0 else h_win
        print(f"   moving window: {h_win} x {h_win} cells")

        H = compute_H_fast(ele_filled, h_win, n_bins=N_BINS)

        out_name = os.path.join(OUT_DIR, f'{PREFIX}_H_{radius_km}km.tif')
        save_tif(H, out_name, profile)

print("\n" + "=" * 60)
print(f"Done. Output directory: {OUT_DIR}")