"""
extract_station_beta.py
=======================
per-station extraction of the seasonal bias coefficients (beta) and
of the covariates used to spatialize them.

For each station, the physical terms of CRISP are subtracted from the daily
observations, leaving the residual attributable to the systematic bias. An
ordinary-least-squares cosine fit of that residual over the year gives the
three coefficients

    residual(t) = beta_amp * cos(2*pi*(t - t_star)) + beta_bias

 The spatial covariates of the bias are
extracted at each station from the ERA5 surface geopotential and from
terrain rasters.

All user configuration is in the block below.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
from pyproj import Transformer

# ============================================================
# 1. Path configuration
# ============================================================

TRAIN_CSV     = r'PATH\TO\stations_daily_train.csv'
VAL_CSV       = r'PATH\TO\stations_daily_val.csv'

ERA5_OROG     = r'PATH\TO\ERA5_static_geopotential.nc'
NORTHNESS_TIF = r'PATH\TO\northness.tif'
SLOPE_TIF     = r'PATH\TO\slope.tif'
SVF_TIF       = r'PATH\TO\svf.tif'
OUT_CSV       = r'PATH\TO\stations_Beta_and_Covariates.csv'

OUT_DIR = os.path.dirname(OUT_CSV)
if OUT_DIR:
    os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 2. Frozen model parameters (from the calibration step)
# ============================================================

FROZEN = {
    'alpha_slope':     0.479304,   # alpha_s
    'alpha_intercept': 0.074777,   # alpha_i
    'lambda':          0.780767,   # lambda
    'gamma_local':     490,        # gamma_L [m]

    'H_col':           'H_80',     # hypsometric-position column
    'R_local_col':     'R_30',     # elevation-range column
    'Vnorm_col':       'Vnorm',    # normalized valley bottom flatness column
}

# ============================================================
# 3. Quality-control thresholds
# ============================================================

MIN_OBS_DAYS   = 300
MIN_MONTH_SPAN = 8

# ============================================================
# 4. Raster extraction
# ============================================================
def extract_era5_grid_elev(lats, lons, era5_orog_path):
    """ERA5 surface elevation at each station (nearest ERA5 cell)."""
    ds = xr.open_dataset(era5_orog_path)
    for vn in ['z', 'geopotential', 'orography']:
        if vn in ds.data_vars:
            data = ds[vn]
            break
    else:
        raise KeyError(f"geopotential variable not found: {list(ds.data_vars)}")
    for tdim in ['valid_time', 'time']:
        if tdim in data.dims:
            data = data.isel({tdim: 0})
            break
    results = []
    for lat, lon in zip(lats, lons):
        val = float(data.sel(latitude=lat, longitude=lon, method='nearest').values)
        results.append(val / 9.80665)     # geopotential -> geopotential height
    ds.close()
    return np.array(results)


def extract_from_tif(lats, lons, tif_path, label='raster'):
    """
    Sample a GeoTIFF at station coordinates. If the raster is not in WGS84,
    the station coordinates are transformed into the raster CRS first.
    """
    if not os.path.exists(tif_path):
        print(f"  [warning] {label} TIF not found: {tif_path}")
        return np.full(len(lats), np.nan)

    with rasterio.open(tif_path) as src:
        band = src.read(1)
        raster_crs = src.crs
        if raster_crs is not None and raster_crs.to_epsg() != 4326:
            transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
            xs, ys = transformer.transform(lons, lats)
        else:
            xs, ys = lons, lats

        results = []
        for x, y in zip(xs, ys):
            try:
                row, col = src.index(x, y)
                val = float(band[row, col])
                if src.nodata is not None and val == src.nodata:
                    val = np.nan
                results.append(val)
            except (IndexError, ValueError):
                results.append(np.nan)

    valid = int(np.sum(~np.isnan(results)))
    print(f"  {label}: {valid}/{len(lats)} stations sampled")
    return np.array(results)

# ============================================================
# 5. Physical model without the bias term
# ============================================================
def compute_residual(df, frozen):
    """
    Residual left after subtracting the physical terms of CRISP from the
    observations:
        residual = T_obs - T_fpl - alpha(h)*DT_new - lambda*Vnorm*DT_new
    This residual is what the seasonal bias term must describe.
    """
    H  = df[frozen['H_col']].values
    Rl = df[frozen['R_local_col']].values
    Vn = df[frozen['Vnorm_col']].values.clip(0, 1)

    S_local = np.exp(-Rl / frozen['gamma_local'])
    h       = H * (1.0 - S_local) + S_local
    alpha   = frozen['alpha_intercept'] + (np.exp(frozen['alpha_slope'] * h) - 1.0)

    T_model_no_beta = (df['T_fpl'].values
                       + alpha * df['DT_new'].values
                       + frozen['lambda'] * Vn * df['DT_new'].values)
    return df['T_obs'].values - T_model_no_beta

# ============================================================
# 6. Per-station OLS cosine fit
# ============================================================
def fit_station_beta(t, residual):
    """
    Fit residual(t) = A*cos(2*pi*t) + B*sin(2*pi*t) + bias by OLS, and convert
    to amplitude / phase:
        beta_amp = sqrt(A^2 + B^2)
        t_star   = atan2(B, A) / (2*pi)  (mod 1)
    Standard errors of beta_bias and beta_amp are propagated from the OLS
    covariance.
    """
    n = len(t)
    X = np.column_stack([np.cos(2 * np.pi * t), np.sin(2 * np.pi * t), np.ones(n)])
    W, _, _, _ = np.linalg.lstsq(X, residual, rcond=None)
    A, B, bias = W[0], W[1], W[2]

    y_pred = X @ W
    ssr    = float(np.sum((residual - y_pred) ** 2))
    rmse   = np.sqrt(ssr / n)
    dof    = n - 3
    sigma2 = ssr / dof if dof > 0 else np.nan

    try:
        inv_XTX  = np.linalg.inv(X.T @ X)
        cov      = sigma2 * inv_XTX
        bias_err = np.sqrt(cov[2, 2])
        beta_amp = np.sqrt(A ** 2 + B ** 2)
        if beta_amp > 1e-8:
            grad    = np.array([A / beta_amp, B / beta_amp])
            amp_err = np.sqrt(grad @ cov[0:2, 0:2] @ grad)
        else:
            amp_err = 0.0
    except np.linalg.LinAlgError:
        bias_err, amp_err = np.nan, np.nan

    beta_amp = np.sqrt(A ** 2 + B ** 2)
    t_star   = (np.arctan2(B, A) / (2 * np.pi)) % 1.0

    return {'A': A, 'B': B,
            'beta_bias': bias, 'bias_err': bias_err,
            'beta_amp': beta_amp, 'amp_err': amp_err,
            't_star': t_star, 'station_RMSE': rmse}

# ============================================================
# 7. Main
# ============================================================
def main():
    print("=" * 65)
    print("Step 3a: per-station beta extraction")
    print("=" * 65)

    print("\nReading daily station data ...")
    df_tr = pd.read_csv(TRAIN_CSV, encoding='utf-8-sig')
    df_va = pd.read_csv(VAL_CSV,   encoding='utf-8-sig')
    df_tr['_split'] = 'train'
    df_va['_split'] = 'val'
    df_all = pd.concat([df_tr, df_va], ignore_index=True)
    df_all['date']   = pd.to_datetime(df_all['date'])
    df_all['_month'] = df_all['date'].dt.month
    print(f"  gamma_L = {FROZEN['gamma_local']:.1f} m")

    # Station metadata (one row per station)
    print("\nExtracting station metadata ...")
    station_info = df_all.groupby('name').first().reset_index()

    elev_col = ('elevation' if 'elevation' in df_all.columns
                else ('ele' if 'ele' in df_all.columns else None))
    if elev_col:
        station_info['elevation'] = station_info[elev_col]
    else:
        raise ValueError("no elevation column (ele/elevation) in the input data")

    lon_col = ('longitude' if 'longitude' in df_all.columns
               else ('lon' if 'lon' in df_all.columns else None))
    lat_col = ('latitude' if 'latitude' in df_all.columns
               else ('lat' if 'lat' in df_all.columns else None))
    if not (lon_col and lat_col):
        raise ValueError("no longitude/latitude columns in the input data")
    station_info['longitude'] = station_info[lon_col]
    station_info['latitude']  = station_info[lat_col]

    n_stations = len(station_info)
    print(f"  {n_stations} stations")

    lats  = station_info['latitude'].values
    lons  = station_info['longitude'].values
    elevs = station_info['elevation'].values

    # Spatial covariates
    print("\nExtracting spatial covariates ...")
    era5_elev = extract_era5_grid_elev(lats, lons, ERA5_OROG)
    dZ        = elevs - era5_elev
    northness = extract_from_tif(lats, lons, NORTHNESS_TIF, 'northness')
    slope     = extract_from_tif(lats, lons, SLOPE_TIF,     'slope')
    svf       = extract_from_tif(lats, lons, SVF_TIF,       'svf')

    H_vals  = station_info[FROZEN['H_col']].values
    Rl_vals = station_info[FROZEN['R_local_col']].values
    Vn_vals = (station_info[FROZEN['Vnorm_col']].values
               if FROZEN['Vnorm_col'] in station_info.columns
               else np.full(n_stations, np.nan))

    # Residual (physical model minus observations, without the bias term)
    print("\nComputing the residual (observations minus physical terms) ...")
    df_all['residual'] = compute_residual(df_all, FROZEN)

    # Per-station cosine fit
    print("Fitting beta per station (OLS cosine) ...")
    name_to_idx = {name: i for i, name in enumerate(station_info['name'].values)}
    split_map   = dict(zip(station_info['name'], station_info['_split']))

    results = []
    n_success, n_dropped_qc = 0, 0
    for name, group in df_all.groupby('name'):
        if name not in name_to_idx:
            continue
        idx        = name_to_idx[name]
        n_obs      = len(group)
        month_span = group['_month'].nunique()

        if n_obs < MIN_OBS_DAYS or month_span < MIN_MONTH_SPAN:
            n_dropped_qc += 1
            continue

        fit = fit_station_beta(group['time_frac'].values, group['residual'].values)
        results.append({
            'name': name, 'split': split_map.get(name, 'unknown'),
            'longitude': lons[idx], 'latitude': lats[idx], 'elevation': elevs[idx],
            'elevation_era5': era5_elev[idx], 'dZ': dZ[idx],
            FROZEN['H_col']: H_vals[idx], FROZEN['R_local_col']: Rl_vals[idx],
            'Vnorm': Vn_vals[idx], 'northness': northness[idx],
            'slope': slope[idx], 'svf': svf[idx],
            'n_obs_days': n_obs, 'month_span': month_span, **fit,
        })
        n_success += 1

    print(f"  fitted: {n_success} stations "
          f"(dropped by QC: {n_dropped_qc})")

    df_out = pd.DataFrame(results)
    df_out.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\nDone -> {OUT_CSV}")

if __name__ == "__main__":
    main()