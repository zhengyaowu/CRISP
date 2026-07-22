"""
spatialize_bias.py
==================
Step 3b: spatialization of the seasonal bias coefficients by regression
kriging.

Each of the three coefficients extracted per station in step 3a
(beta_bias, beta_amp, t_star) is generalized to a continuous field, and
independently, by the same two-step method:

  1. a random forest gives the trend supported by the covariates
  2. ordinary kriging of the random-forest residuals recovers the
     spatially autocorrelated structure the trend does not explain

Their sum is the regression-kriging estimate

  beta_hat(x0) = f_RF(P(x0)) + OK[r](x0)

where P is the covariate vector and r are the residuals at the stations.
Because ordinary kriging is an exact interpolator, the sum reproduces the
fitted station values exactly at the stations.

Covariates:
  longitude, latitude, elevation, dZ, northness, slope, svf
The terrain predictors carried by the physical terms of CRISP (H, R, Vnorm)
are deliberately excluded, so that the same terrain-temperature relationship
is not expressed twice.

Outputs (in OUT_DIR), selected with OUTPUT_MODE:

  OUTPUT_MODE = 'points' or 'both'
    val_predictions.csv      validation points, with the predicted beta
                             columns beta_bias_pred, beta_amp_pred,
                             t_star_pred appended
    train_predictions.csv    training points, same columns

  OUTPUT_MODE = 'grid' or 'both'
    beta_bias_map.tif, beta_amp_map.tif, t_star_map.tif

  always
    model_info.csv           hyperparameters and skill scores

These feed run_downscaling.py: the prediction CSV is read by the point
product (the beta of each point is taken directly from it), and the three
GeoTIFFs are read by the gridded product. Points deliberately do not use the
rasters, since sampling a rasterised field back at a point would interpolate
twice and carry the rasterisation error.

All user configuration is in the block below.
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from pykrige.ok import OrdinaryKriging

# ============================================================
# 1. Path configuration
# ============================================================
# Input: the per-station beta + covariate CSV from step 3a.
BETA_CSV = r'PATH\TO\stations_Beta_and_Covariates.csv'
OUT_DIR  = r'PATH\TO\OUTPUT_FOLDER'

REGION_NAME = 'Qilian Mountains'   # written into the NetCDF metadata

# What to output:
#   'points' : only the per-point beta predictions (CSV). This is what the
#              point product of run_downscaling.py reads.
#   'grid'   : only the gridded beta fields (three GeoTIFFs). This is what the
#              gridded product of run_downscaling.py reads.
#   'both'   : both of the above.
# The station-level skill scores are always printed and written to
# model_info.csv, whichever mode is chosen.
OUTPUT_MODE = 'both'

if OUTPUT_MODE not in ('points', 'grid', 'both'):
    raise ValueError("OUTPUT_MODE must be 'points', 'grid' or 'both'")

# Fine-scale DEM (WGS84 NetCDF), defining the target grid and providing the
# longitude/latitude/elevation covariates. Required for 'grid' and 'both'.
DEM_FINE_NC = r'PATH\TO\DEM.nc'

# Covariate rasters (WGS84 GeoTIFF). Required for 'grid' and 'both'.
# longitude / latitude come from DEM_FINE_NC; elevation is read from it too.
COV_DIR = r'PATH\TO\COVARIATE_FOLDER'
GLOBAL_RASTER_FILES = {
    'slope':     os.path.join(COV_DIR, 'slope.tif'),
    'northness': os.path.join(COV_DIR, 'northness.tif'),
    'dZ':        os.path.join(COV_DIR, 'dZ.tif'),
    'svf':       os.path.join(COV_DIR, 'svf.tif'),
}

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 2. Covariates
# ============================================================
BIAS_COVARIATES = [
    'longitude', 'latitude', 'elevation',
    'dZ', 'northness', 'slope', 'svf',
]

COMPONENTS = ('beta_bias', 'beta_amp', 't_star')

# ============================================================
# 3. Random-forest hyperparameters
# ============================================================
# n_estimators and max_features are fixed; the leaf size and the depth, which
# control overfitting on a small station network, are selected by
# cross-validation. Use larger leaf sizes and shallower trees for sparser
# networks.
RF_PARAM_GRID = {
    'min_samples_leaf': [10, 15, 20, 25],
    'max_depth':        [4, 6, 8, None],
}
RF_FIXED = dict(n_estimators=300, max_features='sqrt', random_state=42, n_jobs=1)

N_CV = 5


# ============================================================
# 4. Data loading
# ============================================================
def load_and_split(csv_path):
    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    missing = [c for c in BIAS_COVARIATES if c not in df.columns]
    if missing:
        raise ValueError(f"covariate columns missing from the CSV: {missing}")

    # Fill the few missing covariate values with the training-set median, so
    # that no information from the validation set enters the imputation
    train_median = df[df['split'] == 'train'][BIAS_COVARIATES].median()
    df[BIAS_COVARIATES] = df[BIAS_COVARIATES].fillna(train_median)

    df_train = df[df['split'] == 'train'].copy().reset_index(drop=True)
    df_val   = df[df['split'] == 'val'  ].copy().reset_index(drop=True)
    print(f"  training stations  : {len(df_train)}")
    print(f"  validation stations: {len(df_val)}")
    if len(df_train) == 0 or len(df_val) == 0:
        raise ValueError("the 'split' column must contain both "
                         "'train' and 'val' rows")
    return df_train, df_val


# ============================================================
# 5. Random forest
# ============================================================
def tune_rf(X_train, y_train, target_name):
    """Select the RF hyperparameters by k-fold CV within the training set."""
    print(f"  hyperparameter search ({target_name}, {N_CV}-fold CV, "
          f"{len(X_train)} stations) ...")
    base = RandomForestRegressor(**RF_FIXED)
    gs = GridSearchCV(base, RF_PARAM_GRID, cv=N_CV,
                      scoring='neg_root_mean_squared_error',
                      n_jobs=-1, verbose=0, refit=True)
    gs.fit(X_train, y_train)
    best = gs.best_params_
    cv_rmse = float(-gs.best_score_)
    print(f"  best: min_samples_leaf={best['min_samples_leaf']}  "
          f"max_depth={best['max_depth']}  CV_RMSE={cv_rmse:.4f}")
    return gs.best_estimator_, best, cv_rmse


def fit_rf(df_train, df_val, target, tune=True):
    """Fit the random-forest trend for one component and evaluate it."""
    print("\n" + "-" * 55)
    print(f"{target}: random-forest trend")
    print("-" * 55)

    X_train = df_train[BIAS_COVARIATES].values
    y_train = df_train[target].values
    X_val   = df_val[BIAS_COVARIATES].values
    y_val   = df_val[target].values

    if tune:
        rf, best_params, cv_rmse = tune_rf(X_train, y_train, target)
    else:
        best_params = dict(min_samples_leaf=10, max_depth=12)
        rf = RandomForestRegressor(**best_params,
                                   n_estimators=RF_FIXED['n_estimators'],
                                   max_features=RF_FIXED['max_features'],
                                   random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        cv_rmse = np.nan

    pred_train = rf.predict(X_train)
    pred_val   = rf.predict(X_val)

    rmse_train = float(np.sqrt(mean_squared_error(y_train, pred_train)))
    rmse_val   = float(np.sqrt(mean_squared_error(y_val, pred_val)))
    r2_train   = float(r2_score(y_train, pred_train))
    r2_val     = float(r2_score(y_val, pred_val))
    bias_val   = float(np.mean(pred_val - y_val))

    print(f"  {'':16s} {'RMSE':>9s} {'R2':>9s} {'BIAS':>9s}")
    print(f"  {'RF training':16s} {rmse_train:>9.4f} {r2_train:>9.4f}")
    print(f"  {'RF validation':16s} {rmse_val:>9.4f} {r2_val:>9.4f} {bias_val:>+9.4f}")

    importances = rf.feature_importances_
    print("  feature importance (MDI):")
    for name, imp in sorted(zip(BIAS_COVARIATES, importances), key=lambda x: -x[1]):
        print(f"    {name:20s}: {imp:.4f}")

    return {
        'rf': rf, 'best_params': best_params, 'cv_rmse': cv_rmse,
        'pred_train': pred_train, 'pred_val': pred_val,
        'y_train': y_train, 'y_val': y_val,
        'residual_train': y_train - pred_train,
        'residual_val':   y_val   - pred_val,
        'feature_importances': dict(zip(BIAS_COVARIATES, importances)),
        'rf_val_RMSE': rmse_val, 'rf_val_R2': r2_val,
    }


# ============================================================
# 6. Ordinary kriging of the RF residuals
# ============================================================
def krige_residuals(df_train, df_val, residual_train, variogram_model):
    lon_tr = df_train['longitude'].values
    lat_tr = df_train['latitude'].values
    lon_va = df_val['longitude'].values
    lat_va = df_val['latitude'].values

    ok = OrdinaryKriging(lon_tr, lat_tr, residual_train,
                         variogram_model=variogram_model,
                         verbose=False, enable_plotting=False, nlags=20)
    z_tr, _ = ok.execute('points', lon_tr, lat_tr)
    z_va, _ = ok.execute('points', lon_va, lat_va)
    params = ok.variogram_model_parameters
    nugget_ratio = float(params[2] / (params[0] + 1e-8))
    return {
        'ok': ok, 'variogram_model': variogram_model,
        'resid_pred_train': np.array(z_tr).flatten(),
        'resid_pred_val':   np.array(z_va).flatten(),
        'nugget_ratio': nugget_ratio,
    }


def select_variogram(df_train, df_val, residual_train, residual_val_true,
                     target):
    """Choose the variogram model minimising the validation residual RMSE."""
    print(f"  variogram model selection ({target}):")
    best_model, best_rmse, best_res = None, np.inf, None
    for model in ('spherical', 'exponential', 'gaussian'):
        try:
            res = krige_residuals(df_train, df_val, residual_train, model)
            rmse = float(np.sqrt(np.mean(
                (residual_val_true - res['resid_pred_val']) ** 2)))
            print(f"    {model:12s} validation residual RMSE = {rmse:.4f}")
            if rmse < best_rmse:
                best_model, best_rmse, best_res = model, rmse, res
        except Exception as exc:
            print(f"    {model:12s} failed: {exc}")
    if best_res is None:
        best_res = krige_residuals(df_train, df_val, residual_train, 'spherical')
        best_model = 'spherical'
    print(f"    selected: {best_model}")
    return best_model, best_res


# ============================================================
# 7. Gridded fields
# ============================================================
def generate_global_maps(rf_models, residuals, variograms, df_full):
    """
    Predict the three components over the whole domain and write them as
    GeoTIFF.
    """
    import netCDF4 as nc_lib
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject, Resampling
    from scipy.ndimage import distance_transform_edt

    print("\n" + "=" * 65)
    print("Gridded fields by regression kriging")
    print("=" * 65)

    missing = [p for p in GLOBAL_RASTER_FILES.values() if not os.path.exists(p)]
    if missing:
        print("  skipped, missing covariate rasters:")
        for p in missing:
            print(f"    {p}")
        return
    if not os.path.exists(DEM_FINE_NC):
        print(f"  skipped, DEM not found: {DEM_FINE_NC}")
        return

    def _fill_edges(arr):
        nan_mask = ~np.isfinite(arr)
        if not nan_mask.any():
            return arr
        _, idx = distance_transform_edt(nan_mask, return_indices=True)
        filled = arr.copy()
        filled[nan_mask] = arr[idx[0][nan_mask], idx[1][nan_mask]]
        return filled

    with nc_lib.Dataset(DEM_FINE_NC, 'r') as ds:
        dem_lat_1d = (ds['latitude'][:].data if 'latitude' in ds.variables
                      else ds['lat'][:].data).astype(np.float32)
        dem_lon_1d = (ds['longitude'][:].data if 'longitude' in ds.variables
                      else ds['lon'][:].data).astype(np.float32)
    if dem_lon_1d[0] > dem_lon_1d[-1]:
        dem_lon_1d = dem_lon_1d[::-1]
    rows, cols = len(dem_lat_1d), len(dem_lon_1d)
    lat_n, lat_s = float(dem_lat_1d.max()), float(dem_lat_1d.min())
    print(f"  target grid: {rows} x {cols}")

    WGS84 = CRS.from_epsg(4326)
    dlon = float(abs(dem_lon_1d[-1] - dem_lon_1d[0])) / max(cols - 1, 1)
    dlat = float(abs(dem_lat_1d[-1] - dem_lat_1d[0])) / max(rows - 1, 1)
    dst_tf = from_bounds(float(dem_lon_1d.min()) - dlon/2, lat_s - dlat/2,
                         float(dem_lon_1d.max()) + dlon/2, lat_n + dlat/2,
                         cols, rows)

    print("  reading covariate rasters ...")
    cov = {}
    for name, path in GLOBAL_RASTER_FILES.items():
        with rasterio.open(path) as src:
            nd = src.nodata if src.nodata is not None else -9999.0
            data = np.empty((rows, cols), dtype=np.float32)
            reproject(source=rasterio.band(src, 1), destination=data,
                      src_transform=src.transform, src_crs=src.crs,
                      src_nodata=nd, dst_transform=dst_tf, dst_crs=WGS84,
                      dst_nodata=np.nan, resampling=Resampling.bilinear)
        cov[name] = _fill_edges(data)
        print(f"    {name}: [{np.nanmin(cov[name]):.2f}, {np.nanmax(cov[name]):.2f}]")

    cov['longitude'] = np.tile(dem_lon_1d, (rows, 1)).astype(np.float32)
    cov['latitude']  = np.tile(dem_lat_1d[:, None], (1, cols)).astype(np.float32)
    with nc_lib.Dataset(DEM_FINE_NC, 'r') as ds:
        _ev = next((k for k in ('elevation', 'dem', 'DEM', 'z', 'height', 'hgt')
                    if k in ds.variables),
                   next(k for k in ds.variables
                        if k not in {'latitude', 'longitude', 'lat', 'lon'}))
        elev = ds[_ev][:].squeeze().astype(np.float32)
    elev[~np.isfinite(elev)] = np.nan
    cov['elevation'] = elev

    mask = np.ones((rows, cols), dtype=bool)
    for arr in cov.values():
        mask &= ~np.isnan(arr)
    print(f"  valid cells: {mask.sum():,} ({mask.mean()*100:.1f}%)")

    pix = np.where(mask.ravel())[0]
    n_pix = len(pix)
    X_glob = np.zeros((n_pix, len(BIAS_COVARIATES)), dtype=np.float32)
    for j, c in enumerate(BIAS_COVARIATES):
        X_glob[:, j] = cov[c].ravel()[pix]
    lon_pts = cov['longitude'].ravel()[pix]
    lat_pts = cov['latitude'].ravel()[pix]

    geo_profile = {'driver': 'GTiff', 'dtype': 'float32', 'count': 1,
                   'height': rows, 'width': cols, 'crs': WGS84,
                   'transform': dst_tf, 'nodata': -9999.0,
                   'compress': 'lzw', 'predictor': 3}

    def _save_tif(arr, name, suffix='map'):
        path = os.path.join(OUT_DIR, f'{name}_{suffix}.tif')
        out = arr.copy(); out[np.isnan(out)] = -9999.0
        with rasterio.open(path, 'w', **geo_profile) as dst:
            dst.write(out, 1)
        print(f"    -> {os.path.basename(path)}")

    def _rk_map(rf, resid_anchor, name, variogram, clip_min=None, clip_max=None):
        print(f"\n  {name}:")
        # Random-forest trend, predicted in chunks to limit peak memory
        chunk = 200_000
        trend = np.empty(n_pix, dtype=np.float32)
        for s in range(0, n_pix, chunk):
            e = min(s + chunk, n_pix)
            trend[s:e] = rf.predict(X_glob[s:e]).astype(np.float32)
            print(f"    RF prediction: {e:,}/{n_pix:,}", end='\r')
        print(f"    RF prediction complete: {n_pix:,} cells      ")

        trend_map = np.full(rows * cols, np.nan, dtype=np.float32)
        trend_map[pix] = trend
        trend_map = trend_map.reshape(rows, cols)

        lons_a = df_full['longitude'].values
        lats_a = df_full['latitude'].values
        print(f"    kriging residuals from {len(lons_a)} stations ...")
        ok = OrdinaryKriging(lons_a, lats_a, resid_anchor,
                             variogram_model=variogram,
                             verbose=False, enable_plotting=False, nlags=20)
        resid = np.full(n_pix, np.nan, dtype=np.float32)
        ck = 50_000
        for s in range(0, n_pix, ck):
            e = min(s + ck, n_pix)
            z, _ = ok.execute('points', lon_pts[s:e], lat_pts[s:e])
            resid[s:e] = np.array(z).flatten()
            print(f"      kriging: {e:,}/{n_pix:,}", end='\r')
        print("      kriging complete                     ")

        resid_map = np.full(rows * cols, np.nan, dtype=np.float32)
        resid_map[pix] = resid
        resid_map = resid_map.reshape(rows, cols)

        rk = trend_map + resid_map
        if clip_min is not None or clip_max is not None:
            rk = np.clip(rk, clip_min, clip_max)
        print(f"    field: mean={np.nanmean(rk):+.4f}  sd={np.nanstd(rk):.4f}")
        _save_tif(rk, name, 'map')
        return rk

    _rk_map(rf_models['beta_bias'], residuals['beta_bias'],
            'beta_bias', variograms['beta_bias'])
    _rk_map(rf_models['beta_amp'], residuals['beta_amp'],
            'beta_amp', variograms['beta_amp'], clip_min=0.0)
    _rk_map(rf_models['t_star'], residuals['t_star'],
            't_star', variograms['t_star'], clip_min=0.0, clip_max=1.0)

    print("\n  gridded fields complete")


# ============================================================
# 8. Main
# ============================================================
def main():
    print("=" * 65)
    print("Step 3b: spatialization of the seasonal bias")
    print("=" * 65)
    print(f"  covariates: {BIAS_COVARIATES}")
    t0 = time.time()

    print("\nLoading station data ...")
    df_train, df_val = load_and_split(BETA_CSV)

    results = {}
    for comp in COMPONENTS:
        print("\n" + "=" * 65)
        print(comp)
        print("=" * 65)

        rf_res = fit_rf(df_train, df_val, comp, tune=True)
        best_vario, kres = select_variogram(
            df_train, df_val,
            rf_res['residual_train'], rf_res['residual_val'], comp)

        pred_train_rk = rf_res['pred_train'] + kres['resid_pred_train']
        pred_val_rk   = rf_res['pred_val']   + kres['resid_pred_val']

        rmse_rk_val = float(np.sqrt(np.mean((pred_val_rk - rf_res['y_val']) ** 2)))
        r2_rk_val = float(1 - np.sum((pred_val_rk - rf_res['y_val']) ** 2)
                          / np.sum((rf_res['y_val'] - rf_res['y_val'].mean()) ** 2))

        print(f"\n  {comp}: RK validation RMSE={rmse_rk_val:.4f}  R2={r2_rk_val:.4f}")
        if kres['nugget_ratio'] > 0.85:
            print("  note: nugget ratio > 85%; the residuals are close to "
                  "white noise, so kriging adds little (expected for a "
                  "component with weak spatial structure)")

        rf_res.update({
            'variogram': best_vario, 'nugget_ratio': kres['nugget_ratio'],
            'pred_train_rk': pred_train_rk, 'pred_val_rk': pred_val_rk,
            'rmse_rk_val': rmse_rk_val, 'r2_rk_val': r2_rk_val,
        })
        results[comp] = rf_res

    # Summary: global constant vs regression kriging
    print("\n" + "=" * 65)
    print("Summary: global constant vs regression kriging")
    print("=" * 65)
    print(f"  {'component':12s} {'baseline':>10s} {'RK':>10s} "
          f"{'change':>10s} {'RK R2':>9s}")
    for comp in COMPONENTS:
        res = results[comp]
        base = float(np.sqrt(np.mean(
            (df_val[comp] - df_train[comp].mean()) ** 2)))
        print(f"  {comp:12s} {base:>10.4f} {res['rmse_rk_val']:>10.4f} "
              f"{base - res['rmse_rk_val']:>+10.4f} {res['r2_rk_val']:>9.4f}")

    print("\nWriting results ...")

    # Per-point beta predictions. These are the point-product input to
    # run_downscaling.py: the beta of each point is predicted directly at its
    # coordinates here, and read from the CSV there.
    if OUTPUT_MODE in ('points', 'both'):
        df_val_out = df_val.copy()
        df_train_out = df_train.copy()
        for comp in COMPONENTS:
            df_val_out[f'{comp}_pred']    = results[comp]['pred_val_rk']
            df_val_out[f'{comp}_pred_rf'] = results[comp]['pred_val']
            df_train_out[f'{comp}_pred']    = results[comp]['pred_train_rk']
            df_train_out[f'{comp}_pred_rf'] = results[comp]['pred_train']
        df_val_out.to_csv(os.path.join(OUT_DIR, 'val_predictions.csv'),
                          index=False, encoding='utf-8-sig')
        df_train_out.to_csv(os.path.join(OUT_DIR, 'train_predictions.csv'),
                            index=False, encoding='utf-8-sig')
        print("  val_predictions.csv, train_predictions.csv")

    info = {'method': 'regression kriging (RF trend + OK residuals)',
            'covariates': ','.join(BIAS_COVARIATES),
            'n_train': len(df_train), 'n_val': len(df_val)}
    for comp in COMPONENTS:
        res = results[comp]
        info.update({
            f'{comp}_best_params': str(res['best_params']),
            f'{comp}_cv_RMSE': res['cv_rmse'],
            f'{comp}_rf_val_RMSE': res['rf_val_RMSE'],
            f'{comp}_rf_val_R2': res['rf_val_R2'],
            f'{comp}_rk_val_RMSE': res['rmse_rk_val'],
            f'{comp}_rk_val_R2': res['r2_rk_val'],
            f'{comp}_variogram': res['variogram'],
            f'{comp}_nugget_ratio': res['nugget_ratio'],
        })
    pd.DataFrame([info]).to_csv(os.path.join(OUT_DIR, 'model_info.csv'),
                                index=False, encoding='utf-8-sig')

    # Gridded fields (refit on all stations, using the selected
    # hyperparameters and variograms)
    if OUTPUT_MODE in ('grid', 'both'):
        df_full = pd.concat([df_train, df_val], ignore_index=True)
        X_full = df_full[BIAS_COVARIATES].values
        print("\n" + "=" * 65)
        print(f"Refitting on all {len(df_full)} stations for the gridded fields")
        print("=" * 65)
        rf_models, residuals, variograms = {}, {}, {}
        for comp in COMPONENTS:
            best = results[comp]['best_params']
            rf_full = RandomForestRegressor(
                min_samples_leaf=best['min_samples_leaf'],
                max_depth=best['max_depth'],
                n_estimators=RF_FIXED['n_estimators'],
                max_features=RF_FIXED['max_features'],
                random_state=42, n_jobs=-1)
            rf_full.fit(X_full, df_full[comp].values)
            rf_models[comp] = rf_full
            residuals[comp] = df_full[comp].values - rf_full.predict(X_full)
            variograms[comp] = results[comp]['variogram']
            print(f"  {comp:12s} refitted")
        generate_global_maps(rf_models, residuals, variograms, df_full)

    print(f"\nOutput directory: {OUT_DIR}")
    print(f"Total run time: {time.time()-t0:.1f} s")


if __name__ == '__main__':
    main()