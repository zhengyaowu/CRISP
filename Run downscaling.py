"""
run_downscaling.py
==================
CRISP downscaling: main program. Edit this file only; all the model code is
in crisp_core.py.

What you can configure below:
  1  DATA_ROOT      root data directory
  2  YEARS / MONTHS years and months to generate
  3  RUN_SPATIAL / RUN_STATION   gridded product and/or point time series
  4  BETA_MODE      'global' or 'spatial' (see below)
  5  station input  a short in-file list, or a CSV, for the point product
  6  file paths     normally only DATA_ROOT needs changing

Seasonal bias (BETA_MODE)
-------------------------
'global'
    One constant triple (beta_bias, beta_amp, t_star) is applied everywhere.
    Nothing extra is needed. 

'spatial'
    Spatially varying bias, from spatialize_bias.py. What is required depends
    on the product:
      - gridded product -> the three GeoTIFFs beta_bias_map.tif,
                           beta_amp_map.tif, t_star_map.tif
      - point product   -> a CSV carrying, for each point, the beta predicted
                           directly at its coordinates (columns
                           beta_bias_pred, beta_amp_pred, t_star_pred), i.e.
                           the spatialize_bias.py prediction output
      - both            -> both the rasters and the CSV
"""

import os
import time
import netCDF4 as nc

from Crisp_core import CrispDownscaler, load_stations_from_csv


def main():

    # ---------------------------------------------------------
    #  1  Root data directory
    # ---------------------------------------------------------
    DATA_ROOT = r'PATH\TO\DATA_ROOT'

    REGION_NAME = 'Qilian Mountains'   # written into the NetCDF metadata

    # ---------------------------------------------------------
    #  2  Years and months
    # ---------------------------------------------------------
    YEARS  = list(range(2018, 2023))   # 2018-2022
    # YEARS = [2019]                   # single year, for a quick test

    MONTHS = None                      # None = all twelve months
    # MONTHS = [6, 7, 8]               # summer only
    # MONTHS = [12, 1, 2]              # winter only

    # ---------------------------------------------------------
    #  3  Product switches
    # ---------------------------------------------------------
    RUN_SPATIAL = True     # gridded product   (T_downscaled_{YYYY}.nc)
    RUN_STATION = False    # point time series (station_T_{YYYY}.csv)

    # ---------------------------------------------------------
    #  4  Seasonal-bias mode  ('global' or 'spatial')
    # ---------------------------------------------------------
    BETA_MODE = 'spatial'

    # (a) global mode: one constant triple applied everywhere.
    #     The defaults below are the Qilian Mountains values from the paper;
    #     replace them for another region.
    BETA_GLOBAL = {
        'beta_bias': -0.442,   # annual-mean bias      [degC]
        'beta_amp':   0.764,   # seasonal amplitude    [degC]
        't_star':     0.468,   # phase of the peak     [year fraction]
    }

    # (b) spatial mode, gridded product: the three beta rasters from
    #     spatialize_bias.py. Paths are set in section 6 below
    #     (beta_bias_tif, beta_amp_tif, t_star_tif).
    #
    # (c) spatial mode, point product: the beta of each point comes from
    #     spatialize_bias.py, supplied in section 5 below, either as
    #     STATIONS_CSV (its prediction output, carrying beta_bias_pred /
    #     beta_amp_pred / t_star_pred) or as beta_bias / beta_amp / t_star
    #     keys in the in-file station list. There is no separate variable for
    #     it: the point list and its beta travel together.

    # ---------------------------------------------------------
    #  5  Station input (used when RUN_STATION = True)
    #
    #     Option A: write a few points directly in the list below.
    #               In 'spatial' mode each entry must also carry beta_bias,
    #               beta_amp and t_star (there is no other source of beta for
    #               a hand-written point).
    #     Option B: read the points from a CSV.
    #               Required columns: name, latitude, longitude, elevation
    #               (case-insensitive; lat/lon/lng/ele/elev/altitude accepted).
    #               In 'spatial' mode the CSV must ALSO carry beta_bias_pred,
    #               beta_amp_pred, t_star_pred, i.e. it is the prediction
    #               output of spatialize_bias.py for exactly these points.
    #
    #     Set STATIONS_CSV = None to use option A, or a path for option B.
    # ---------------------------------------------------------
    stations = [
        # {'name': 'Qilian',  'latitude': 38.18, 'longitude': 100.25, 'elevation': 2787},
        # {'name': 'Qilian',  'latitude': 38.18, 'longitude': 100.25, 'elevation': 2787,
        #  'beta_bias': -0.44, 'beta_amp': 0.76, 't_star': 0.47},
    ]

    STATIONS_CSV = None
    # STATIONS_CSV = r'PATH\TO\val_predictions.csv'   # spatialize_bias.py output

    # ---------------------------------------------------------
    #  6  File paths (normally only DATA_ROOT needs changing)
    # ---------------------------------------------------------
    static_dir = os.path.join(DATA_ROOT, 'Static')
    cfg = {
        # ERA5 directories and file-name templates
        'tsur_dir':  os.path.join(DATA_ROOT, 'Tsur'),
        'pl_t_dir':  os.path.join(DATA_ROOT, 'Pressure_levels'),
        'pl_z_dir':  os.path.join(DATA_ROOT, 'Geopotential'),
        'tsur_tmpl': 'ERA5_2m_temperature_daily_{year}.nc',
        'pl_t_tmpl': 'ERA5_pl_temperature_daily_{year}_300_1000hpa.nc',
        'pl_z_tmpl': 'ERA5_pl_geopotential_daily_{year}_300_1000hpa.nc',

        # Static inputs
        'dem_fine_nc':   os.path.join(static_dir, 'QiLian_DEM.nc'),
        'dem_coarse_nc': os.path.join(static_dir, 'ERA5_static_geopotential.nc'),
        'h_tif':         os.path.join(static_dir, 'QiLian_H_80km_GCS.tif'),
        'r_tif':         os.path.join(static_dir, 'QiLian_R_30km_GCS.tif'),
        'vnorm_tif':     os.path.join(static_dir, 'QiLian_Vnorm_GCS.tif'),

        # Spatialized beta rasters (spatial mode, gridded product only)
        'beta_bias_tif': os.path.join(static_dir, 'beta_bias_map.tif'),
        'beta_amp_tif':  os.path.join(static_dir, 'beta_amp_map.tif'),
        't_star_tif':    os.path.join(static_dir, 't_star_map.tif'),

        # Output
        'out_dir': os.path.join(DATA_ROOT, 'Downscaled'),

        # Calibrated model parameters (Qilian Mountains; replace for another
        # region, and recompute H/R with the matching neighbourhood radii)
        'frozen': {
            'alpha_s':  0.479304,
            'alpha_i':  0.074777,
            'lambda':   0.780767,
            'gamma_L':  490,
        },

        # Assembled from the settings above
        'region_name': REGION_NAME,
        'beta_mode':   BETA_MODE,
        'beta_global': BETA_GLOBAL,
    }

    # =========================================================
    #  Nothing below needs editing
    # =========================================================
    t0 = time.time()
    model = CrispDownscaler(cfg)

    if RUN_SPATIAL:
        model.spatialProduct(YEARS, months=MONTHS)

    if RUN_STATION:
        # In spatial mode each point carries its own beta, predicted at that
        # coordinate by spatialize_bias.py and read from the CSV. In global
        # mode the constants are applied and the CSV needs only coordinates.
        require_beta = (BETA_MODE == 'spatial')

        if STATIONS_CSV is not None:
            # DEM bounding box for a range check, without reading the whole
            # DEM array
            try:
                with nc.Dataset(cfg['dem_fine_nc'], 'r') as _ds:
                    _la = (_ds['latitude'][:].data if 'latitude' in _ds.variables
                           else _ds['lat'][:].data)
                    _lo = (_ds['longitude'][:].data if 'longitude' in _ds.variables
                           else _ds['lon'][:].data)
                _bounds = (float(_la.min()), float(_la.max()),
                           float(_lo.min()), float(_lo.max()))
            except Exception as _e:
                print(f"could not read the DEM extent, range check skipped: {_e}")
                _bounds = None
            stations_used = load_stations_from_csv(
                STATIONS_CSV, dem_bounds=_bounds, require_beta=require_beta)
        else:
            # In-file list. In spatial mode each entry must also carry
            # beta_bias / beta_amp / t_star; otherwise there is no beta for
            # that point.
            if require_beta:
                missing = [s.get('name', f'#{i}')
                           for i, s in enumerate(stations)
                           if not {'beta_bias', 'beta_amp',
                                   't_star'} <= set(s)]
                if missing:
                    raise ValueError(
                        "BETA_MODE = 'spatial' with the point product: these "
                        f"entries of the in-file list carry no beta: {missing}. "
                        "Either add 'beta_bias', 'beta_amp' and 't_star' to "
                        "each entry, or set STATIONS_CSV to the "
                        "spatialize_bias.py prediction CSV, or use "
                        "BETA_MODE = 'global'.")
            stations_used = stations

        if not stations_used:
            print('RUN_STATION is True but the station list is empty, skipped')
        else:
            model.stationProduct(stations_used, YEARS, months=MONTHS)

    print(f'\nTotal run time: {time.time()-t0:.0f} s')


if __name__ == '__main__':
    main()