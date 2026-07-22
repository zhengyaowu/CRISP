"""
crisp_core.py
=============
CRISP downscaling engine (core library). Users normally do not edit this
file; all configuration is done in run_downscaling.py.

"""
import os
import csv
import time
import warnings
import numpy as np
import netCDF4 as nc
import rasterio
from datetime import datetime, timedelta
from scipy.interpolate import RegularGridInterpolator
warnings.filterwarnings('ignore')


# ============================================================
# Helper functions
# ============================================================
def is_leap(year):
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def time_frac(date):
    """
    Position of the day within the year, used by the cosine term of beta(t).
    Defined exactly as in the beta-extraction step:
      time_frac = (dayofyear - 1) / (366 if leap else 365)
    """
    days_in_year = 366 if is_leap(date.year) else 365
    return (date.timetuple().tm_yday - 1) / days_in_year


def _fast1d_vec(t_interp, z_interp, target_ele):
    """
    Vectorised linear interpolation along the vertical (about 100 times
    faster than a per-point bisect_left).

    Parameters:
      t_interp   (n_lev, n_pts)  temperature on each level, ascending height
      z_interp   (n_lev, n_pts)  elevation of each level, ascending height
      target_ele (n_pts,)        target elevation [m]

    Targets below the lowest level are extrapolated from the lowest two
    levels; targets above the highest level use the top two levels.
    """
    n_lev, n_pts = t_interp.shape
    out = np.full(n_pts, np.nan, dtype=np.float32)
    ok  = np.isfinite(target_ele)
    if not ok.any():
        return out

    ele_ok = target_ele[ok].astype(np.float64)
    cols   = np.where(ok)[0]

    above    = z_interp[:, cols] > ele_ok[np.newaxis, :]
    n_idx    = np.argmax(above, axis=0)
    no_above = ~above.any(axis=0)
    no_below = above.all(axis=0)
    n_idx[no_above] = n_lev - 1
    n_idx[no_below] = 1
    n_idx    = np.clip(n_idx, 1, n_lev - 1)
    lo       = n_idx - 1

    upperT = t_interp[n_idx, cols]
    upperZ = z_interp[n_idx, cols]
    lowerT = t_interp[lo,    cols]
    lowerZ = z_interp[lo,    cols]

    dz = upperZ - lowerZ
    dz[dz == 0] = 1.0
    frac = (ele_ok - lowerZ) / dz
    out[ok] = (lowerT + (upperT - lowerT) * frac).astype(np.float32)
    return out


def load_stations_from_csv(csv_path, dem_bounds=None, require_beta=False):
    """
    Read a station list from CSV.

    Required columns (case-insensitive, common aliases accepted):
      name              station name
      latitude          latitude   (alias: lat)
      longitude         longitude  (aliases: lon, lng)
      elevation         elevation in m (aliases: ele, elev, altitude)

    When require_beta is True (point product in spatial beta mode), three
    further columns must be present, holding the spatialized beta of each
    point as predicted by spatialize_bias.py:
      beta_bias_pred, beta_amp_pred, t_star_pred
    (the bare names beta_bias, beta_amp, t_star are also accepted). In that
    case the CSV is the spatialize_bias.py output (e.g. val_predictions.csv),
    which carries both the coordinates and the predicted beta of each point.
    The point beta is taken directly from these columns, exactly as in the
    end-to-end evaluation, and is NOT resampled from the gridded beta rasters.

    Parameters:
      csv_path     : path to the CSV (utf-8 / utf-8-sig / gbk are tried)
      dem_bounds   : optional (lat_min, lat_max, lon_min, lon_max); stations
                     outside the range are skipped with a warning
      require_beta : if True, read and attach beta_bias / beta_amp / t_star

    Returns:
      list[dict], each with name/latitude/longitude/elevation and, when
      require_beta is True, beta_bias/beta_amp/t_star
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"station CSV not found: {csv_path}")

    alias = {
        'name':      {'name', 'station', 'station_name', 'id', 'station_id'},
        'latitude':  {'latitude', 'lat'},
        'longitude': {'longitude', 'lon', 'lng', 'long'},
        'elevation': {'elevation', 'ele', 'elev', 'altitude', 'alt'},
    }
    beta_alias = {
        'beta_bias': {'beta_bias_pred', 'beta_bias'},
        'beta_amp':  {'beta_amp_pred', 'beta_amp'},
        't_star':    {'t_star_pred', 't_star'},
    }

    rows = None
    for enc in ('utf-8-sig', 'utf-8', 'gbk', 'cp1252'):
        try:
            with open(csv_path, 'r', encoding=enc, newline='') as f:
                rows = list(csv.DictReader(f))
            break
        except UnicodeDecodeError:
            continue
    if rows is None:
        raise IOError(f"cannot determine the encoding of: {csv_path}")
    if not rows:
        raise ValueError(f"the CSV is empty: {csv_path}")

    headers     = [h for h in rows[0].keys() if h is not None]
    headers_low = {h.strip().lower(): h for h in headers}

    col_map = {}
    for canon, names in alias.items():
        found = next((headers_low[n] for n in names if n in headers_low), None)
        if found is None:
            raise ValueError(
                f"required column '{canon}' not found (accepted aliases: "
                f"{sorted(names)}). Header: {headers}")
        col_map[canon] = found

    beta_map = {}
    if require_beta:
        for canon, names in beta_alias.items():
            found = next((headers_low[n] for n in names if n in headers_low),
                         None)
            if found is None:
                raise ValueError(
                    f"BETA_MODE = 'spatial' (point product) requires column "
                    f"'{canon}' (accepted: {sorted(names)}), produced by "
                    f"spatialize_bias.py. Header: {headers}")
            beta_map[canon] = found

    stations  = []
    seen_name = set()
    n_dup, n_oob, n_badnum = 0, 0, 0
    for irow, r in enumerate(rows, start=2):
        try:
            nm  = str(r[col_map['name']]).strip()
            lat = float(r[col_map['latitude']])
            lon = float(r[col_map['longitude']])
            ele = float(r[col_map['elevation']])
        except (TypeError, ValueError):
            n_badnum += 1
            print(f"  row {irow}: cannot parse the values, skipped")
            continue
        if not nm:
            print(f"  row {irow}: empty station name, skipped")
            continue
        if dem_bounds is not None:
            la_min, la_max, lo_min, lo_max = dem_bounds
            if not (la_min <= lat <= la_max and lo_min <= lon <= lo_max):
                n_oob += 1
                print(f"  {nm} (lat={lat:.4f}, lon={lon:.4f}) "
                      f"outside the DEM extent, skipped")
                continue

        sta = {'name': nm, 'latitude': lat, 'longitude': lon, 'elevation': ele}

        if require_beta:
            try:
                sta['beta_bias'] = float(r[beta_map['beta_bias']])
                sta['beta_amp']  = float(r[beta_map['beta_amp']])
                sta['t_star']    = float(r[beta_map['t_star']])
            except (TypeError, ValueError):
                n_badnum += 1
                print(f"  {nm}: cannot parse the beta columns, skipped")
                continue

        if nm in seen_name:
            n_dup += 1
            print(f"  duplicate station name '{nm}' (row {irow}); kept, but "
                  f"the output CSV will contain repeated column names")
        seen_name.add(nm)
        stations.append(sta)

    print(f"  read from {os.path.basename(csv_path)}: "
          f"{len(stations)} valid stations"
          + (f", {n_oob} out of range" if n_oob else "")
          + (f", {n_dup} duplicate names" if n_dup else "")
          + (f", {n_badnum} unparsable rows" if n_badnum else ""))
    if not stations:
        raise ValueError(f"no valid stations in {csv_path}")
    return stations


# ============================================================
# Main class
# ============================================================
class CrispDownscaler:
    """
    CRISP downscaling product generator.

    Constructed from a configuration dictionary (see run_downscaling.py).
    The seasonal-bias configuration is read from the keys:
      beta_mode         : 'global' or 'spatial'
      beta_global       : dict(beta_bias=, beta_amp=, t_star=)   [global mode]
      beta_bias_tif     : path to beta_bias_map.tif   [spatial mode, grid]
      beta_amp_tif      : path to beta_amp_map.tif    [spatial mode, grid]
      t_star_tif        : path to t_star_map.tif      [spatial mode, grid]

    Public methods:
      spatialProduct(years, months=None)
      stationProduct(stations, years, out_csv=None, months=None)
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.frozen = cfg['frozen']
        self.region_name = cfg.get('region_name', 'study region')
        self.institution = cfg.get('institution', '')
        self.out_dir = cfg['out_dir']
        os.makedirs(self.out_dir, exist_ok=True)

        self.beta_mode = cfg.get('beta_mode', 'spatial')
        if self.beta_mode not in ('global', 'spatial'):
            raise ValueError(
                f"beta_mode must be 'global' or 'spatial', got "
                f"'{self.beta_mode}'")

        print("=" * 65)
        print(f"CRISP downscaling ({self.region_name})")
        print("=" * 65)
        print("  parameters: " +
              "  ".join(f"{k}={v}" for k, v in self.frozen.items()))
        print(f"  beta mode : {self.beta_mode}")

        # Files always required
        required = {
            'fine DEM':          cfg['dem_fine_nc'],
            'ERA5 geopotential': cfg['dem_coarse_nc'],
            'H':                 cfg['h_tif'],
            'R':                 cfg['r_tif'],
            'Vnorm':             cfg['vnorm_tif'],
        }
        # Beta rasters required only for spatial mode. Whether they are
        # actually needed also depends on whether the gridded product is
        # requested; that check is made in spatialProduct(), so that a
        # point-only run in spatial mode does not demand the TIFs.
        missing = {k: v for k, v in required.items() if not os.path.exists(v)}
        if missing:
            raise FileNotFoundError(
                "the following static files were not found:\n" +
                "\n".join(f"  [{k}] {v}" for k, v in missing.items()))
        print(f"  static files verified ({len(required)})")

        if self.beta_mode == 'global':
            g = cfg['beta_global']
            print(f"  global beta: beta_bias={g['beta_bias']}  "
                  f"beta_amp={g['beta_amp']}  t_star={g['t_star']}")

        self._sa = self._pl = self._gz = None
        self._static     = None
        self._year_cache = None

    def __del__(self):
        self._close_year()

    # -- file handles -------------------------------------------------

    def _year_files(self, year):
        cfg = self.cfg
        paths = {
            'sa': os.path.join(cfg['tsur_dir'], cfg['tsur_tmpl'].format(year=year)),
            'pl': os.path.join(cfg['pl_t_dir'], cfg['pl_t_tmpl'].format(year=year)),
            'gz': os.path.join(cfg['pl_z_dir'], cfg['pl_z_tmpl'].format(year=year)),
        }
        for key, path in paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"[{year}] '{key}' file not found:\n  {path}\n"
                    f"  check the path configuration in run_downscaling.py")
        return paths

    def _open_year(self, year):
        self._close_year()
        paths = self._year_files(year)
        self._sa = nc.Dataset(paths['sa'], 'r')
        self._pl = nc.Dataset(paths['pl'], 'r')
        self._gz = nc.Dataset(paths['gz'], 'r')
        print(f"  opened files for {year}:")
        for k in ('sa', 'pl', 'gz'):
            print(f"    {k.upper()}: {os.path.basename(paths[k])}")

    def _close_year(self):
        for attr in ('_sa', '_pl', '_gz'):
            ds = getattr(self, attr, None)
            if ds is not None:
                try:
                    ds.close()
                except Exception:
                    pass
                setattr(self, attr, None)
        self._year_cache = None

    # -- static fields ------------------------------------------------

    def _load_static(self, need_beta_rasters):
        """
        Load the DEM, terrain predictors and, when need_beta_rasters is True,
        the three spatialized beta rasters. Loaded once per run and cached.
        """
        if self._static is not None:
            return self._static
        cfg = self.cfg
        print("\nLoading static fields ...")
        t0 = time.time()

        # Fine-scale DEM
        with nc.Dataset(cfg['dem_fine_nc'], 'r') as ds:
            _ev = next((k for k in ('elevation','dem','DEM','z','height','hgt')
                        if k in ds.variables),
                       next(k for k in ds.variables
                            if k not in {'latitude','longitude','lat','lon'}))
            dem_data    = ds[_ev][:].squeeze().astype(np.float32)
            dem_lat_raw = (ds['latitude'][:].data  if 'latitude'  in ds.variables
                           else ds['lat'][:].data).astype(np.float32)
            dem_lon_raw = (ds['longitude'][:].data if 'longitude' in ds.variables
                           else ds['lon'][:].data).astype(np.float32)
        ny, nx = dem_data.shape
        dem_data[~np.isfinite(dem_data)] = np.nan

        lons2d = np.tile(dem_lon_raw,          (ny, 1)).astype(np.float32)
        lats2d = np.tile(dem_lat_raw[:, None], (1, nx)).astype(np.float32)
        dem_lon = dem_lon_raw.copy()
        dem_lat = dem_lat_raw.copy()

        # Reference flip to ascending latitude. dem_lat, dem_data and lats2d
        # are flipped together so that they stay aligned. Arrays read later
        # from GeoTIFFs (north to south) are flipped once to match (see
        # _read_tif); arrays interpolated onto lats_flat are not (see
        # era_ele_flat).
        flipped = dem_lat[0] > dem_lat[-1]
        if flipped:
            dem_lat  = dem_lat[::-1]
            dem_data = dem_data[::-1, :]
            lats2d   = lats2d[::-1, :]

        lats_flat = lats2d.ravel().astype(np.float64)
        lons_flat = lons2d.ravel().astype(np.float64)
        ele_flat  = dem_data.ravel().astype(np.float32)

        print(f"  DEM: {ny}x{nx}  "
              f"lat=[{dem_lat.min():.2f},{dem_lat.max():.2f}]  "
              f"lon=[{dem_lon.min():.2f},{dem_lon.max():.2f}]")

        # ERA5 surface elevation bilinearly interpolated onto the fine grid.
        # Bilinear (not nearest neighbour) is required so that DT_new does not
        # jump at ERA5 cell boundaries. era_ele_flat is built from the
        # already-ascending lats_flat and must NOT be flipped again.
        with nc.Dataset(cfg['dem_coarse_nc'], 'r') as ds:
            _gv = next((k for k in ('z','geopotential','Z')
                        if k in ds.variables), None)
            if _gv is None:
                raise ValueError(f"geopotential variable not found: "
                                 f"{list(ds.variables.keys())}")
            era5_geo = ds[_gv][:].squeeze().astype(np.float32) / 9.80665
            era5_lat = (ds['latitude'][:].data  if 'latitude'  in ds.variables
                        else ds['lat'][:].data).astype(np.float64)
            era5_lon = (ds['longitude'][:].data if 'longitude' in ds.variables
                        else ds['lon'][:].data).astype(np.float64)
        if era5_lat[0] > era5_lat[-1]:
            era5_lat = era5_lat[::-1]; era5_geo = era5_geo[::-1, :]

        f_era_ele = RegularGridInterpolator(
            (era5_lat, era5_lon), era5_geo.astype(np.float64),
            method='linear', bounds_error=False, fill_value=None)
        pts_all = np.column_stack([lats_flat, lons_flat])
        era_ele_flat = f_era_ele(pts_all).astype(np.float32)
        print(f"  ERA5 surface elevation: mean={np.nanmean(era_ele_flat):.0f} m")

        # Raster alignment helpers
        from rasterio.warp import reproject as _rp, Resampling as _RS2
        from rasterio.crs import CRS as _CRS2
        from rasterio.transform import from_bounds as _fb

        WGS84 = _CRS2.from_epsg(4326)
        _lon_min = float(dem_lon_raw.min()); _lon_max = float(dem_lon_raw.max())
        _lat_min = float(dem_lat_raw.min()); _lat_max = float(dem_lat_raw.max())
        _dlon = float(abs(float(dem_lon_raw[-1]) - float(dem_lon_raw[0]))) / max(nx-1, 1)
        _dlat = float(abs(float(dem_lat_raw[-1]) - float(dem_lat_raw[0]))) / max(ny-1, 1)
        _dst_tf = _fb(_lon_min - _dlon/2, _lat_min - _dlat/2,
                      _lon_max + _dlon/2, _lat_max + _dlat/2, nx, ny)

        def _fill_edges(arr):
            from scipy.ndimage import distance_transform_edt
            nan_mask = ~np.isfinite(arr)
            if not nan_mask.any():
                return arr
            _, idx = distance_transform_edt(nan_mask, return_indices=True)
            filled = arr.copy()
            filled[nan_mask] = arr[idx[0][nan_mask], idx[1][nan_mask]]
            return filled

        def _read_tif(path, name):
            with rasterio.open(path) as src_:
                nd_  = src_.nodata if src_.nodata is not None else -9999.0
                data = np.empty((ny, nx), dtype=np.float32)
                _rp(source=rasterio.band(src_, 1), destination=data,
                    src_transform=src_.transform, src_crs=src_.crs,
                    src_nodata=nd_, dst_transform=_dst_tf,
                    dst_crs=WGS84, dst_nodata=np.nan,
                    resampling=_RS2.bilinear)
            data[~np.isfinite(data)] = np.nan
            data = _fill_edges(data)
            # reproject returns rows north-to-south; flip once to the
            # ascending convention (unlike era_ele_flat, which was already
            # interpolated onto ascending lats_flat).
            if flipped:
                data = data[::-1, :]
            print(f"  {name}: [{np.nanmin(data):.3f}, {np.nanmax(data):.3f}]")
            return data

        H   = _read_tif(cfg['h_tif'],     'H')
        R   = _read_tif(cfg['r_tif'],     'R')
        Vn  = _read_tif(cfg['vnorm_tif'], 'Vnorm').clip(0, 1)

        # Precompute h and alpha(h)
        S_L   = np.exp(-R / self.frozen['gamma_L'])
        h     = H * (1.0 - S_L) + S_L
        alpha = self.frozen['alpha_i'] + np.exp(self.frozen['alpha_s'] * h) - 1.0
        print(f"  alpha(h): mean={np.nanmean(alpha):.4f}")

        static = dict(
            ny=ny, nx=nx,
            dem_lat=dem_lat, dem_lon=dem_lon,
            lats_flat=lats_flat, lons_flat=lons_flat,
            ele_flat=ele_flat, era_ele_flat=era_ele_flat,
            alpha=alpha, Vnorm=Vn,
        )

        # Spatialized beta rasters (only when a gridded product in spatial
        # mode is requested)
        if need_beta_rasters:
            bb = _read_tif(cfg['beta_bias_tif'], 'beta_bias')
            ba = np.maximum(_read_tif(cfg['beta_amp_tif'], 'beta_amp'), 0.0)
            ts = np.clip(_read_tif(cfg['t_star_tif'], 't_star'), 0.0, 1.0)
            static.update(beta_bias=bb, beta_amp=ba, t_star=ts)

        print(f"  static fields loaded ({time.time()-t0:.1f} s)")
        self._static = static
        return static

    # -- ERA5 surface elevation at stations (single interpolation) ----

    def _get_station_era_ele(self, sta_lats, sta_lons):
        """
        ERA5 surface elevation at station locations, interpolated in one step
        from the ERA5 geopotential file. See the class docstring for why the
        rasterised era_ele_flat must not be reused here.
        """
        with nc.Dataset(self.cfg['dem_coarse_nc'], 'r') as ds:
            _gv = next((k for k in ('z','geopotential','Z')
                        if k in ds.variables), None)
            if _gv is None:
                raise ValueError(f"geopotential variable not found: "
                                 f"{list(ds.variables.keys())}")
            era5_geo = ds[_gv][:].squeeze().astype(np.float64) / 9.80665
            era5_lat = (ds['latitude'][:].data  if 'latitude'  in ds.variables
                        else ds['lat'][:].data).astype(np.float64)
            era5_lon = (ds['longitude'][:].data if 'longitude' in ds.variables
                        else ds['lon'][:].data).astype(np.float64)
        if era5_lat[0] > era5_lat[-1]:
            era5_lat = era5_lat[::-1]; era5_geo = era5_geo[::-1, :]
        if era5_lon[0] > era5_lon[-1]:
            era5_lon = era5_lon[::-1]; era5_geo = era5_geo[:, ::-1]

        f_ele = RegularGridInterpolator(
            (era5_lat, era5_lon), era5_geo,
            method='linear', bounds_error=False, fill_value=None)
        pts = np.column_stack([np.asarray(sta_lats, dtype=np.float64),
                               np.asarray(sta_lons, dtype=np.float64)])
        return f_ele(pts).astype(np.float32)

    # -- annual ERA5 cache --------------------------------------------

    def _load_year_cache(self):
        """
        Read one year of ERA5 into memory and precompute the bilinear
        interpolation weights. Removes random disk I/O from the time-step
        loop.
        """
        print("  reading one year of ERA5 into memory ...")
        t0 = time.time()
        pl, gz, sa = self._pl, self._gz, self._sa

        _ex = {'latitude','longitude','level','pressure_level','time','valid_time'}
        _se = {'latitude','longitude','time','valid_time'}
        try:
            t_name = next((k for k in ('t','temperature','T','air_temperature')
                           if k in pl.variables),
                          next(k for k in pl.variables if k not in _ex))
        except StopIteration:
            raise ValueError(f"cannot identify the pressure-level temperature "
                             f"variable: {list(pl.variables.keys())}")
        try:
            z_name = next((k for k in ('z','geopotential','Z')
                           if k in gz.variables),
                          next(k for k in gz.variables if k not in _ex))
        except StopIteration:
            raise ValueError(f"cannot identify the geopotential variable: "
                             f"{list(gz.variables.keys())}")
        try:
            t2m_name = next((k for k in ('t2m','T2m','VAR_2T','2m_temperature')
                             if k in sa.variables),
                            next(k for k in sa.variables if k not in _se))
        except StopIteration:
            raise ValueError(f"cannot identify the 2 m temperature variable: "
                             f"{list(sa.variables.keys())}")

        lev_name = 'pressure_level' if 'pressure_level' in pl.variables else 'level'

        all_T   = pl.variables[t_name][:].astype(np.float32) - 273.15
        all_Z   = gz.variables[z_name][:].astype(np.float32) / 9.80665
        all_T2m = sa.variables[t2m_name][:].astype(np.float32) - 273.15
        lev     = pl.variables[lev_name][:].astype(np.float32)

        # Normalise to time-first: some geopotential files are stored as
        # (level, time, lat, lon) while temperature is (time, level, lat,
        # lon). Indexing all_Z with a time index would otherwise go out of
        # bounds.
        def _ensure_time_first_4d(arr, n_lev_sz):
            if arr.ndim == 4 and arr.shape[0] == n_lev_sz and arr.shape[1] != n_lev_sz:
                arr = arr.transpose(1, 0, 2, 3)
            return arr
        n_lev_sz = len(lev)
        all_T = _ensure_time_first_4d(all_T, n_lev_sz)
        all_Z = _ensure_time_first_4d(all_Z, n_lev_sz)

        r_lat  = (pl.variables['latitude'][:]  if 'latitude'  in pl.variables
                  else pl.variables['lat'][:]).astype(np.float32)
        r_lon  = (pl.variables['longitude'][:] if 'longitude' in pl.variables
                  else pl.variables['lon'][:]).astype(np.float32)
        sa_lat = (sa.variables['latitude'][:]  if 'latitude'  in sa.variables
                  else sa.variables['lat'][:]).astype(np.float32)
        sa_lon = (sa.variables['longitude'][:] if 'longitude' in sa.variables
                  else sa.variables['lon'][:]).astype(np.float32)

        if r_lat[0]  > r_lat[-1]:  r_lat  = r_lat[::-1];  all_T = all_T[:,:,::-1,:]; all_Z = all_Z[:,:,::-1,:]
        if r_lon[0]  > r_lon[-1]:  r_lon  = r_lon[::-1];  all_T = all_T[:,:,:,::-1]; all_Z = all_Z[:,:,:,::-1]
        if sa_lat[0] > sa_lat[-1]: sa_lat = sa_lat[::-1]; all_T2m = all_T2m[:,::-1,:]
        if sa_lon[0] > sa_lon[-1]: sa_lon = sa_lon[::-1]; all_T2m = all_T2m[:,:,::-1]
        # Ascending pressure means descending height: reverse the level axis
        if lev[0] < lev[-1]:
            all_T = all_T[:,::-1,:,:]; all_Z = all_Z[:,::-1,:,:]

        print(f"    PL-T: {all_T.shape}  PL-Z: {all_Z.shape}  "
              f"T2m: {all_T2m.shape}  ({time.time()-t0:.1f} s)")

        dem_lat = self._static['dem_lat']
        dem_lon = self._static['dem_lon']

        def _bilinear_weights(dem_coord, era_coord):
            coords = np.interp(dem_coord, era_coord.astype(np.float64),
                               np.arange(len(era_coord), dtype=np.float64)
                               ).astype(np.float32)
            lo   = np.clip(coords.astype(np.int32), 0, len(era_coord) - 2)
            frac = coords - lo.astype(np.float32)
            return lo, frac

        lat_lo_pl, fy_pl = _bilinear_weights(dem_lat, r_lat)
        lon_lo_pl, fx_pl = _bilinear_weights(dem_lon, r_lon)
        lat_lo_sa, fy_sa = _bilinear_weights(dem_lat, sa_lat)
        lon_lo_sa, fx_sa = _bilinear_weights(dem_lon, sa_lon)

        self._year_cache = dict(
            all_T=all_T, all_Z=all_Z, all_T2m=all_T2m,
            r_lat=r_lat, r_lon=r_lon, sa_lat=sa_lat, sa_lon=sa_lon,
            lat_lo_pl=lat_lo_pl, lon_lo_pl=lon_lo_pl, fy_pl=fy_pl, fx_pl=fx_pl,
            lat_lo_sa=lat_lo_sa, lon_lo_sa=lon_lo_sa, fy_sa=fy_sa, fx_sa=fx_sa,
        )
        print("  year cached; interpolation weights precomputed")

    # -- ERA5 interpolation: gridded ----------------------------------

    def _get_Tfpl_DT(self, ind_time, ele_2d, era_ele_2d):
        cache = self._year_cache
        ny, nx = self._static['ny'], self._static['nx']

        gridT = cache['all_T'][ind_time]
        gridZ = cache['all_Z'][ind_time]
        t2m   = cache['all_T2m'][ind_time]

        lat_lo = cache['lat_lo_pl']; lon_lo = cache['lon_lo_pl']
        fy = cache['fy_pl'][:, None]; fx = cache['fx_pl'][None, :]
        n_lev = gridT.shape[0]

        t_blk = np.empty((n_lev, ny * nx), dtype=np.float32)
        z_blk = np.empty((n_lev, ny * nx), dtype=np.float32)
        for i in range(n_lev):
            dT = gridT[i]; dZ = gridZ[i]
            t_blk[i] = (dT[np.ix_(lat_lo,   lon_lo  )] * (1-fy) * (1-fx) +
                         dT[np.ix_(lat_lo,   lon_lo+1)] * (1-fy) *    fx  +
                         dT[np.ix_(lat_lo+1, lon_lo  )] *    fy  * (1-fx) +
                         dT[np.ix_(lat_lo+1, lon_lo+1)] *    fy  *    fx ).ravel()
            z_blk[i] = (dZ[np.ix_(lat_lo,   lon_lo  )] * (1-fy) * (1-fx) +
                         dZ[np.ix_(lat_lo,   lon_lo+1)] * (1-fy) *    fx  +
                         dZ[np.ix_(lat_lo+1, lon_lo  )] *    fy  * (1-fx) +
                         dZ[np.ix_(lat_lo+1, lon_lo+1)] *    fy  *    fx ).ravel()

        t_fpl = _fast1d_vec(t_blk, z_blk, ele_2d.ravel())
        t_cpl = _fast1d_vec(t_blk, z_blk, era_ele_2d.ravel())

        lat_s = cache['lat_lo_sa']; lon_s = cache['lon_lo_sa']
        fy_s = cache['fy_sa'][:, None]; fx_s = cache['fx_sa'][None, :]
        t_sur = (t2m[np.ix_(lat_s,   lon_s  )] * (1-fy_s) * (1-fx_s) +
                 t2m[np.ix_(lat_s,   lon_s+1)] * (1-fy_s) *    fx_s  +
                 t2m[np.ix_(lat_s+1, lon_s  )] *    fy_s  * (1-fx_s) +
                 t2m[np.ix_(lat_s+1, lon_s+1)] *    fy_s  *    fx_s
                 ).ravel().astype(np.float32)

        return t_fpl, t_sur - t_cpl

    # -- ERA5 interpolation: points -----------------------------------

    def _get_Tfpl_DT_pts(self, ind_time, sta_lats, sta_lons,
                          sta_eles, sta_era_ele):
        cache = self._year_cache
        gridT = cache['all_T'][ind_time]
        gridZ = cache['all_Z'][ind_time]
        t2m   = cache['all_T2m'][ind_time]
        r_lat = cache['r_lat'].astype(np.float64)
        r_lon = cache['r_lon'].astype(np.float64)
        sa_lat= cache['sa_lat'].astype(np.float64)
        sa_lon= cache['sa_lon'].astype(np.float64)

        pts = np.column_stack([sta_lats, sta_lons])
        n_lev = gridT.shape[0]
        n_sta = len(sta_eles)

        t_on_sta = np.empty((n_lev, n_sta), dtype=np.float32)
        z_on_sta = np.empty((n_lev, n_sta), dtype=np.float32)
        for i in range(n_lev):
            ft = RegularGridInterpolator((r_lat, r_lon), gridT[i].astype(np.float64),
                                         method='linear', bounds_error=False, fill_value=None)
            fz = RegularGridInterpolator((r_lat, r_lon), gridZ[i].astype(np.float64),
                                         method='linear', bounds_error=False, fill_value=None)
            t_on_sta[i] = ft(pts).astype(np.float32)
            z_on_sta[i] = fz(pts).astype(np.float32)

        t_fpl = _fast1d_vec(t_on_sta, z_on_sta, sta_eles)
        t_cpl = _fast1d_vec(t_on_sta, z_on_sta, sta_era_ele)

        ft2m = RegularGridInterpolator((sa_lat, sa_lon), t2m.astype(np.float64),
                                        method='linear', bounds_error=False, fill_value=None)
        t_sur = ft2m(pts).astype(np.float32)
        return t_fpl, t_sur - t_cpl

    # -- model equation -----------------------------------------------

    def _compute_T(self, T_fpl, DT_new, alpha, Vnorm,
                   beta_bias, beta_amp, t_star, tfrac):
        """
        T_mod = T_fpl + alpha(h)*DT_new + lambda*Vnorm*DT_new
              + beta_amp*cos(2*pi*(t - t_star)) + beta_bias
        Arguments broadcast, so scalars and arrays may be mixed (this is how
        global and spatial beta are handled by the same code).
        """
        cos_term = np.cos(2.0 * np.pi * (tfrac - t_star))
        beta_t   = beta_amp * cos_term + beta_bias
        T_mod    = (T_fpl
                    + alpha * DT_new
                    + self.frozen['lambda'] * Vnorm * DT_new
                    + beta_t)
        return T_mod.astype(np.float32)

    # -- gridded product ----------------------------------------------

    def spatialProduct(self, years, months=None):
        """
        Annual gridded downscaled temperature NetCDF.

        In spatial beta mode, the three beta rasters are required here and
        their absence raises an error (a point-only run does not reach this
        method and therefore does not need them).
        """
        need_beta_rasters = (self.beta_mode == 'spatial')
        if need_beta_rasters:
            for key, label in [('beta_bias_tif', 'beta_bias_map'),
                               ('beta_amp_tif', 'beta_amp_map'),
                               ('t_star_tif', 't_star_map')]:
                p = self.cfg.get(key, '')
                if not p or not os.path.exists(p):
                    raise FileNotFoundError(
                        f"BETA_MODE = 'spatial' with a gridded product requires "
                        f"the {label} GeoTIFF (from spatialize_bias.py):\n  {p}")

        static  = self._load_static(need_beta_rasters=need_beta_rasters)
        ny, nx  = static['ny'], static['nx']
        dem_lat = static['dem_lat']; dem_lon = static['dem_lon']

        alpha = static['alpha'].ravel()
        Vnorm = static['Vnorm'].ravel()
        ele_2d     = static['ele_flat'].reshape(ny, nx)
        era_ele_2d = static['era_ele_flat'].reshape(ny, nx)

        if self.beta_mode == 'spatial':
            bb = static['beta_bias'].ravel()
            ba = static['beta_amp'].ravel()
            ts = static['t_star'].ravel()
        else:
            g = self.cfg['beta_global']
            bb = np.float32(g['beta_bias'])
            ba = np.float32(g['beta_amp'])
            ts = np.float32(g['t_star'])

        for year in years:
            t_year = time.time()
            print(f"\n{'='*65}")
            print(f"{year}: gridded product ...")

            try:
                self._open_year(year)
            except FileNotFoundError as e:
                print(f"  skipped: {e}"); continue

            sa = self._sa
            tv = 'time' if 'time' in sa.variables else 'valid_time'
            era5_dts = [datetime(t.year, t.month, t.day, t.hour)
                        for t in nc.num2date(
                            sa.variables[tv][:], units=sa.variables[tv].units,
                            calendar=getattr(sa.variables[tv], 'calendar', 'standard'))]
            print(f"  ERA5 time steps: {len(era5_dts)}")

            self._load_year_cache()

            dates = [datetime(year, 1, 1) + timedelta(days=i)
                     for i in range(366 if is_leap(year) else 365)]
            if months is not None:
                dates = [d for d in dates if d.month in months]
                month_suffix = '_m' + '_'.join(f'{m:02d}' for m in sorted(set(months)))
                print(f"  month filter: {sorted(set(months))}  {len(dates)} days")
            else:
                month_suffix = ''
            n_days = len(dates)

            out_path = os.path.join(self.out_dir, f'T_downscaled_{year}{month_suffix}.nc')
            ds = nc.Dataset(out_path, 'w', format='NETCDF4')
            n_ok = 0; n_miss = 0
            try:
                ds.setncattr('Conventions', 'CF-1.7')
                ds.setncattr('title',
                    f'CRISP downscaled 2 m air temperature, {self.region_name}, {year}')
                if self.institution:
                    ds.setncattr('institution', self.institution)
                ds.setncattr('source', 'ERA5 reanalysis; CRISP v1.0')
                ds.setncattr('history', f'Created {datetime.now().isoformat()}')
                ds.setncattr('frozen_params', str(self.frozen))
                ds.setncattr('beta_mode', self.beta_mode)

                ds.createDimension('time', None)
                ds.createDimension('lat',  ny)
                ds.createDimension('lon',  nx)

                tv_nc = ds.createVariable('time', 'i4', ('time',))
                tv_nc.units = f'days since {year}-01-01'
                tv_nc.calendar = 'standard'; tv_nc.standard_name = 'time'
                tv_nc.long_name = 'time'; tv_nc.axis = 'T'

                lv = ds.createVariable('lat', 'f4', ('lat',))
                lv[:] = dem_lat.astype(np.float32)
                lv.units = 'degrees_north'; lv.standard_name = 'latitude'
                lv.long_name = 'latitude'; lv.axis = 'Y'

                lov = ds.createVariable('lon', 'f4', ('lon',))
                lov[:] = dem_lon.astype(np.float32)
                lov.units = 'degrees_east'; lov.standard_name = 'longitude'
                lov.long_name = 'longitude'; lov.axis = 'X'

                # int16 packed storage, scale_factor 0.01 (CF convention);
                # readers unpack transparently. One chunk per day.
                T_v = ds.createVariable(
                    'T2m', 'i2', ('time', 'lat', 'lon'),
                    fill_value=np.int16(-32768),
                    chunksizes=(1, ny, nx),
                    zlib=True, complevel=9, shuffle=False)
                T_v.scale_factor = np.float32(0.01)
                T_v.add_offset   = np.float32(0.0)
                T_v.units = 'degrees_Celsius'
                T_v.long_name = 'Downscaled 2m air temperature'
                T_v.standard_name = 'air_temperature'
                T_v.model = 'CRISP v1.0'

                for d_idx, date in enumerate(dates):
                    day_idx = [i for i, dt in enumerate(era5_dts)
                               if dt.date() == date.date()]
                    if not day_idx:
                        T_v[d_idx, :, :] = np.full((ny, nx), np.nan, dtype=np.float32)
                        tv_nc[d_idx] = d_idx; n_miss += 1; continue

                    tfrac = time_frac(date)
                    T_sum = np.zeros(ny * nx, dtype=np.float64)
                    for i_era in day_idx:
                        T_fpl, DT_new = self._get_Tfpl_DT(i_era, ele_2d, era_ele_2d)
                        T_mod = self._compute_T(T_fpl, DT_new, alpha, Vnorm,
                                                bb, ba, ts, tfrac)
                        T_sum += T_mod.astype(np.float64)

                    T_day = (T_sum / len(day_idx)).astype(np.float32)
                    T_v[d_idx, :, :] = np.ma.masked_invalid(T_day.reshape(ny, nx))
                    tv_nc[d_idx] = d_idx; n_ok += 1

                    if (d_idx + 1) % 10 == 0 or d_idx == n_days - 1:
                        elapsed = time.time() - t_year
                        rate = (d_idx + 1) / elapsed
                        remain = (n_days - d_idx - 1) / rate if rate > 0 else 0
                        print(f"  {date.strftime('%Y-%m-%d')}: {d_idx+1}/{n_days} "
                              f"days  {rate:.1f} days/s  eta {remain:.0f} s", end='\r')
            finally:
                ds.close()

            self._close_year()
            print(f"\n  {year}: {n_ok} days written, {n_miss} missing "
                  f"({time.time()-t_year:.0f} s)")
            if os.path.exists(out_path):
                print(f"  -> {os.path.basename(out_path)}  "
                      f"{os.path.getsize(out_path)/1024**3:.2f} GB")

        print(f"\nGridded product complete. Output directory: {self.out_dir}")

    # -- point product ------------------------------------------------

    def stationProduct(self, stations, years, out_csv=None, months=None):
        """
        Daily downscaled temperature time series at given stations.

        In spatial beta mode, each station dict must carry its own
        beta_bias / beta_amp / t_star (as read from the spatialize_bias.py
        output CSV). In global mode, the single global triple is applied to
        all stations.
        """
        # In spatial mode the three beta rasters are required and are sampled
        # at the station coordinates, exactly as the gridded product samples
        # them per cell. The station list itself only provides coordinates and
        # elevation; the beta of each point comes from the CSV.
        #
        # The point product never uses the gridded beta rasters. Sampling them
        # at a station would interpolate twice (regression kriging onto the
        # 90 m grid, then back to the point) and carry the smoothing error of
        # the rasterisation, in the same way that reusing the rasterised ERA5
        # surface elevation would (see _get_station_era_ele). Instead the beta
        # predicted directly at each point by spatialize_bias.py is read from
        # the CSV, which is what the end-to-end evaluation does.
        static  = self._load_static(need_beta_rasters=False)
        dem_lat = static['dem_lat']; dem_lon = static['dem_lon']

        names    = [s['name']      for s in stations]
        sta_lats = np.array([s['latitude']  for s in stations], dtype=np.float64)
        sta_lons = np.array([s['longitude'] for s in stations], dtype=np.float64)
        sta_eles = np.array([s['elevation'] for s in stations], dtype=np.float32)

        print(f"\nPoint product: {len(stations)} stations")

        def _interp(raster_2d, lq, lnq):
            f = RegularGridInterpolator(
                (dem_lat.astype(np.float64), dem_lon.astype(np.float64)),
                raster_2d.astype(np.float64),
                method='linear', bounds_error=False, fill_value=None)
            return f(np.column_stack([lq, lnq])).astype(np.float32)

        # alpha and Vnorm are always sampled from the terrain-derived rasters
        sta_alpha = _interp(static['alpha'], sta_lats, sta_lons)
        sta_vnorm = np.clip(_interp(static['Vnorm'], sta_lats, sta_lons), 0, 1)

        # beta: per-point values from the CSV (spatial) or the global triple
        if self.beta_mode == 'spatial':
            try:
                sta_bb = np.array([s['beta_bias'] for s in stations],
                                  dtype=np.float32)
                sta_ba = np.maximum(np.array([s['beta_amp'] for s in stations],
                                             dtype=np.float32), 0.0)
                sta_ts = np.clip(np.array([s['t_star'] for s in stations],
                                          dtype=np.float32), 0.0, 1.0)
            except KeyError:
                raise ValueError(
                    "BETA_MODE = 'spatial' (point product) requires each point "
                    "to carry beta_bias / beta_amp / t_star, read from the "
                    "spatialize_bias.py output CSV (columns beta_bias_pred / "
                    "beta_amp_pred / t_star_pred).")
            print(f"  beta: per-point values predicted by spatialize_bias.py "
                  f"(beta_bias {sta_bb.min():+.3f} to {sta_bb.max():+.3f})")
        else:
            g = self.cfg['beta_global']
            sta_bb = np.float32(g['beta_bias'])
            sta_ba = np.float32(max(g['beta_amp'], 0.0))
            sta_ts = np.float32(min(max(g['t_star'], 0.0), 1.0))
            print(f"  beta: global constant "
                  f"(beta_bias={sta_bb}, beta_amp={sta_ba}, t_star={sta_ts})")

        # Single-step ERA5 surface elevation at the stations
        sta_era_ele = self._get_station_era_ele(sta_lats, sta_lons)
        print(f"  station ERA5 surface elevation (single interpolation): "
              f"mean={np.nanmean(sta_era_ele):.0f} m")

        for year in years:
            t_year = time.time()
            print(f"\n{'='*65}")
            print(f"{year}: point product ...")

            try:
                self._open_year(year)
            except FileNotFoundError as e:
                print(f"  skipped: {e}"); continue

            sa = self._sa
            tv = 'time' if 'time' in sa.variables else 'valid_time'
            era5_dts = [datetime(t.year, t.month, t.day, t.hour)
                        for t in nc.num2date(
                            sa.variables[tv][:], units=sa.variables[tv].units,
                            calendar=getattr(sa.variables[tv], 'calendar', 'standard'))]
            self._load_year_cache()

            dates = [datetime(year, 1, 1) + timedelta(days=i)
                     for i in range(366 if is_leap(year) else 365)]
            if months is not None:
                dates = [d for d in dates if d.month in months]
                print(f"  month filter: {sorted(set(months))}  {len(dates)} days")

            if out_csv:
                base, ext = os.path.splitext(out_csv)
                csv_path  = f'{base}_{year}{ext}'
            else:
                csv_path = os.path.join(self.out_dir, f'station_T_{year}.csv')

            with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(['date'] + names)
                for d_idx, date in enumerate(dates):
                    day_idx = [i for i, dt in enumerate(era5_dts)
                               if dt.date() == date.date()]
                    if not day_idx:
                        writer.writerow([date.strftime('%Y-%m-%d')] + ['NA'] * len(stations))
                        continue
                    tfrac = time_frac(date)
                    T_sum = np.zeros(len(stations), dtype=np.float64)
                    for i_era in day_idx:
                        T_fpl, DT_new = self._get_Tfpl_DT_pts(
                            i_era, sta_lats, sta_lons, sta_eles, sta_era_ele)
                        T_mod = self._compute_T(T_fpl, DT_new, sta_alpha, sta_vnorm,
                                                sta_bb, sta_ba, sta_ts, tfrac)
                        T_sum += T_mod.astype(np.float64)
                    T_day = T_sum / len(day_idx)
                    writer.writerow([date.strftime('%Y-%m-%d')] +
                                    [f'{v:.3f}' if np.isfinite(v) else 'NA' for v in T_day])
                    if (d_idx + 1) % 30 == 0 or d_idx == len(dates) - 1:
                        print(f"  {date.strftime('%Y-%m-%d')}: {d_idx+1}/{len(dates)} days", end='\r')

            self._close_year()
            print(f"\n  {year}: {os.path.basename(csv_path)}  ({time.time()-t_year:.0f} s)")

        print(f"\nPoint product complete. Output directory: {self.out_dir}")