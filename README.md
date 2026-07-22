# CRISP v1.0

**C**old-**R**egion **I**nversion **S**patial downscaling **P**arameterization —
a physical–statistical model for downscaling near-surface air temperature from
reanalysis to fine-scale grids over complex mountain terrain.

CRISP obtains its background temperature by three-dimensional interpolation of
pressure-level profiles, adds a dynamic inversion correction based on
hypsometric position and modulated by elevation range, adds a valley term based
on valley bottom flatness, and applies a seasonal bias term that can be either a
single global constant or a spatially varying field:

```

```

| symbol | meaning | unit |
| --- | --- | --- |
| `T_fpl` | pressure-level temperature at the fine-scale (DEM) elevation | °C |
| `T_cpl` | pressure-level temperature at the ERA5 grid-cell elevation | °C |
| `T_sur` | ERA5 2 m air temperature | °C |
| `DT` | surface-effect proxy | °C |
| `H` | hypsometric position (1 at valley bottom, 0 at summit) | – |
| `R` | elevation range within the analysis neighbourhood | m |
| `Vnorm` | normalized valley bottom flatness, MRVBF / MRVBF_max | – |
| `beta_*` | seasonal bias components | °C, – |

---

## Pipeline

| script | purpose |
| --- | --- |
| `compute_h_and_R.py` | terrain predictors `H` and `R` from DEM |
| `extract_station_beta.py` | per-station seasonal bias (`beta`) and covariates |
| `spatialize_bias.py` | spatialize the three bias coefficients by regression kriging |
| `run_downscaling.py` + `crisp_core.py` | generate the gridded and/or point product |


`extract_station_beta.py` and `spatialize_bias.py` are only needed for the **spatially varying** bias. For the **global-constant** bias, skip them and set the three constants in`run_downscaling.py`.

---

## Requirements
Python 3.8 or later.
```
numpy
scipy
pandas
netCDF4
rasterio
xarray
pyproj
scikit-learn
pykrige
```

```bash
pip install numpy scipy pandas netCDF4 rasterio xarray pyproj scikit-learn pykrige
```

---
## Input data
| input | source |
| --- | --- |
| ERA5 pressure-level temperature and geopotential | Copernicus Climate Data Store, `reanalysis-era5-pressure-levels` |
| ERA5 2 m temperature and invariant surface geopotential | Copernicus Climate Data Store, `reanalysis-era5-single-levels` |
| Digital elevation model | SRTM, 3 arcsec |
| `Vnorm` | MRVBF computed from the DEM (e.g. with SAGA GIS or WhiteboxTools), normalized by its regional maximum |
| slope, northness, sky-view factor | derived from the DEM |
---

---
## downscaling

Edit `run_downscaling.py` (only this file) and run:

```bash
python run_downscaling.py
```

`crisp_core.py` holds the model code and does not need to be edited.

In `run_downscaling.py` you set:

* `DATA_ROOT` and the file paths
* `YEARS`, `MONTHS`
* `RUN_SPATIAL` (gridded product) and `RUN_STATION` (point product); either or
  both
* `BETA_MODE`:
  * `'global'` — one constant triple `BETA_GLOBAL = {beta_bias, beta_amp,
    t_star}` applied everywhere;
  * `'spatial'` — spatially varying bias :
    * gridded product → the three `*_map.tif` are required
    * point product → `STATIONS_CSV` must carry `beta_bias_pred`,
      `beta_amp_pred`, `t_star_pred`, i.e. the prediction output for
      exactly the points to downscale
    * both → both are required

The point product takes each point's beta directly from the CSV rather than
sampling the rasters, because sampling a rasterised field back at a point
would interpolate twice and carry the rasterisation error. This matches the
end-to-end evaluation.

Points are given either as a short in-file list (global mode only, since in
spatial mode each point needs its own predicted beta) or as a CSV
(`STATIONS_CSV`) with `name`, `latitude`, `longitude`, `elevation`.

### Output

* Gridded: `T_downscaled_{year}.nc`, CF-1.7, variable `T2m` in °C, stored as
  `int16` with `scale_factor = 0.01` (readers following the CF convention
  unpack this transparently).
* Point: `station_T_{year}.csv`, one column per station.

---

## Model parameters

The calibrated parameters are set in `run_downscaling.py` . The defaults are the Qilian Mountains values from
the paper; the Alps values are also reported there. When applying the model to a
new region, substitute the parameters and recompute `H`/`R` with the matching
neighbourhood radii.

---

