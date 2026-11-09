# SAMKit — Spatial Analysis and Mapping Toolkit (QGIS Plugin)

**Author:** Samira Safaee, Ph.D. ([samira7v@gmail.com](mailto:samira7v@gmail.com))
**Repository:** https://github.com/ssafaee/SamKit

A QGIS plugin with three tabs:

1. **DEM (download / project)** — gets a DEM for your study area and puts it
   on a metre-based UTM grid, ready for tab 2:
   - Define the area of interest by **drawing a rectangle on the map canvas**,
     by taking **the extent of any loaded layer**, by using the current map
     view, or by typing a lon/lat bounding box.
   - Download elevation data from **AWS Terrain Tiles** (no API key, global)
     or from **OpenTopography** (SRTM GL1/GL3, Copernicus GLO-30/GLO-90,
     NASADEM, ALOS AW3D30 — free API key required — and, for the USA, the
     USGS 3DEP 1 m lidar DEM), or skip downloading and
     just reproject **a DEM you already have loaded in QGIS**.
   - **Convert from raw degrees to metres**: the UTM zone is detected
     automatically from the area's centroid (including the irregular
     south-west Norway and Svalbard zones), with an optional manual EPSG
     override, target pixel size, and resampling method. Elevation values
     themselves can also be converted (international/US survey feet, or
     centimetres, to metres) for DEMs whose *vertical* unit isn't metres.

   This matters more than it looks. Slope, curvature, TWI and MRVBF all
   divide an elevation difference by a horizontal distance, so a DEM in
   degrees mixes incompatible units and produces covariates that are wrong by
   a latitude-dependent factor — about 111 km per degree of longitude at the
   equator, but only about 56 km at 60° latitude. Reprojecting to a
   metre-based CRS before computing covariates is a prerequisite, not a
   formality. Elevations in feet have the same problem in the other
   direction: they inflate every slope by about 3.28x, with no obvious
   symptom in the output raster.

2. **Environmental Covariates** — takes a DEM raster and computes:
   - Slope (degrees)
   - Aspect (degrees, 0-360, North=0 clockwise)
   - Curvature (general / profile / plan)
   - TPI (Topographic Position Index)
   - TWI (Topographic Wetness Index)
   - MRVBF & MRRTF (Multiresolution Valley Bottom / Ridge Top Flatness)

   By default, every covariate is computed directly from the DEM with
   numpy/GDAL — **no SAGA installation is required**. Slope, aspect, and TPI
   can optionally be routed through QGIS's own Processing framework instead
   for an exact GDAL match (still no SAGA needed — see below), and
   MRVBF/MRRTF can optionally use SAGA's own reference tool if you have SAGA
   installed and want exact output rather than SAMKit's approximation (also
   see below). TWI uses a real D8/MFD flow-accumulation
   algorithm (with basic pit filling) to estimate contributing/catchment
   area — this matters, because some quick "TWI" implementations you may
   encounter elsewhere substitute something like TPI for catchment area,
   which has no real hydrological basis and produces an index that doesn't
   actually represent wetness. Likewise, MRVBF/MRRTF here use a genuine
   multi-scale combination of flatness and relative elevation across
   increasing levels of DEM smoothing (per Gallant & Dowling, 2003) rather
   than a single-scale ad hoc formula. If you're sourcing terrain covariates
   from other scripts/tools, it's worth checking that their TWI/MRVBF
   actually do this multi-step/multi-scale computation — a covariate that
   looks like TWI or MRVBF but isn't will correlate poorly with real soil
   properties and can be a hidden cause of weak model performance. TPI here
   uses a uniform (equally-weighted) square window; SAGA's TPI tool instead
   uses a distance-weighted annulus kernel (verified directly against SAGA's
   source), so values can differ somewhat between the two, particularly at
   larger radii — a documented kernel-shape difference, not a bug. If you
   need results that exactly match SAGA's official tools, run those directly
   and compare — these are good-quality, standard, but not byte-identical
   implementations.

3. **Spatial Predictive Modeling** — takes a vector layer of georeferenced
   points (measured samples with X,Y) plus a stack of raster covariates
   (e.g. the environmental covariates from Tab 1, or anything else — NDVI,
   EC, gamma radiometrics, geology, etc.), and fits one of:
   - Random Forest
   - Support Vector Machine (SVM / SVR, with automatic feature standardization)
   - Regression Kriging (2-step: Random Forest or linear trend, then krige the residuals)
   - Ordinary Kriging

   ...to predict the attribute across the landscape and writes the result as
   a GeoTIFF raster, along with a validation report (R², RMSE, MAE, bias, and
   Lin's concordance correlation coefficient).

## Reference matching and validation (v1.0.0)

In v1.0.0 every covariate and model in SAMKit reproduces an established
implementation, verified with the scripts in `tests/`:

| SAMKit output | Reproduces | Agreement (validation DEM) |
|---|---|---|
| Slope / aspect, edge rule "GDAL" | `gdaldem slope/aspect -compute_edges` | max 3e-5 deg, all cells |
| Slope / aspect, edge rule "WhiteboxTools" / "SAGA" | WhiteboxTools Slope/Aspect; SAGA ta_morphometry 0 (Horn) | max < 1e-6 deg |
| Curvature, convention "SAGA-identical" | SAGA ta_morphometry 0 (Z&T) C_GENE, C_PROF, C_PLAN, C_TANG | max 6e-8 1/m |
| TPI circle / WhiteboxTools-exact / 3x3 no centre | SAGA TPI; WhiteboxTools DiffFromMeanElev; gdaldem TPI | max < 5e-7 m |
| MRVBF / MRRTF (built-in) | SAGA ta_morphometry 8 | bit-identical (max diff 0) |
| MRVBF / MRRTF (SAGA route) | SAGA ta_morphometry 8 | identical (cell size verified) |
| TWI (built-in) | WhiteboxTools FillDepressions + D8 SCA + WetnessIndex | r = 0.98 (exact with the WhiteboxTools route) |
| SVM | R e1071 (libsvm) | identical predictions |
| Random Forest (preset "R randomForest") | R randomForest defaults | statistically equivalent (different RNG) |
| Ordinary / regression kriging, gstat fitting | gstat variogram + fit.variogram(fit.method 7) + krige | identical sample variogram and WLS objective |
| Accuracy metrics | scikit-learn / SciPy | max 6e-16 |

Edge cells differ between GDAL, WhiteboxTools and SAGA only because each tool
handles missing neighbours differently; choose the edge rule of the tool you want
to match. Code derived from SAGA GIS (MRVBF port, SAGA curvature formulas) is
GPL-2.0-or-later and credited in `NOTICE`; SAMKit is distributed under the same licence.

**Upgrading from pre-1.0 builds:** regenerate MRVBF/MRRTF (the pre-1.0 SAGA route was
mis-scaled for non-square pixels and the pre-1.0 built-in version was only an
approximation), and note that the Random Forest default is now R randomForest's
settings (choose "pre-1.0 SAMKit" to reproduce old runs).

## Applications

This point-data + environmental-covariate + machine-learning/geostatistics
workflow is a well-established approach across several fields, not just
soil science. Reliable, peer-reviewed applications include:

- **Digital soil mapping (DSM)** — predicting soil properties (organic
  carbon, texture, pH, nutrients, salinity) from terrain and remote-sensing
  covariates using Random Forest, Cubist, regression kriging and related
  methods is one of the most common DSM workflows in the literature,
  and combinations such as Random Forest + Ordinary/Regression Kriging
  routinely perform best in comparative studies.
- **Soil erosion and gully erosion susceptibility mapping** — slope,
  curvature, and the Topographic Wetness Index are consistently identified
  as top predictors of erosion susceptibility in machine-learning erosion
  studies, alongside land cover and rainfall.
- **Landslide susceptibility mapping** — slope, curvature, TWI, terrain
  roughness/relief and similar terrain derivatives are standard inputs to
  Random Forest and other ML-based landslide susceptibility models, which
  are widely used for hazard planning.
- **Groundwater potential and flood/wetness mapping** — TWI in particular
  was originally developed to represent soil moisture and saturation
  potential, and terrain-derived covariates are commonly combined with
  Random Forest for groundwater potential and flood-susceptibility mapping.
- **Precision agriculture** — terrain attributes (slope, TWI, curvature,
  relative position) are widely used to delineate management zones and
  guide site-specific nutrient/irrigation management.
- **Land degradation and habitat/land-use suitability mapping** — the same
  general covariate + point-sample + spatial-prediction pattern is applied
  in land degradation assessment and species/habitat distribution modeling.

If you use SAMKit for one of these applications (or a different one),
feel free to open an issue on the GitHub repository — real-world use cases
help prioritize future features.

## Installation

1. Download/unzip this plugin folder as `samkit`.
2. Copy the whole `samkit` folder into your QGIS plugins directory:
   - Windows: `C:\Users\<you>\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\`
   - macOS: `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/`
   - Linux: `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/`
3. In QGIS: **Plugins → Manage and Install Plugins → Installed**, tick
   "SAMKit (Spatial Analysis and Mapping Toolkit)" to enable it (or restart QGIS).
4. A toolbar icon and a "SAMKit" entry under the Plugins menu will appear.

The plugin window is sized to fit inside your screen/QGIS window automatically,
and each tab scrolls internally if its content doesn't fully fit — the SAMKit
window itself will never grow taller than your screen.

## Python package requirements

QGIS ships numpy, scipy and GDAL already, so **Tab 1 works out of the box**.

Tab 2 needs two optional packages depending on which model you use:
`scikit-learn` (Random Forest, SVM, and the trend model in Regression
Kriging) and `pykrige` (Ordinary Kriging and Regression Kriging).

### Option A — built-in installer (try this first)

Open Tab 2 ("Spatial Predictive Modeling") — at the top there is a
**"Python dependencies"** panel that:
- shows the exact Python interpreter SAMKit is running under,
- shows install status + version for each optional package,
- lets you tick the ones you want and click **"Install checked packages"**,
  which runs `pip install` against that interpreter and streams the output
  into the log box below. It automatically retries with `--user` on
  permission errors and with `--break-system-packages` on Linux
  "externally managed environment" errors, so most installs succeed without
  any manual steps.

**Important:** because Python caches which packages are available for the
lifetime of a running process, if you install a package while QGIS is
already open, you may need to **close and reopen the SAMKit window (or
restart QGIS) once** before the status updates from "not installed" to
"installed" — this is a general Python/QGIS behavior, not specific to this
plugin's installer.

### Option B — install manually

If the built-in installer fails (e.g. locked-down permissions), install the
packages by hand into QGIS's own Python environment:

- **Windows**, from the OSGeo4W Shell:
  ```
  py3_env
  python -m pip install scikit-learn pykrige
  ```
- **macOS/Linux**, from a terminal:
  ```
  /path/to/qgis/bin/python3 -m pip install scikit-learn pykrige
  ```
  (Find the exact interpreter by running `import sys; print(sys.executable)`
  in the QGIS Python console — the "Python dependencies" panel in Tab 2
  also shows this path directly.)
- If you see an "externally managed environment" error on Linux, add
  `--break-system-packages` to the pip command above.
- If you see a permissions error on Windows, either run the OSGeo4W Shell /
  QGIS as Administrator once, or add `--user` to the pip command.

After installing manually, restart QGIS (or reopen the SAMKit window) so the
new packages are detected, then click "Refresh status" in the dependency
panel to confirm.

### Option C — run the standalone `install_dependencies.py` script

If the in-app installer (Option A) fails on a particular machine — this can
happen for reasons outside SAMKit's control (locked-down permissions,
network/proxy restrictions, an unusual QGIS install layout) — a
self-contained script is included at the root of the plugin folder:
`install_dependencies.py`. It installs the same two packages
(scikit-learn, pykrige) with the same permission/environment fallbacks as
the in-app installer, but can be run two ways:

- **Inside QGIS's own Python Console** (recommended — this always uses the
  correct environment, no matter how QGIS was installed): open
  **Plugins → Python Console**, click the "Show Editor" icon, open
  `install_dependencies.py` from your samkit plugin folder, and click
  "Run script". Or simply copy the whole file's contents and paste them
  directly into the console's input line.
- **From a terminal**, once you know which Python QGIS uses (the script
  prints this path the first time you run it via the console, so you can
  reuse it later): e.g. `python install_dependencies.py` from an OSGeo4W
  Shell on Windows (after running `py3_env`), or
  `/path/to/qgis/bin/python3 install_dependencies.py` on macOS/Linux.

The script prints exactly what it tried and why, so if packages still fail
to install, the printed diagnostics (no internet/proxy access, a broken pip,
a stubborn permissions error) tell you what to raise with your IT team or
in a GitHub issue.

| Model                        | Requires        |
|-------------------------------|-----------------|
| Random Forest                 | scikit-learn    |
| Support Vector Machine (SVM)  | scikit-learn    |
| Regression Kriging            | pykrige (+ scikit-learn if trend="random_forest") |
| Ordinary Kriging              | pykrige         |

If a package is missing, the plugin will tell you exactly what to `pip install`.

## Usage notes

- **Tab 1 (DEM — download / project)**: set the area of interest (draw on the
  canvas, use a layer's extent, use the current map view, or type a bounding
  box), pick a source, choose the output file, and click **"Get DEM"**. The
  download runs in the background, so QGIS stays usable and the transfer can
  be cancelled. When it finishes, the DEM is loaded into QGIS and selected
  automatically as the input DEM in Tab 2.
  - **AWS Terrain Tiles** need no registration. The zoom level sets the
    detail — roughly 30 m per pixel at zoom 12 near the equator, halving with
    each extra level while quadrupling the number of tiles to download (a
    400-tile safety limit applies). These tiles are a merged product and
    include ocean bathymetry, so negative values offshore are expected.
  - **OpenTopography** needs a free API key (register at
    <https://portal.opentopography.org/requestService?service=api>, then paste
    the key in — it's remembered in your QGIS settings on this machine only).
    Use it when you need a single, citable dataset with known provenance.
    SRTM and NASADEM only cover 60°N–56°S; use Copernicus GLO-30/GLO-90
    closer to the poles.
  - **USGS 3DEP 1 m via OpenTopography (USA only)**: the 1 m lidar-derived
    bare-earth DEM, from OpenTopography's separate USGS 3DEP API with the same
    API key. It exists only where 3DEP lidar has been collected, and
    OpenTopography currently limits it to academic API keys. 1 m data grows by about a million cells (4 MB) per km², so
    keep the area small -- Tab 1 shows an estimate before you download, and
    the download is streamed to disk and can be cancelled. Leave pixel size on
    "Auto" to keep the native 1 m.
  - **Convert my own DEM** downloads nothing. At the top of the tab choose
    **"Convert my own DEM to UTM (no download)"**, then click **"Browse…"** to open
    a DEM straight from disk (it is added to QGIS and selected), or pick a raster
    already loaded in QGIS; no area of interest is needed. The UTM zone is detected from that DEM, which is
    then reprojected (and optionally clipped). This is the option for a
    DEM you already have in degrees, or one whose elevations are in feet.
  - If you're behind a corporate proxy, set it in QGIS under *Settings >
    Options > Network*; if downloads still fail, fetch the DEM manually in a
    browser and use the "already loaded in QGIS" option to convert it.

- **Tab 2 (Environmental Covariates) refuses DEMs in degrees.** If the selected DEM
  is in a geographic CRS, SAMKit stops and offers to reproject it to its UTM zone
  first: computed on degrees, slope comes out near 90° everywhere and MRVBF/MRRTF
  are 0 (built-in) or entirely nodata (SAGA route).

- **Tab 2 (Environmental Covariates)**: pick a loaded DEM raster layer, an
  output folder, tick the covariates you want, and click "Calculate
  environmental covariates". TWI and MRVBF/MRRTF can be slow on large DEMs.

  **Input DEM quality matters a lot.** Every covariate here is only as good
  as the DEM it's derived from. A noisy, unfiltered, or artifact-laden DEM
  (sensor noise, interpolation artifacts, striping) propagates directly into
  slope/curvature/TWI/MRVBF and can produce misleading covariates and weak
  model results downstream in Tab 2. Clean/filter your DEM appropriately for
  your study before running Tab 1, and consider whether man-made structures
  (buildings, berms, roads, levees) should be removed — for natural
  terrain/soil-forming process analysis a bare-earth DEM is usually
  preferable, while some engineering/hydrology studies may want them kept.

- **Tab 3 (Spatial Predictive Modeling)**: pick your point layer
  (georeferenced samples with X,Y) and the numeric field to model, check the
  covariate rasters you want to use as predictors, pick a model, choose a
  validation strategy, and click "Run spatial predictive model".
  - Covariate rasters should ideally already share the same grid (extent,
    resolution, CRS). If they don't, tick "Align covariate rasters to a
    common grid" and the plugin will resample everything to match the first
    checked raster.
  - For **Ordinary Kriging**, the covariate raster(s) you check are only used
    to define the output grid (extent + resolution) — their pixel values are
    not used as predictors.
  - For **Regression Kriging**, the trend is fit on your chosen covariates
    (Random Forest by default, or plain linear regression), and the leftover
    spatial pattern in the residuals is captured with Ordinary Kriging.
  - For **SVM**, predictors *and* the target are automatically standardized
    before fitting.
  - The point layer can be in a different CRS than the rasters — it will be
    reprojected automatically.

### Covariate correlation analysis

Before running a model, click **"Compute correlation table & heatmap"** in
Tab 3 (uses your currently selected points, field, and checked covariates).
This writes a CSV correlation matrix (target vs. each covariate, and
covariate vs. covariate), pops up a color heatmap for a quick visual read
(with a **"Save PNG As…"** button in that preview window so you can save a
copy anywhere you like, in addition to the one saved automatically next to
the CSV), and logs the covariates sorted by strength of relationship with
your target, plus flags any covariate pairs correlated ≥0.8 with each other
(redundant information -- consider dropping one). Use this to decide which
covariates are actually worth including before committing to a full model run.

### Validation strategies

Three ways to split your data into calibration/validation, selectable in
Tab 2's "Validation strategy" box:

- **k-fold cross-validation (random)** — the data is split into k folds
  (default k=10, adjustable), each fold held out and predicted from a model
  trained on the rest, and all held-out predictions pooled into one
  R²/RMSE/MAE. Far more stable than a single random train/test split,
  especially for small sample sizes, where a single split can easily
  produce a misleadingly poor (or even negative) R² purely by chance. The
  **exact fold membership of every point is saved** to
  `<output_name>_cv_folds/`:
  - `fold_assignments.csv` — one row per point, which fold it belongs to
  - `all_folds_training_and_validation.csv` — a single consolidated table
    with every fold's calibration *and* validation rows stacked together
    (tagged by `fold` number and `role`), so all 10 folds' training and
    validation samples are in one file to open
  - `fold_N_calibration.csv` / `fold_N_validation.csv` — the same
    information split into one pair of files per fold, for anyone who
    prefers that layout

  so any split can be referenced or reproduced later.
- **Spatial block (calibration polygon)** — for spatially clustered or
  autocorrelated data, random k-fold splits can be overly optimistic because
  nearby held-out points are easy to predict from their neighbors. This
  option lets you define an actual calibration *region* two ways:
  - **Draw it directly**: click "Draw calibration polygon on map…", then
    left-click on the QGIS canvas to place each vertex of any shape.
    Double-click, right-click or press Enter to finish (at least 3
    vertices); Backspace removes the last vertex and Esc cancels. A
    self-intersecting outline is repaired automatically. This creates a
    small layer named `SAMKit_calibration_region` and selects it
    automatically; drawing again replaces it.
  - **Choose an existing polygon layer**: pick any polygon layer already in
    the project (for example management zones, or several polygons
    digitized with QGIS's own editing tools) from the dropdown.

  Either way, choose whether points inside or outside the polygon are the
  calibration set. The model is fit on the calibration points only and
  evaluated on the (spatially separate) validation points — a stronger test
  of how well it extrapolates to unsampled areas. Both subsets are saved to
  `04_calibration_validation_split/calibration_points.csv` (inside the run's results folder) and
  `validation_points.csv`, and the calibration region boundary itself is
  saved next to them as `calibration_region.gpkg` (so a drawn polygon is
  kept with the run even though the on-map layer is temporary).
- **External validation layer** — calibrate on your whole points layer (no
  internal split) and always validate against a completely separate,
  independently-collected point layer. This matches the repeated-calibration
  benchmarking design used in some field research workflows: many different
  calibration subsets (or scenarios/resolutions/depths) are each compared
  against the *same* fixed, independent held-out validation set, so results
  are directly comparable across runs. Pick the validation layer and, if its
  attribute field has a different name than your calibration field, uncheck
  "Same field name..." and pick it explicitly. Both point sets are saved to
  `<output_name>_external_validation/calibration_points.csv` and
  `external_validation_points.csv`.

If your R² still looks weak after proper validation, it more likely reflects
a genuinely weak relationship between your covariates and the target
property (or too few samples / covariates that don't actually explain its
variation) rather than an evaluation artifact.

### Variogram controls (Ordinary/Regression Kriging)

Each kriging-based model's parameter panel lets you pick the variogram model
shape (linear, power, gaussian, spherical, exponential) and, by default,
**auto-fits** nugget/sill/range from your data — the same behavior as
pykrige's and gstat's defaults. Two additional controls:

- **Manual nugget/sill/range** — uncheck "Auto-fit ... (recommended)" to
  enter exact values instead (spherical/exponential/gaussian only; `sill`
  here means the *total* sill, i.e. nugget + partial sill, matching common
  geostatistics convention). Useful for matching a predetermined variogram
  from prior work.
- **Automatic fallback to a linear variogram** — if the chosen model fits a
  degenerate (near-zero sill) or fails to fit at all, SAMKit automatically
  retries with a linear variogram and logs that it did so, rather than
  silently returning a poor/meaningless surface.
- **Prediction uncertainty**: for Ordinary and Regression
  Kriging, under every validation design, a companion `<output_name>_kriging_variance.tif` raster is
  written alongside the prediction, giving a per-pixel estimate of kriging
  variance (loaded into QGIS automatically too, if that option is checked).
  For Regression Kriging this is the residual-kriging variance only — it
  does not include the trend model's own uncertainty.

### Flow routing for TWI: D8 or MFD

Tab 1's TWI option lets you choose the flow-routing method: **D8** (default;
each cell's flow goes entirely to its single steepest-descent neighbour) or
**MFD** (multiple flow direction; flow spreads across all downslope
neighbours weighted by slope). MFD generally gives smoother, often more
realistic catchment areas on hillslopes, at the cost of somewhat more
dispersed flow in narrow valleys where D8's thread-like channels can be more
realistic. Both are available so you can choose what suits your terrain.

### Computation method for slope / aspect / TPI: built-in vs. exact GDAL

Tab 1's "Computation method" selector controls how slope, aspect, and TPI
are computed:

- **SAMKit built-in (default)** — self-contained numpy/GDAL implementation,
  no dependencies beyond what QGIS already ships with, and TPI's window
  radius is freely adjustable.
- **Exact GDAL via QGIS Processing** — routes these three through QGIS's
  own Processing framework (`gdal:slope`, `gdal:aspect`,
  `gdal:tpitopographicpositionindex`), which calls `gdaldem` directly. Since
  GDAL ships as a core part of QGIS itself, this needs no extra install and
  reproduces byte-for-byte what QGIS's own Processing Toolbox would give
  you for the same DEM. One limitation: GDAL's TPI implementation always
  uses a fixed 3x3 (immediate 8-neighbour) window, so the radius setting is
  disabled and ignored in this mode.

This choice **only** affects slope, aspect, and TPI. Curvature and TWI have
no GDAL equivalent at all, and no separate exact option is offered for
either — an exact version of them exists only in SAGA GIS, and adding that
would reintroduce the SAGA-provider dependency (and its associated
"provider not registered" failures) that Tab 1 is deliberately built to
avoid; if you need SAGA-exact curvature/TWI, run SAGA's own tools directly
and compare, as already noted above.

### Exact MRVBF/MRRTF via SAGA (opt-in)

MRVBF/MRRTF is the one covariate where SAMKit *does* offer an exact,
SAGA-backed alternative, alongside its built-in approximation — because
unlike curvature/TWI, there's a specific, well-known reference tool
(SAGA's own `ta_morphometry` 8, the genuine Gallant & Dowling (2003)
implementation) worth making directly accessible for anyone who already
has SAGA installed. Tick **"Use SAGA for exact MRVBF/MRRTF"** in Tab 1 to
call it instead of SAMKit's built-in approximation.

SAMKit uses whichever SAGA route works, so it is **not tied to a specific
SAGA or plugin version**:

1. If a QGIS SAGA Processing provider is registered (`saga:` on QGIS < 3.30,
   or `sagang:` from the "Processing Saga NextGen Provider" plugin), SAMKit
   uses it. The provider prefix is resolved at runtime.
2. Otherwise SAMKit calls SAGA's own command-line program
   (`saga_cmd ta_morphometry 8`) **directly**. This needs no provider plugin
   and works with any installed SAGA version (2.x–9.x), including the SAGA
   that ships with OSGeo4W — the MRVBF tool's parameters have been stable
   across all of them. All raster conversion is handled by SAMKit via GDAL,
   so it doesn't depend on a particular SAGA build's import/export tools.

This matters because QGIS 3.30 removed the built-in SAGA provider, and its
replacement NextGen plugin was later pulled from the official plugin
repository and pins SAGA ≥ 9.1 — so the `saga_cmd` route is the reliable,
version-independent path. SAMKit auto-detects `saga_cmd` (PATH, the
`SAGA_CMD` environment variable, the SAGA bundled next to QGIS, and common
install folders); if it isn't found, enter the SAGA folder in the **SAGA
folder** field beneath the checkbox. If no SAGA is available at all you get a
clear error, and can simply uncheck the box to fall back to the built-in
version. The "scale levels" setting only
applies to the built-in version — SAGA's tool has its own internal scale
handling (`MAX_RES`), not exposed in SAMKit's UI.

### Publication-quality maps (automatic)

Every Spatial Predictive Modeling run also saves print-ready maps to
`05_maps/` inside the run's results folder — one for the prediction and, for
Ordinary/Regression Kriging, one for the kriging variance. They are built with
QGIS's own print-layout engine (no matplotlib needed), and each map includes
the elements that cartographic standards and journal map guidelines expect:

| Element | What SAMKit draws |
|---|---|
| Title / subtitle | "Predicted <attribute>" (or your own title), model and validation strategy |
| Map body | the raster in a neatline, at a **rounded representative fraction** (1:25 000, 1:50 000 …) |
| Coordinate grid | labelled exterior ticks, user's choice: latitude/longitude in degrees-minutes-seconds ("39°02′31″N", "85°31′55″W"; a WGS 84 graticule drawn over the map, the default) or the map's own projected coordinates in metres ("4,323,000 m N"); labels that would run past a corner are left out |
| Colour ramp | perceptually uniform, colour-vision-deficiency-safe ramps: Viridis (default), Cividis, Magma, Plasma, Inferno, Greyscale |
| Legend | variable name **with units**, as a continuous colour bar with rounded tick values (vector SVG) or as classes with rounded breaks |
| Samples | calibration (open circles) and validation (filled triangles) points shown separately with their counts; k-fold runs show every sample point |
| Scale | graphic scale bar in m/km plus the numeric scale for the chosen page size |
| North arrow | true north (it rotates with meridian convergence on projected grids) |
| Locator inset | where the study area is, drawn from QGIS's bundled world map |
| Metadata | CRS name and EPSG code, projection, ellipsoid, cell size, mapped range, model, held-out R²/RMSE/MAE/CCC, sample size, covariates, author, date, software credit |

Options are in the **"Publication maps"** box under *Output*: title, units,
author/credit (remembered between sessions), colour ramp (and reverse),
continuous or classed legend, page size (A4, A3, US Letter; orientation is
picked automatically to give the largest map), resolution (300 dpi default,
the usual journal minimum; use 600 for line-art-heavy figures) and formats:

- **PNG** at the chosen dpi,
- **PDF** with vector text and linework, georeferenced,
- **SVG** (optional) for further editing in Inkscape/Illustrator.

If *Load output raster into QGIS* is ticked, the loaded prediction layer gets
the same styling, and (optionally) the layouts are added to **Project → Layout
Manager** so you can adjust any element by hand and re-export. Supporting
files (colour-bar SVG, sample-location GeoPackage) are kept in
`05_maps/layout_resources/` so those layouts keep working after the project is
saved and reopened. A map-rendering problem never fails the run: it is logged
and the raster and tables are still written.

The same function can be used on any single-band raster from the QGIS Python
console:

```python
from samkit import map_export
map_export.export_prediction_map(
    "/path/to/prediction.tif", "/path/to/out/soc_map",
    variable="SOC", units="g/kg", author="Your Name",
    formats=("png", "pdf"), dpi=300)
```

### Point-data scenarios (sample-size tests)

Under *Input data → Point-data scenario*, choose a percentage and random seed
and click **"Create scenario, save and load in QGIS"**. SAMKit draws that
percentage of the usable points (simple random sampling without replacement,
rounded to the nearest whole point, minimum one) from the full original layer,
saves it as a GeoPackage (all attributes, the CRS, and the original feature id
in a `source_fid` column) with a JSON description and a row in
`samkit_scenarios_index.csv`, loads it in the "SAMKit scenarios" group and —
if ticked — selects it as the input points. The same percentage and seed
always give the same subset.

**A model run always uses exactly the layer selected under "Georeferenced
points".** The percentage box on its own does nothing until you create a
scenario (earlier pre-release builds also drew a hidden subset at run time
whenever the box was below 100 %, which silently degraded maps). The results
window, the log and `00_RUN_SUMMARY.txt` state how many points the mapped model
was fitted with, and flag a scenario when one is used.

### When points find no covariate values

If most points get no covariate values (the run stops with "Fewer than 5 usable
points remain"), the log now prints a diagnosis: the point layer's CRS and
whether the points were reprojected, the point extent and each raster's extent
in the same units, how many points each covariate misses, and the most likely
cause. The usual one is a point layer whose coordinates are longitude/latitude
degrees while its CRS is set to a projected (metre) CRS, or the reverse -- QGIS
then sees no need to reproject, and every point lands outside the rasters. Fix
it under Layer Properties > Source > Assigned CRS.

### Covariate importance

For Random Forest — and for the Random Forest trend of Regression Kriging —
every run ranks the covariates by importance (mean decrease in impurity) in
the log, the run summary, the results window ("Covariate importance" tab) and
`01_accuracy/covariate_importance.csv`. Reading the importances does not
change the fitted model or the map.

### Data diagnostics and saved accuracy tables

- Every run logs **data diagnostics**: summary statistics for the target and
  each covariate at your sample locations, plus each covariate's correlation
  with the target. A covariate that comes back constant (std=0) almost
  always means the points and that raster don't actually overlap (check
  CRS/extent); if *none* of your covariates correlate with the target at
  all, a poor R² is telling you about your data/covariate choice, not about
  the model. The plugin also writes a `<output_name>_training_samples.csv`
  file next to your output raster containing the exact extracted table (x,
  y, target, each covariate) so you can inspect it directly in Excel or
  QGIS's attribute table.
- Every run's accuracy metrics are **saved to tables**: a per-run
  `<output_name>_metrics.csv`, and a cumulative
  `samkit_model_performance_log.csv` in the same folder that a new row is
  *appended* to on every run (timestamp, model, target field, validation
  method, sample size, hyperparameters, R², RMSE, MAE, bias, Lin's
  concordance correlation coefficient (CCC), and the regression-style R² of
  predicted-vs-observed) — so you can compare different
  models/settings/covariate choices over time in one table, using the same
  metric set commonly reported in DSM validation workflows (e.g. `goof()`
  -style R²/CCC/MSE/RMSE/bias functions).

  **Note on R²:** SAMKit's `R2` column is prediction-efficiency R²
  (1 − SSres/SStot, equivalent to Nash-Sutcliffe efficiency) — this is *not*
  the same quantity as the R² from a linear regression of predicted-on-observed
  values (the definition some R-based DSM workflows, e.g. `goof()`/`goof2()`
  -style functions, report as "R2"). That alternative definition is also
  reported here, as the `R2_regression` column, specifically so results can
  be compared directly against workflows using either convention.



- `samkit_plugin.py` — plugin registration (toolbar/menu entry)
- `main_dialog.py` — the three-tab dialog window (with scroll-area + screen-fit sizing)
- `dem_tab.py` — Tab 1 UI + logic (DEM download / UTM projection)
- `dem_download.py` — DEM sources (OpenTopography, AWS Terrain Tiles), UTM zone detection, warping and vertical-unit conversion (no QGIS imports, so it can be tested standalone)
- `terrain_tab.py` — Tab 2 UI + logic (Environmental Covariates)
- `dsm_tab.py` — Tab 3 UI + logic (Spatial Predictive Modeling: dependency panel, correlation analysis, k-fold/spatial-block validation, observed-vs-predicted reporting, metrics logging)
- `deps.py` — dependency detection + one-click pip installer
- `install_dependencies.py` — standalone fallback installer, runnable via the QGIS Python Console or a terminal
- `map_export.py` — publication-quality map layouts (title, legend, grid, scale bar, north arrow, locator inset, metadata) exported as PNG/PDF/SVG after every model run; also usable from the Python console
- `map_tools.py` — interactive map tools: rectangle drawing for the DEM area of interest, and free-form polygon drawing for the spatial-block calibration region
- `utils.py` — raster I/O, terrain-attribute math, point sampling, alignment
- `qgis_algorithms.py` — exact GDAL slope/aspect/TPI via QGIS Processing (Tab 1's "Exact GDAL" mode)
- `dsm_models.py` — model fitting/prediction backends (RF, SVM, Kriging, RK), k-fold utilities, and accuracy metrics
- `tests/test_terrain_analytics.py` — unit tests validating terrain covariates against analytical surfaces (run with `python3 test_terrain_analytics.py` or `pytest`)
- `tests/test_map_export.py` — renders example maps headlessly (projected and geographic grids, continuous and classed legends); skipped when QGIS isn't importable
- `tests/test_tab3_workflow.py` — end-to-end check of the Spatial Predictive Modeling tab (all models, validation designs, scenarios, outputs, maps); needs scikit-learn and pykrige, takes a few minutes
- `tests/test_reference_tools.py` — checks the built-in covariates against gdaldem, saga_cmd and whitebox_tools when those are installed
- `LICENSE`, `NOTICE` — GPL-2.0-or-later licence and third-party provenance

## License / citation

SAMKit is licensed under the GNU General Public License v2 or later (see
`LICENSE`); third-party provenance is listed in `NOTICE`. Please also cite the
reference tools whose algorithms you use (SAGA GIS - Conrad et al. 2015 and
Gallant & Dowling 2003 for MRVBF; GDAL; WhiteboxTools - Lindsay 2016; gstat -
Pebesma 2004; scikit-learn - Pedregosa et al. 2011).

If you use SAMKit in published research, please credit: Safaee et al., 2027 (paper details to be added)
— SAMKit (Spatial Analysis and Mapping Toolkit), https://github.com/ssafaee/SamKit

---

## Installing the optional Python packages (scikit-learn, pykrige)

**Short version:** open the *Spatial Predictive Modeling* tab, tick the packages
you need, click **Install checked packages**, wait, then **restart QGIS**.

### Where things get installed

SAMKit installs these packages into its own private folder:

    <your QGIS profile>/python/plugins/samkit/libs/

It does **not** install into QGIS's own Python. This is deliberate:

* No administrator rights are needed.
* Your QGIS installation — including its `numpy` and `scipy` — is never
  modified, so QGIS cannot be broken by installing SAMKit's dependencies.
* Uninstalling SAMKit removes the dependencies with it; nothing is left behind.

`libs/` is *appended* to `sys.path`, so QGIS's own packages always take
priority over anything in there.

### Why you must restart QGIS afterwards

`scikit-learn`, `pykrige`, `numpy` and `scipy` are compiled C extensions.
Python cannot safely load a newly installed compiled package into a session
that is already running — the process already holds a different build of
`numpy` in memory, and mixing the two segfaults, which closes QGIS instantly
with no error message.

So after a successful install SAMKit will show
*"installed — restart QGIS to use"* and refuse to run a model until you
restart. Restart once and everything works; nothing else needs doing.

### Do not install these the "normal" way

Avoid running this against QGIS's Python:

    pip install --upgrade scikit-learn     # ← don't

`--upgrade` also upgrades `numpy` and `scipy`, which QGIS's GDAL, processing
and matplotlib bindings are compiled against. On Windows the DLLs are locked
so the upgrade half-fails and leaves a broken install; elsewhere it succeeds
and breaks the ABI. Either way QGIS starts crashing. SAMKit's installer pins
`numpy`/`scipy`/`joblib`/`threadpoolctl` to the versions you are already
running so this cannot happen.

### If the in-app button fails

Use the standalone installer, which does exactly the same thing:

1. QGIS → **Plugins → Python Console** (Ctrl+Alt+P)
2. Show Editor → Open script → pick `install_dependencies.py` from the samkit
   plugin folder → Run
3. Restart QGIS

It prints the exact interpreter path and pip command, so you can also run it
by hand from an OSGeo4W Shell if you prefer. Behind a corporate proxy, add
`--proxy http://user:pass@host:port` to the pip command it prints.
