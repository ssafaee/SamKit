# -*- coding: utf-8 -*-
"""
qgis_algorithms.py
Exact-match terrain covariate computation via QGIS's own Processing
framework, calling GDAL directly (gdaldem under the hood) rather than
SAMKit's self-contained NumPy approximations.

Why this exists alongside utils.py's NumPy implementations
------------------------------------------------------------
SAMKit's default (utils.py) computes every Tab 1 covariate itself with
NumPy/GDAL raster I/O only, specifically so nothing in Tab 1 depends on
having SAGA GIS installed and its QGIS provider enabled -- a common source
of "provider not registered" failures. That default is a faithful,
standard implementation, but it is not guaranteed byte-identical to what
QGIS's own Processing Toolbox would produce for the same DEM.

GDAL (via `gdaldem`) ships as a core part of QGIS itself -- it is always
available, with no extra install -- and provides exact, canonical
implementations of slope, aspect, and TPI. Routing those three specific
covariates through QGIS's `gdal:slope` / `gdal:aspect` /
`gdal:tpitopographicpositionindex` Processing algorithms therefore gives
byte-for-byte the same result QGIS/GDAL would produce natively, with no new
dependency.

Curvature and TWI are deliberately NOT offered here: GDAL has no equivalent
algorithm for either, so an "exact" version of them is only available from
SAGA GIS, which would reintroduce the very provider-availability problem
this module mostly avoids. Those two always use SAMKit's built-in
implementation regardless of the computation method chosen in Tab 1.

MRVBF/MRRTF is the one exception: GDAL has no equivalent for this either,
but unlike curvature/TWI, SAMKit exposes an explicit, opt-in SAGA-backed
exact version of it (`compute_mrvbf_mrrtf_saga`) for users who have SAGA
GIS installed and want the genuine Gallant & Dowling (2003) reference
output rather than SAMKit's built-in approximation. This is a deliberate,
separate opt-in from the GDAL-backed slope/aspect/TPI methods above, since
it does reintroduce a SAGA-provider dependency for that one covariate.

Every function requires being run inside a live QGIS session (they import
`processing` at call time) and cannot be exercised outside QGIS.
"""

import os
import glob
import shutil
import subprocess
import tempfile



class ProcessingUnavailableError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# SAGA grid hand-over (fixes the pre-1.0 SAMKit cell-size bug)
# ---------------------------------------------------------------------------
# SAGA grids must have square cells. pre-1.0 SAMKit converted the DEM with
# gdal.Translate(format="SAGA"); for DEMs with even slightly non-square pixels
# GDAL cannot write the geotransform and silently writes CELLSIZE = 1 and origin
# (0, 0), so SAGA computed MRVBF/MRRTF on a 1 m grid (slopes ~cellsize-times too
# steep) and the result was then re-stamped with the DEM's georeferencing, hiding
# the error. SAMKit now writes the SAGA grid itself and verifies it.
SAGA_MAX_ANISOTROPY = 0.02   # up to 2 % x/y difference: keep the lattice, use mean cell size


def _prepare_saga_grid(dem_path, workdir, log=print):
    from osgeo import gdal
    import numpy as np
    gdal.UseExceptions()
    ds = gdal.Open(dem_path)
    gt = ds.GetGeoTransform()
    if gt[2] != 0 or gt[4] != 0:
        raise ProcessingUnavailableError("Rotated rasters are not supported by SAGA.")
    csx, csy = abs(gt[1]), abs(gt[5])
    info = {"dem_gt": gt, "dem_wkt": ds.GetProjection(), "rows": ds.RasterYSize,
            "cols": ds.RasterXSize, "resampled": False}
    src_ds = ds
    aniso = abs(csx - csy) / max(csx, csy)
    if aniso > SAGA_MAX_ANISOTROPY:
        cs = min(csx, csy)
        sq = os.path.join(workdir, "dem_square.tif")
        gdal.Warp(sq, dem_path, xRes=cs, yRes=cs, resampleAlg="bilinear", format="GTiff")
        src_ds = gdal.Open(sq)
        info.update(resampled=True, square_path=sq)
        log("  DEM cells are {:.3f} x {:.3f} (anisotropy {:.1%}); resampled to square {:.3f} "
            "cells for SAGA, results are warped back onto the DEM grid.".format(csx, csy, aniso, cs))
    else:
        cs = 0.5 * (csx + csy)
        if aniso > 0:
            log("  DEM cells are {:.4f} x {:.4f}; SAGA needs square cells - using the mean "
                "cell size {:.4f} on the same grid (distortion {:.2%}).".format(csx, csy, cs, aniso / 2))
    g2 = src_ds.GetGeoTransform()
    band = src_ds.GetRasterBand(1)
    arr = band.ReadAsArray().astype(np.float64)
    nd = band.GetNoDataValue()
    if nd is not None:
        arr[arr == nd] = np.nan
    arr[~np.isfinite(arr)] = np.nan
    rows, cols = arr.shape
    cs_txt = "%.10f" % cs
    x0 = g2[0] + 0.5 * abs(g2[1])
    y0 = g2[3] + rows * g2[5] + 0.5 * abs(g2[5])
    out = arr.astype(np.float32)
    out[np.isnan(out)] = -99999.0
    sgrd = os.path.join(workdir, "dem.sgrd")
    out[::-1].tofile(os.path.join(workdir, "dem.sdat"))
    with open(sgrd, "w") as f:
        f.write("NAME\t= dem\nDESCRIPTION\t=\nUNIT\t=\nDATAFILE_OFFSET\t= 0\n"
                "DATAFORMAT\t= FLOAT\nBYTEORDER_BIG\t= FALSE\n"
                "POSITION_XMIN\t= %.10f\nPOSITION_YMIN\t= %.10f\n"
                "CELLCOUNT_X\t= %d\nCELLCOUNT_Y\t= %d\nCELLSIZE\t= %s\n"
                "Z_FACTOR\t= 1.000000\nNODATA_VALUE\t= -99999.000000\nTOPTOBOTTOM\t= FALSE\n"
                % (x0, y0, cols, rows, cs_txt))
    info.update(sgrd=sgrd, cellsize=float(cs_txt), grid_rows=rows, grid_cols=cols,
                grid_gt=(g2[0], float(cs_txt), 0.0, g2[3], 0.0, -float(cs_txt)))
    # square-pixel GeoTIFF on the same lattice, for the QGIS SAGA provider route
    tif = os.path.join(workdir, "dem_for_saga.tif")
    drv = gdal.GetDriverByName("GTiff")
    o = drv.Create(tif, cols, rows, 1, gdal.GDT_Float32)
    o.SetGeoTransform(info["grid_gt"]); o.SetProjection(info["dem_wkt"])
    o.GetRasterBand(1).WriteArray(out); o.GetRasterBand(1).SetNoDataValue(-99999.0); o = None
    info["tif"] = tif
    return info


def _read_saga_grid(sgrd_path, expected_cellsize=None):
    import numpy as np
    hdr = {}
    for line in open(sgrd_path, encoding="utf-8", errors="replace"):
        if "=" in line:
            k, v = line.split("=", 1)
            hdr[k.strip().upper()] = v.strip()
    nx, ny = int(hdr["CELLCOUNT_X"]), int(hdr["CELLCOUNT_Y"])
    cs = float(hdr["CELLSIZE"])
    if expected_cellsize is not None and abs(cs - expected_cellsize) > 1e-6 * expected_cellsize:
        raise ProcessingUnavailableError(
            "SAGA output cell size {} does not match the input cell size {} - refusing to "
            "use a mis-scaled result.".format(cs, expected_cellsize))
    fmt = hdr.get("DATAFORMAT", "FLOAT").upper()
    dt = {"FLOAT": np.float32, "DOUBLE": np.float64}.get(fmt, np.float32)
    big = hdr.get("BYTEORDER_BIG", "FALSE").upper() == "TRUE"
    raw = np.fromfile(os.path.splitext(sgrd_path)[0] + ".sdat", dtype=np.dtype(dt).newbyteorder(">" if big else "<"),
                      offset=int(hdr.get("DATAFILE_OFFSET", "0")), count=nx * ny).reshape(ny, nx).astype(np.float64)
    if hdr.get("TOPTOBOTTOM", "FALSE").upper() != "TRUE":
        raw = raw[::-1]
    nd = [float(t) for t in hdr.get("NODATA_VALUE", "-99999").replace(",", ";").split(";") if t.strip()]
    lo, hi = min(nd), max(nd)          # SAGA 9 writes a nodata range "a;b"
    raw[(raw >= lo) & (raw <= hi)] = np.nan
    return raw


def _write_saga_result(arr, info, out_tif):
    """Write a SAGA result (on the SAGA grid) as a GeoTIFF on the DEM grid."""
    from osgeo import gdal
    import numpy as np
    drv = gdal.GetDriverByName("GTiff")
    a = arr.astype(np.float32).copy()
    a[np.isnan(a)] = -9999.0
    if not info["resampled"]:
        assert a.shape == (info["rows"], info["cols"])
        o = drv.Create(out_tif, info["cols"], info["rows"], 1, gdal.GDT_Float32, options=["COMPRESS=LZW", "TILED=YES"])
        o.SetGeoTransform(info["dem_gt"]); o.SetProjection(info["dem_wkt"])
        o.GetRasterBand(1).WriteArray(a); o.GetRasterBand(1).SetNoDataValue(-9999.0); o = None
        return out_tif
    tmp = out_tif + ".square.tif"
    o = drv.Create(tmp, a.shape[1], a.shape[0], 1, gdal.GDT_Float32)
    o.SetGeoTransform(info["grid_gt"]); o.SetProjection(info["dem_wkt"])
    o.GetRasterBand(1).WriteArray(a); o.GetRasterBand(1).SetNoDataValue(-9999.0); o = None
    gt = info["dem_gt"]
    gdal.Warp(out_tif, tmp, outputBounds=(gt[0], gt[3] + info["rows"] * gt[5], gt[0] + info["cols"] * gt[1], gt[3]),
              width=info["cols"], height=info["rows"], dstSRS=info["dem_wkt"], resampleAlg="bilinear",
              dstNodata=-9999.0, creationOptions=["COMPRESS=LZW", "TILED=YES"])
    os.remove(tmp)
    return out_tif


def _run_gdal_algorithm(alg_id, params, log=print):
    try:
        from qgis import processing
    except ImportError:
        raise ProcessingUnavailableError(
            "The QGIS 'processing' module is not available. Exact GDAL mode only "
            "works when SAMKit is run inside QGIS itself."
        )
    log(f"Running QGIS Processing algorithm '{alg_id}'...")
    try:
        result = processing.run(alg_id, params)
    except Exception as e:
        raise ProcessingUnavailableError(
            f"QGIS Processing algorithm '{alg_id}' failed: {e}\n"
            f"This usually means the GDAL provider isn't available in this QGIS "
            f"install (rare, since GDAL ships with QGIS itself) -- check "
            f"Processing > Toolbox for a 'GDAL' provider entry, or fall back to "
            f"SAMKit's built-in computation method instead."
        )
    return result


def compute_slope_gdal(dem_path, output_path, z_factor=1.0, as_percent=False,
                        compute_edges=False, zevenbergen=False, log=print):
    """
    Exact GDAL slope via QGIS's 'gdal:slope' Processing algorithm (gdaldem slope).
    zevenbergen=False (GDAL/QGIS default) uses Horn's (1981) method; True uses
    Zevenbergen & Thorne (1987) instead.
    """
    params = {
        "INPUT": dem_path,
        "BAND": 1,
        "SCALE": z_factor,
        "AS_PERCENT": as_percent,
        "COMPUTE_EDGES": compute_edges,
        "ZEVENBERGEN": zevenbergen,
        "OPTIONS": "",
        "EXTRA": "",
        "OUTPUT": output_path,
    }
    result = _run_gdal_algorithm("gdal:slope", params, log=log)
    return result["OUTPUT"]


def compute_aspect_gdal(dem_path, output_path, compute_edges=False,
                         zevenbergen=False, trig_angle=False, zero_flat=True, log=print):
    """Exact GDAL aspect via QGIS's 'gdal:aspect' Processing algorithm (gdaldem aspect)."""
    params = {
        "INPUT": dem_path,
        "BAND": 1,
        "TRIG_ANGLE": trig_angle,
        "ZERO_FLAT": zero_flat,
        "COMPUTE_EDGES": compute_edges,
        "ZEVENBERGEN": zevenbergen,
        "OPTIONS": "",
        "EXTRA": "",
        "OUTPUT": output_path,
    }
    result = _run_gdal_algorithm("gdal:aspect", params, log=log)
    return result["OUTPUT"]


def compute_tpi_gdal(dem_path, output_path, compute_edges=False, log=print):
    """
    Exact GDAL TPI via QGIS's 'gdal:tpitopographicpositionindex' Processing
    algorithm (gdaldem TPI). Note: GDAL's TPI implementation always uses a
    fixed 3x3 (immediate 8-neighbour) window -- unlike SAMKit's built-in TPI,
    the neighbourhood radius is not adjustable in this mode.
    """
    params = {
        "INPUT": dem_path,
        "BAND": 1,
        "COMPUTE_EDGES": compute_edges,
        "OPTIONS": "",
        "EXTRA": "",
        "OUTPUT": output_path,
    }
    result = _run_gdal_algorithm("gdal:tpitopographicpositionindex", params, log=log)
    return result["OUTPUT"]


# The one algorithm-name suffix shared by every SAGA provider variant.
_SAGA_MRVBF_SUFFIX = "multiresolutionindexofvalleybottomflatnessmrvbf"


def _resolve_saga_mrvbf_algorithm_id():
    """
    Return the concrete QGIS Processing algorithm id for SAGA's MRVBF/MRRTF
    tool in THIS QGIS install, or None if no SAGA provider exposes it.

    The provider prefix is not stable across QGIS versions ('saga:' pre-3.30,
    'sagang:' via the NextGen plugin), so we look the algorithm up in the live
    Processing registry by its stable suffix rather than hardcoding a prefix.
    """
    try:
        from qgis.core import QgsApplication
    except ImportError:
        return None
    registry = QgsApplication.processingRegistry()
    for prefix in ("sagang", "saga"):
        alg_id = "{}:{}".format(prefix, _SAGA_MRVBF_SUFFIX)
        if registry.algorithmById(alg_id) is not None:
            return alg_id
    for alg in registry.algorithms():
        if alg.id().split(":")[-1] == _SAGA_MRVBF_SUFFIX:
            return alg.id()
    return None


def _find_saga_cmd(explicit=None, log=print):
    """
    Locate the SAGA command-line executable (saga_cmd[.exe]) WITHOUT relying on
    any QGIS SAGA provider plugin. Checks, in order: an explicit path/folder
    passed in, the SAGA_CMD/SAGA_PATH/SAGA env vars, the system PATH, the SAGA
    that ships alongside the running QGIS, and common install locations.
    Returns the first existing executable, or None.
    """
    exe = "saga_cmd.exe" if os.name == "nt" else "saga_cmd"
    candidates = []

    def add(p):
        if p:
            candidates.append(p)

    if explicit:
        e = os.path.expanduser(str(explicit).strip().strip('"'))
        if e:
            add(os.path.join(e, exe) if os.path.isdir(e) else e)

    for var in ("SAGA_CMD", "SAGA_PATH", "SAGA"):
        val = os.environ.get(var)
        if val:
            add(val if os.path.isfile(val) else os.path.join(val, exe))

    add(shutil.which(exe))
    add(shutil.which("saga_cmd"))

    # SAGA bundled next to the running QGIS: <prefix>/apps/qgis -> <prefix>/apps/saga
    try:
        from qgis.core import QgsApplication
        prefix = QgsApplication.prefixPath()
        if prefix:
            apps = os.path.dirname(prefix)
            for sub in ("saga", "saga-ltr", "saga9"):
                add(os.path.join(apps, sub, exe))
    except Exception:
        pass

    if os.name == "nt":
        for pat in (
            r"C:\Program Files\SAGA-GIS\saga_cmd.exe",
            r"C:\Program Files\SAGA\saga_cmd.exe",
            r"C:\Program Files (x86)\SAGA-GIS\saga_cmd.exe",
            r"C:\OSGeo4W\apps\saga\saga_cmd.exe",
            r"C:\OSGeo4W64\apps\saga\saga_cmd.exe",
            r"C:\saga*\saga_cmd.exe",
            r"C:\saga*\*\saga_cmd.exe",
            r"C:\Program Files\QGIS*\apps\saga\saga_cmd.exe",
        ):
            candidates.extend(glob.glob(pat))

    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return None


def _conform_output_to_dem(output_path, dem_path, log=print):
    """
    Force a SAGA-produced raster to carry exactly the same CRS and geotransform
    as the source DEM.

    SAGA's saga_cmd / .sdat round-trip (and, on some builds, the QGIS SAGA
    provider) routinely drops the coordinate reference system and can shift the
    grid origin. When that happens the MRVBF/MRRTF GeoTIFF loads with no
    projection, mislocated and wrongly scaled relative to the other covariates
    and the sample points -- even though every other SAMKit covariate is
    georeferenced correctly, because those are written through
    utils.write_raster(), which stamps the DEM's geotransform + CRS onto them.

    SAGA computes MRVBF/MRRTF on exactly the grid it was handed (the DEM), so the
    output has the same number of rows/cols as the DEM; copying the DEM's
    geotransform and projection therefore restores pixel-perfect alignment with
    the DEM, the other covariates and the points. If the sizes unexpectedly
    differ we only fill in a missing CRS and warn, rather than silently forcing a
    geotransform that might not apply.
    """
    try:
        from osgeo import gdal
        gdal.UseExceptions()
    except ImportError:
        log("  (GDAL Python unavailable; cannot verify/repair the CRS of {})"
            .format(os.path.basename(output_path)))
        return

    ref = gdal.Open(dem_path)
    if ref is None:
        log("  (could not reopen the DEM to copy its CRS onto {})"
            .format(os.path.basename(output_path)))
        return
    ref_gt = ref.GetGeoTransform()
    ref_proj = ref.GetProjection()
    ref_x, ref_y = ref.RasterXSize, ref.RasterYSize
    ref = None

    ds = gdal.Open(output_path, gdal.GA_Update)
    if ds is None:
        log("  (could not open {} to set its georeferencing)"
            .format(os.path.basename(output_path)))
        return
    out_x, out_y = ds.RasterXSize, ds.RasterYSize
    had_proj = bool(ds.GetProjection())
    same_grid = (out_x == ref_x and out_y == ref_y)

    if same_grid:
        ds.SetGeoTransform(ref_gt)
        ds.SetProjection(ref_proj)
        if had_proj:
            log("  aligned {} to the DEM's grid and CRS."
                .format(os.path.basename(output_path)))
        else:
            log("  repaired {}: SAGA output had no projection; assigned the "
                "DEM's CRS and geotransform so it lines up with the other "
                "covariates and the points."
                .format(os.path.basename(output_path)))
    else:
        if not had_proj and ref_proj:
            ds.SetProjection(ref_proj)
        log("  WARNING: {} is {}x{} but the DEM is {}x{}; SAGA did not return "
            "the input grid. Set the CRS but left the geotransform SAGA wrote "
            "-- check this layer."
            .format(os.path.basename(output_path), out_x, out_y, ref_x, ref_y))
    ds.FlushCache()
    ds = None


def compute_mrvbf_mrrtf_saga_provider(alg_id, dem_path, mrvbf_output_path,
                                       mrrtf_output_path,
                                       t_slope=16.0, t_pctl_v=0.4, t_pctl_r=0.35,
                                       p_slope=4.0, p_pctl=3.0, max_res=100.0,
                                       update=False, classify=False, log=print):
    """MRVBF/MRRTF via a registered QGIS SAGA Processing provider (alg_id)."""
    from qgis import processing  # provider route only exists inside QGIS
    workdir = tempfile.mkdtemp(prefix="samkit_sagaprov_")
    info = _prepare_saga_grid(dem_path, workdir, log=log)
    params = {
        "DEM": info["tif"],
        "T_SLOPE": t_slope, "T_PCTL_V": t_pctl_v, "T_PCTL_R": t_pctl_r,
        "P_SLOPE": p_slope, "P_PCTL": p_pctl,
        "UPDATE": update, "CLASSIFY": classify, "MAX_RES": max_res,
        "MRVBF": mrvbf_output_path, "MRRTF": mrrtf_output_path,
    }
    log("Running QGIS Processing algorithm '{}'...".format(alg_id))
    try:
        result = processing.run(alg_id, params)
    except Exception as e:
        raise ProcessingUnavailableError(
            "SAGA provider algorithm '{}' failed: {}".format(alg_id, e)
        )
    from osgeo import gdal
    import numpy as np
    outs = []
    for produced, target in ((result["MRVBF"], mrvbf_output_path), (result["MRRTF"], mrrtf_output_path)):
        ds = gdal.Open(str(produced))
        gt = ds.GetGeoTransform()
        if (ds.RasterXSize, ds.RasterYSize) != (info["grid_cols"], info["grid_rows"]) or \
                abs(abs(gt[1]) - info["cellsize"]) > 1e-6 * info["cellsize"]:
            raise ProcessingUnavailableError("SAGA provider returned a grid that does not match "
                                             "the input grid - refusing a mis-scaled result.")
        b = ds.GetRasterBand(1); a = b.ReadAsArray().astype(np.float64); nd = b.GetNoDataValue()
        if nd is not None:
            a[a == nd] = np.nan
        outs.append(_write_saga_result(a, info, target))
    shutil.rmtree(workdir, ignore_errors=True)
    return tuple(outs)


def compute_mrvbf_mrrtf_saga_cmd(dem_path, mrvbf_output_path, mrrtf_output_path,
                                 saga_cmd_path=None,
                                 t_slope=16.0, t_pctl_v=0.4, t_pctl_r=0.35,
                                 p_slope=4.0, p_pctl=3.0, max_res=100.0,
                                 update=False, classify=False, log=print):
    """
    Version-independent MRVBF/MRRTF: call SAGA's own command-line tool
    ('saga_cmd ta_morphometry 8') directly, bypassing the QGIS SAGA provider
    plugin entirely.

    Why this is not version-specific: the ta_morphometry MRVBF tool (tool 8)
    and its parameter names -- DEM, MRVBF, MRRTF, T_SLOPE, T_PCTL_V, T_PCTL_R,
    P_SLOPE, P_PCTL, UPDATE, CLASSIFY, MAX_RES -- have been stable across SAGA
    2.x through 9.x, so this runs against whatever SAGA build is installed
    (including the SAGA that ships with OSGeo4W). There is no provider-plugin
    dependency and no "SAGA >= 9.1" requirement. All raster format conversion
    is done here with GDAL, so it also does not depend on any particular SAGA
    build's GDAL import/export tools.
    """
    saga_cmd = _find_saga_cmd(saga_cmd_path, log=log)
    if saga_cmd is None:
        raise ProcessingUnavailableError(
            "Could not find SAGA's command-line program (saga_cmd). Install SAGA "
            "GIS (e.g. from the SAGA-GIS project on SourceForge, or via OSGeo4W) "
            "and either add its folder to PATH, set the SAGA_CMD environment "
            "variable, or enter the SAGA folder in SAMKit's SAGA path field."
        )
    log("Using saga_cmd: {}".format(saga_cmd))

    def _num(v):
        return str(float(v))

    workdir = tempfile.mkdtemp(prefix="samkit_saga_")
    try:
        # Hand SAGA a native grid written by SAMKit with the correct (square) cell
        # size - never gdal.Translate(format="SAGA"), which writes CELLSIZE=1 for
        # non-square pixels (the pre-1.0 SAMKit bug).
        info = _prepare_saga_grid(dem_path, workdir, log=log)
        dem_for_saga = info["sgrd"]

        mrvbf_sgrd = os.path.join(workdir, "mrvbf.sgrd")
        mrrtf_sgrd = os.path.join(workdir, "mrrtf.sgrd")
        cmd = [
            saga_cmd, "ta_morphometry", "8",
            "-DEM", dem_for_saga,
            "-MRVBF", mrvbf_sgrd,
            "-MRRTF", mrrtf_sgrd,
            "-T_SLOPE", _num(t_slope),
            "-T_PCTL_V", _num(t_pctl_v),
            "-T_PCTL_R", _num(t_pctl_r),
            "-P_SLOPE", _num(p_slope),
            "-P_PCTL", _num(p_pctl),
            "-MAX_RES", _num(max_res),
            "-UPDATE", "1" if update else "0",
            "-CLASSIFY", "1" if classify else "0",
        ]
        log("Running SAGA directly: {}".format(" ".join(cmd)))

        run_kwargs = dict(
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=os.path.dirname(saga_cmd) or None,
            env=os.environ.copy(),
        )
        # decode text without blowing up on Windows codepage bytes
        try:
            run_kwargs.update(text=True, encoding="utf-8", errors="replace")
        except TypeError:
            pass
        if os.name == "nt":
            run_kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW

        proc = subprocess.run(cmd, **run_kwargs)
        out = proc.stdout or ""
        for line in out.splitlines()[-20:]:
            if line.strip():
                log("  saga: " + line)
        if proc.returncode != 0:
            raise ProcessingUnavailableError(
                "saga_cmd exited with code {} running ta_morphometry 8. See the "
                "SAGA output logged above for the reason.".format(proc.returncode)
            )

        for sgrd, out_tif in ((mrvbf_sgrd, mrvbf_output_path), (mrrtf_sgrd, mrrtf_output_path)):
            if not os.path.isfile(os.path.splitext(sgrd)[0] + ".sdat"):
                raise ProcessingUnavailableError(
                    "SAGA reported success but produced no output grid for {}".format(os.path.basename(out_tif)))
            arr = _read_saga_grid(sgrd, expected_cellsize=info["cellsize"])
            _write_saga_result(arr, info, out_tif)
            log("  wrote {} on the DEM grid (SAGA cell size verified: {:.4f}).".format(
                os.path.basename(out_tif), info["cellsize"]))
        return mrvbf_output_path, mrrtf_output_path
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def compute_mrvbf_mrrtf_saga(dem_path, mrvbf_output_path, mrrtf_output_path,
                              saga_cmd_path=None,
                              t_slope=16.0, t_pctl_v=0.4, t_pctl_r=0.35,
                              p_slope=4.0, p_pctl=3.0, max_res=100.0,
                              update=False, classify=False, log=print):
    """
    Compute exact (Gallant & Dowling 2003) MRVBF/MRRTF with SAGA, choosing
    whatever route works in this install so the result does not depend on a
    specific SAGA / plugin version:

      1. A registered QGIS SAGA Processing provider ('saga:' or 'sagang:'),
         if one is present; otherwise
      2. SAGA's command-line tool (saga_cmd) called directly -- no provider
         plugin, works with any installed SAGA version.

    Raises ProcessingUnavailableError only if neither route is available, in
    which case unticking 'Use SAGA for exact MRVBF/MRRTF' falls back to
    SAMKit's built-in (SAGA-free) approximation.
    """
    kw = dict(t_slope=t_slope, t_pctl_v=t_pctl_v, t_pctl_r=t_pctl_r,
              p_slope=p_slope, p_pctl=p_pctl, max_res=max_res,
              update=update, classify=classify, log=log)
    errors = []

    provider_id = _resolve_saga_mrvbf_algorithm_id()
    if provider_id is not None:
        try:
            return compute_mrvbf_mrrtf_saga_provider(
                provider_id, dem_path, mrvbf_output_path, mrrtf_output_path, **kw)
        except ProcessingUnavailableError as e:
            errors.append("QGIS SAGA provider: {}".format(e))
            log("  QGIS SAGA provider route failed; trying saga_cmd directly...")
    else:
        log("  No QGIS SAGA provider registered; using saga_cmd directly "
            "(version-independent)...")

    try:
        return compute_mrvbf_mrrtf_saga_cmd(
            dem_path, mrvbf_output_path, mrrtf_output_path,
            saga_cmd_path=saga_cmd_path, **kw)
    except ProcessingUnavailableError as e:
        errors.append("saga_cmd: {}".format(e))

    raise ProcessingUnavailableError(
        "Couldn't compute exact MRVBF/MRRTF with SAGA by any available method:\n"
        + "\n".join("  - " + m for m in errors)
        + "\n\nEasiest fix: untick 'Use SAGA for exact MRVBF/MRRTF' to use "
          "SAMKit's built-in approximation (no SAGA install required). To use "
          "SAGA, install any SAGA GIS build and point SAMKit's SAGA path field "
          "at its folder -- saga_cmd works with any SAGA version, unlike the "
          "QGIS SAGA provider plugin."
    )


# ---------------------------------------------------------------------------
# Optional exact WhiteboxTools route for TWI (no QGIS plugin needed)
# ---------------------------------------------------------------------------

def _find_whitebox_tools(explicit=None):
    exe = "whitebox_tools.exe" if os.name == "nt" else "whitebox_tools"
    cands = []
    if explicit:
        e = os.path.expanduser(str(explicit).strip().strip('"'))
        cands.append(os.path.join(e, exe) if os.path.isdir(e) else e)
    for var in ("WBT_PATH", "WHITEBOX_TOOLS", "WHITEBOXTOOLS"):
        v = os.environ.get(var)
        if v:
            cands.append(v if os.path.isfile(v) else os.path.join(v, exe))
    cands.append(shutil.which(exe))
    try:
        import whitebox  # the MIT-licensed 'whitebox' Python package bundles the binary
        cands.append(os.path.join(os.path.dirname(whitebox.__file__), "WBT", exe))
    except Exception:
        pass
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return None


def compute_twi_whitebox(dem_path, output_path, wbt_path=None, log=print):
    """
    TWI exactly as WhiteboxTools computes it: FillDepressions --fix_flats ->
    D8FlowAccumulation --out_type=sca -> Slope (degrees) -> WetnessIndex.
    Requires a WhiteboxTools binary (Lindsay 2016; MIT licence).
    """
    exe = _find_whitebox_tools(wbt_path)
    if exe is None:
        raise ProcessingUnavailableError(
            "WhiteboxTools (whitebox_tools) was not found. Install it (e.g. 'pip install whitebox' "
            "or the WhiteboxTools for QGIS plugin) or set WBT_PATH, or untick the WhiteboxTools option.")
    wd = tempfile.mkdtemp(prefix="samkit_wbt_")
    try:
        steps = [
            ["-r=FillDepressions", "--dem=" + dem_path, "-o=" + os.path.join(wd, "filled.tif"), "--fix_flats"],
            ["-r=D8FlowAccumulation", "-i=" + os.path.join(wd, "filled.tif"), "-o=" + os.path.join(wd, "sca.tif"),
             "--out_type=sca"],
            ["-r=Slope", "--dem=" + dem_path, "-o=" + os.path.join(wd, "slope.tif")],
            ["-r=WetnessIndex", "--sca=" + os.path.join(wd, "sca.tif"), "--slope=" + os.path.join(wd, "slope.tif"),
             "-o=" + output_path],
        ]
        for st in steps:
            # WhiteboxTools 2.x FillDepressions has an intermittent thread race
            # ("Error unwrapping 'output'"); it is harmless to simply re-run it.
            cmd = [exe] + st + ["--max_procs=1"]
            log("  whitebox_tools " + " ".join(st))
            import time
            for attempt in range(8):
                p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   errors="replace")
                if p.returncode == 0:
                    break
                log("    (WhiteboxTools failed, retrying {}/7)".format(attempt + 1))
                time.sleep(0.3)
            if p.returncode != 0:
                raise ProcessingUnavailableError("WhiteboxTools failed:\n" + p.stdout[-1500:])
        return output_path
    finally:
        shutil.rmtree(wd, ignore_errors=True)
