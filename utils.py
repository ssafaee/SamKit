# -*- coding: utf-8 -*-
"""
utils.py
Self-contained NumPy/GDAL helpers used by the SAMKit tabs.

Every terrain covariate is computed here without requiring SAGA/Whitebox, and
(since SAMKit 1.0.0) each one reproduces an established reference tool:

  - Slope / aspect : Horn (1981); edge_mode 'gdal' == gdaldem -compute_edges,
                     'whitebox' == WhiteboxTools, 'saga' == SAGA method 2.
  - Curvature      : Zevenbergen & Thorne (1987); convention 'saga' == SAGA
                     ta_morphometry 0 (C_GENE/C_PROF/C_PLAN/C_TANG), 'samkit'
                     (Esri-style x100), 'florinsky' (exact geometric).
  - TPI            : Weiss (2001); nodata-aware window. square == Whitebox
                     DiffFromMeanElev (compat='whitebox' is bit-identical),
                     circle == SAGA TPI, 3x3 without centre == gdaldem TPI.
  - TWI            : Beven & Kirkby (1979); Priority-Flood filling (Barnes et al.
                     2014) with WhiteboxTools-style flat fixing and D8/SCA definitions.
  - MRVBF / MRRTF  : Gallant & Dowling (2003) - line-by-line port of SAGA GIS
                     ta_morphometry 8 (GPL-2.0-or-later, see NOTICE),
                     bit-identical to SAGA 9.3.1.

All functions operate on 2-D numpy float arrays with NaN as nodata.
"""

import numpy as np

try:
    from osgeo import gdal, osr
    gdal.UseExceptions()
    HAS_GDAL = True
except Exception:
    HAS_GDAL = False

try:
    from scipy.ndimage import uniform_filter, minimum_filter
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Raster I/O
# ---------------------------------------------------------------------------

class RasterInfo:
    """Lightweight container for a raster's array + georeferencing."""

    def __init__(self, array, geotransform, projection_wkt, nodata, path=None):
        self.array = array  # float64 2D array, nodata replaced with np.nan
        self.geotransform = geotransform
        self.projection_wkt = projection_wkt
        self.nodata = nodata
        self.path = path

    @property
    def shape(self):
        return self.array.shape

    @property
    def cellsize_x(self):
        return abs(self.geotransform[1])

    @property
    def cellsize_y(self):
        return abs(self.geotransform[5])

    def pixel_to_xy(self, row, col):
        gt = self.geotransform
        x = gt[0] + (col + 0.5) * gt[1] + (row + 0.5) * gt[2]
        y = gt[3] + (col + 0.5) * gt[4] + (row + 0.5) * gt[5]
        return x, y

    def xy_to_pixel(self, x, y):
        gt = self.geotransform
        # invert affine transform (assumes no rotation, standard north-up)
        col = (x - gt[0]) / gt[1]
        row = (y - gt[3]) / gt[5]
        return row, col


def read_raster(path, band=1):
    if not HAS_GDAL:
        raise RuntimeError("GDAL (osgeo) is not available in this Python environment.")
    ds = gdal.Open(path)
    if ds is None:
        raise RuntimeError(f"Could not open raster: {path}")
    bnd = ds.GetRasterBand(band)
    nodata = bnd.GetNoDataValue()
    arr = bnd.ReadAsArray().astype(np.float64)
    if nodata is not None:
        arr[arr == nodata] = np.nan
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    ds = None
    return RasterInfo(arr, gt, proj, nodata, path=path)


def write_raster(path, array, ref: RasterInfo, nodata=-9999.0, dtype="float32"):
    if not HAS_GDAL:
        raise RuntimeError("GDAL (osgeo) is not available in this Python environment.")
    rows, cols = array.shape
    gdal_dtype = gdal.GDT_Float32 if dtype == "float32" else gdal.GDT_Float64
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(path, cols, rows, 1, gdal_dtype, options=["COMPRESS=LZW", "TILED=YES"])
    out_ds.SetGeoTransform(ref.geotransform)
    out_ds.SetProjection(ref.projection_wkt)
    out_arr = array.copy()
    out_arr[np.isnan(out_arr)] = nodata
    band = out_ds.GetRasterBand(1)
    band.WriteArray(out_arr)
    band.SetNoDataValue(nodata)
    band.FlushCache()
    out_ds = None
    return path


def align_raster_to_reference(src_path, ref_path, out_path, resample_alg="bilinear"):
    """Resample/reproject src_path onto the exact grid (extent, resolution, CRS) of ref_path."""
    if not HAS_GDAL:
        raise RuntimeError("GDAL (osgeo) is not available in this Python environment.")
    ref_ds = gdal.Open(ref_path)
    gt = ref_ds.GetGeoTransform()
    proj = ref_ds.GetProjection()
    cols = ref_ds.RasterXSize
    rows = ref_ds.RasterYSize
    xmin = gt[0]
    ymax = gt[3]
    xres = gt[1]
    yres = gt[5]
    xmax = xmin + cols * xres
    ymin = ymax + rows * yres

    resample_map = {
        "bilinear": gdal.GRA_Bilinear,
        "nearest": gdal.GRA_NearestNeighbour,
        "cubic": gdal.GRA_Cubic,
    }
    warp_opts = gdal.WarpOptions(
        format="GTiff",
        outputBounds=(xmin, min(ymin, ymax), xmax, max(ymin, ymax)),
        xRes=abs(xres),
        yRes=abs(yres),
        dstSRS=proj,
        resampleAlg=resample_map.get(resample_alg, gdal.GRA_Bilinear),
        width=cols,
        height=rows,
        creationOptions=["COMPRESS=LZW"],
    )
    gdal.Warp(out_path, src_path, options=warp_opts)
    ref_ds = None
    return out_path


# ---------------------------------------------------------------------------
# 3x3 neighbourhoods with the edge / nodata rules of the reference tools
# ---------------------------------------------------------------------------
#
# Every 3x3 operator (slope, aspect, curvature) needs a value for neighbours that
# fall outside the raster or on nodata. The established tools differ here, which
# is the ONLY reason their outputs differ at edges:
#   'gdal'     gdaldem -compute_edges: nodata neighbour -> centre value; outside
#              the first/last row -> linear extrapolation 2*a - b; outside the
#              first/last column -> edge column repeated (gdaldem_lib.cpp).
#   'whitebox' WhiteboxTools: any missing neighbour -> centre value.
#   'saga'     SAGA ta_morphometry: missing neighbour -> mirrored opposite
#              neighbour (2*z - z_opposite), or the centre if both are missing.
#   'legacy'   pre-1.0 SAMKit: edge replication, nodata propagates as NaN.
EDGE_MODES = ("gdal", "whitebox", "saga", "legacy")


def _window3x3_gdal(z):
    """gdaldem -compute_edges neighbourhood (see gdaldem_lib.cpp): middle rows
    extrapolate a missing column as 2*a - b; the first/last row extrapolates the
    missing row the same way but repeats edge columns; nodata -> centre value."""
    rows, cols = z.shape
    if cols > 1:
        ce = np.hstack([(2.0 * z[:, :1] - z[:, 1:2]), z, (2.0 * z[:, -1:] - z[:, -2:-1])])
    else:
        ce = np.pad(z, ((0, 0), (1, 1)), mode="edge")
    rep = np.pad(z, ((0, 0), (1, 1)), mode="edge")
    p = np.pad(ce, ((1, 1), (0, 0)), mode="constant", constant_values=np.nan)
    w = [p[r:r + rows, c:c + cols].copy() for r in range(3) for c in range(3)]
    if rows > 1:
        top = 2.0 * rep[0] - rep[1]
        bot = 2.0 * rep[-1] - rep[-2]
        first = (top, rep[0], rep[1])          # window lines for output row 0
        last = (rep[-2], rep[-1], bot)         # window lines for output row n-1
        for k in range(9):
            dr, dc = divmod(k, 3)
            w[k][0] = first[dr][dc:dc + cols]
            w[k][-1] = last[dr][dc:dc + cols]
    c = w[4]
    return [np.where(np.isnan(v), c, v) for v in w]


def _window3x3(dem, edge_mode="gdal"):
    """Return [NW, N, NE, W, C, E, SW, S, SE] 2-D arrays for every cell."""
    if edge_mode not in EDGE_MODES:
        raise ValueError(f"edge_mode must be one of {EDGE_MODES}")
    z = np.asarray(dem, dtype=np.float64)
    rows, cols = z.shape
    if edge_mode == "legacy":
        p = np.pad(z, 1, mode="edge")
    elif edge_mode == "gdal":
        return _window3x3_gdal(z)
    else:
        p = np.pad(z, 1, mode="constant", constant_values=np.nan)
    w = [p[r:r + rows, c:c + cols] for r in range(3) for c in range(3)]
    c = w[4]
    if edge_mode in ("gdal", "whitebox"):
        w = [np.where(np.isnan(v), c, v) for v in w]
    elif edge_mode == "saga":
        out = list(w)
        for k in range(9):
            if k != 4:
                opp = w[8 - k]
                out[k] = np.where(np.isnan(w[k]), np.where(np.isnan(opp), c, 2.0 * c - opp), w[k])
        w = out
    return w


# ---------------------------------------------------------------------------
# Slope / aspect (Horn 1981, identical to gdaldem / WhiteboxTools / SAGA method 2)
# ---------------------------------------------------------------------------

def compute_slope_aspect(dem, cellsize_x, cellsize_y, edge_mode="gdal"):
    """
    Horn (1981) slope and aspect. Returns (slope_deg, aspect_deg, slope_rad, dzdx, dzdy).

    Slope uses the x and y cell sizes separately (as gdaldem does). Aspect follows
    gdaldem/WhiteboxTools/SAGA: it is computed from the unscaled Horn differences,
    azimuth 0 = north, clockwise, and flat cells (no gradient) are NaN.
    edge_mode='gdal' reproduces `gdaldem slope/aspect -compute_edges`, 'whitebox'
    WhiteboxTools Slope/Aspect (x resolution used for both axes), 'saga' SAGA
    ta_morphometry 0 method 2 (square cells).
    """
    if edge_mode == "whitebox":
        cellsize_y = cellsize_x   # WhiteboxTools Slope uses the x resolution for both axes
    z1, z2, z3, z4, _z5, z6, z7, z8, z9 = _window3x3(dem, edge_mode)
    gx = (z3 + 2.0 * z6 + z9) - (z1 + 2.0 * z4 + z7)
    gy = (z7 + 2.0 * z8 + z9) - (z1 + 2.0 * z2 + z3)
    dzdx = gx / (8.0 * cellsize_x)
    dzdy = gy / (8.0 * cellsize_y)
    slope_rad = np.arctan(np.hypot(dzdx, dzdy))
    slope_deg = np.degrees(slope_rad)
    a = np.degrees(np.arctan2(gy, -gx))
    aspect_deg = np.where(a > 90.0, 450.0 - a, 90.0 - a)
    aspect_deg = np.where(aspect_deg >= 360.0, aspect_deg - 360.0, aspect_deg)
    aspect_deg = np.where((gx == 0) & (gy == 0), np.nan, aspect_deg)
    nan_mask = np.isnan(np.asarray(dem, dtype=float))
    for arr in (slope_deg, aspect_deg, slope_rad):
        arr[nan_mask] = np.nan
    return slope_deg, aspect_deg, slope_rad, dzdx, dzdy


# ---------------------------------------------------------------------------
# Curvature
# ---------------------------------------------------------------------------
CURVATURE_CONVENTIONS = ("samkit", "saga", "florinsky")


def compute_curvature(dem, cellsize_x, cellsize_y, edge_mode=None, convention="samkit"):
    """
    Curvature from the Zevenbergen & Thorne (1987) 9-term polynomial.

    convention='samkit'    (default, unchanged meaning) Esri-style x100 units:
                           general = -2(D+E)*100, profile = -2(DG^2+EH^2+FGH)/(G^2+H^2)*100,
                           plan = 2(DH^2+EG^2-FGH)/(G^2+H^2)*100 (plan > 0 = concave).
    convention='saga'      identical to SAGA ta_morphometry 0 (method 6) outputs C_GENE,
                           C_PROF, C_PLAN, C_TANG in 1/m (formulas of SAGA's
                           CMorphometry::Set_From_Polynom), SAGA edge rule by default.
    convention='florinsky' exact geometric definitions (Florinsky 2016), 1/m, convex > 0:
                           general = -(zxx+zyy), profile = vertical curvature kv,
                           plan = horizontal (tangential) curvature kh.
    Separate x/y cell sizes are used in every case.
    Returns a dict with 'general', 'profile', 'plan' (+ 'tangential' for 'saga').
    """
    if convention not in CURVATURE_CONVENTIONS:
        raise ValueError(f"convention must be one of {CURVATURE_CONVENTIONS}")
    if edge_mode is None:
        edge_mode = "saga" if convention == "saga" else "gdal"
    z1, z2, z3, z4, z5, z6, z7, z8, z9 = _window3x3(dem, edge_mode)
    Lx, Ly = float(cellsize_x), float(cellsize_y)
    D = ((z4 + z6) / 2.0 - z5) / Lx ** 2          # 1/2 d2z/dx2
    E = ((z2 + z8) / 2.0 - z5) / Ly ** 2          # 1/2 d2z/dy2 (y = north)
    F = (-z1 + z3 + z7 - z9) / (4.0 * Lx * Ly)    # d2z/dxdy
    G = (z6 - z4) / (2.0 * Lx)                    # dz/dx
    H = (z2 - z8) / (2.0 * Ly)                    # dz/dy
    g2 = G ** 2 + H ** 2
    out = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        if convention == "samkit":
            out["general"] = -2.0 * (D + E) * 100.0
            prof = -2.0 * (D * G ** 2 + E * H ** 2 + F * G * H) / g2 * 100.0
            plan = 2.0 * (D * H ** 2 + E * G ** 2 - F * G * H) / g2 * 100.0
            out["profile"] = np.where(g2 == 0, 0.0, prof)
            out["plan"] = np.where(g2 == 0, 0.0, plan)
        elif convention == "saga":
            # SAGA's variable naming: r,t = polynomial coefficients D,E of the
            # N-S and E-W axes, s = F, p = dz/dy(north), q = dz/dx(east)
            r, t, s, p, q = E, D, F, H, G
            p2, q2 = p * p, q * q
            spq = s * p * q
            out["general"] = -2.0 * (r + t)
            out["profile"] = np.where(g2 > 0, -(r * p2 + t * q2 + 2 * spq) / (g2 * (1 + g2) ** 1.5), np.nan)
            out["plan"] = np.where(g2 > 0, -(t * p2 + r * q2 - 2 * spq) / g2 ** 1.5, np.nan)
            out["tangential"] = np.where(g2 > 0, -(t * p2 + r * q2 - 2 * spq) / (g2 * np.sqrt(1 + g2)), np.nan)
        else:
            zxx, zyy, zxy, p, q = 2 * D, 2 * E, F, G, H
            out["general"] = -(zxx + zyy)
            out["profile"] = np.where(g2 > 0, -(zxx * p * p + 2 * zxy * p * q + zyy * q * q) / (g2 * (1 + g2) ** 1.5), 0.0)
            out["plan"] = np.where(g2 > 0, -(zyy * p * p - 2 * zxy * p * q + zxx * q * q) / (g2 * np.sqrt(1 + g2)), 0.0)
    nan_mask = np.isnan(np.asarray(dem, dtype=float))
    for v in out.values():
        v[nan_mask] = np.nan
    return out


# ---------------------------------------------------------------------------
# TPI (Topographic Position Index, Weiss 2001)
# ---------------------------------------------------------------------------

def _focal_sum(arr, kernel):
    if not HAS_SCIPY:
        raise RuntimeError("scipy is required")
    if kernel.shape[0] > 15:
        from scipy.signal import fftconvolve
        return fftconvolve(arr, kernel, mode="same")
    from scipy.ndimage import convolve
    return convolve(arr, kernel, mode="constant", cval=0.0)


def compute_tpi(dem, radius_cells=5, kernel="square", exclude_center=False, compat=None):
    """
    TPI = z - mean(z in the neighbourhood), using ONLY valid cells inside the raster
    (nodata and off-raster cells are ignored, as in WhiteboxTools and SAGA).

    kernel='square'  (2r+1)x(2r+1) window  -> WhiteboxTools DiffFromMeanElev
    kernel='circle'  cells with distance < r (cells)   -> SAGA 'Topographic Position
                     Index' with RADIUS 0..r*cellsize and no distance weighting
    exclude_center=True with radius 1, square -> gdaldem TPI (Weiss 2001 annulus)
    compat='whitebox' reproduces WhiteboxTools DiffFromMeanElev bit-for-bit in
                     spirit: elevations binned to 0.01 and its summed-area window,
                     which drops the first row/column for windows touching the
                     top/left raster edge.
    """
    if compat == "whitebox":
        return _tpi_whitebox(dem, radius_cells)
    dem = np.asarray(dem, dtype=np.float64)
    valid = np.isfinite(dem)
    r = int(radius_cells)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    if kernel == "circle":
        k = (np.hypot(xx, yy) < r).astype(np.float64)   # SAGA: distance < radius
    elif kernel == "square":
        k = np.ones((2 * r + 1, 2 * r + 1))
    else:
        raise ValueError("kernel must be 'square' or 'circle'")
    if exclude_center:
        k[r, r] = 0.0
    if HAS_SCIPY:
        num = _focal_sum(np.where(valid, dem, 0.0), k)
        den = _focal_sum(valid.astype(np.float64), k)
    else:  # square-only fallback without scipy
        num = _box_mean_fallback(np.where(valid, dem, 0.0), r) * k.size
        den = _box_mean_fallback(valid.astype(np.float64), r) * k.size
    with np.errstate(divide="ignore", invalid="ignore"):
        tpi = dem - num / den
    tpi[~valid | (den < 0.5)] = np.nan
    return tpi


def _tpi_whitebox(dem, radius_cells):
    dem = np.asarray(dem, dtype=np.float64)
    valid = np.isfinite(dem)
    rows, cols = dem.shape
    mult = 100.0
    min_bin = np.floor(np.nanmin(dem) * mult)
    b = np.where(valid, np.floor(dem * mult) - min_bin, 0.0)
    I = np.cumsum(np.cumsum(b, 0), 1)
    In = np.cumsum(np.cumsum(valid.astype(np.float64), 0), 1)
    mid = int(radius_cells)
    r = np.arange(rows); c = np.arange(cols)
    y1 = np.clip(r - mid - 1, 0, rows - 1); y2 = np.clip(r + mid, 0, rows - 1)
    x1 = np.clip(c - mid - 1, 0, cols - 1); x2 = np.clip(c + mid, 0, cols - 1)
    Y1, X1 = np.meshgrid(y1, x1, indexing="ij"); Y2, X2 = np.meshgrid(y2, x2, indexing="ij")
    n = In[Y2, X2] + In[Y1, X1] - In[Y1, X2] - In[Y2, X1]
    sm = I[Y2, X2] + I[Y1, X1] - I[Y1, X2] - I[Y2, X1]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (b - sm / n) / mult
    out[~valid | (n <= 0)] = np.nan
    return out


def _box_mean_fallback(arr, radius):
    k = 2 * radius + 1
    padded = np.pad(arr, radius, mode="constant", constant_values=0.0)
    cumsum = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    cumsum = np.pad(cumsum, ((1, 0), (1, 0)), mode="constant")
    rows, cols = arr.shape
    total = (cumsum[k:k + rows, k:k + cols] - cumsum[0:rows, k:k + cols]
             - cumsum[k:k + rows, 0:cols] + cumsum[0:rows, 0:cols])
    return total / float(k * k)


# ---------------------------------------------------------------------------
# Depression filling, D8 / MFD flow accumulation, TWI
# ---------------------------------------------------------------------------
# neighbour order and distances follow WhiteboxTools' D8 tools
# (NE, E, SE, S, SW, W, NW, N) so that ties resolve identically
_WBT_DC = (1, 1, 1, 0, -1, -1, -1, 0)
_WBT_DR = (-1, 0, 1, 1, 1, 0, -1, -1)
# kept for backward compatibility
_D8_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_D8_DIST_FACTOR = [np.sqrt(2), 1, np.sqrt(2), 1, 1, np.sqrt(2), 1, np.sqrt(2)]


def default_flat_increment(dem, cellsize_x=1.0, cellsize_y=1.0):
    """WhiteboxTools FillDepressions default flat increment."""
    zmax = float(np.nanmax(dem))
    digits = len(str(int(abs(zmax)))) if np.isfinite(zmax) else 3
    return (1.0 / 10.0 ** (9 - digits)) * np.ceil(np.hypot(cellsize_x, cellsize_y))


def fill_depressions(dem, cellsize_x=1.0, cellsize_y=1.0, epsilon=None, flats="whitebox"):
    """
    Depression filling that drains every valid cell.

    1. Priority-Flood (Barnes, Lehman & Mulla 2014): each depression is raised to
       its spill elevation.
    2. Flat resolution:
       flats='whitebox' (default) - as WhiteboxTools FillDepressions --fix_flats:
           starting from the outlet cells, flat cells are visited in order of their
           ORIGINAL elevation and each is raised by a tiny increment above the cell
           it was reached from, so flow inside filled depressions follows the
           original micro-topography towards the outlet.
       flats='epsilon' - Priority-Flood+epsilon (gradient by distance from outlet).
    epsilon: increment (default = WhiteboxTools' default flat increment).
    """
    import heapq
    orig = np.array(dem, dtype=np.float64)
    z = orig.copy()
    rows, cols = z.shape
    valid = np.isfinite(z)
    if epsilon is None:
        epsilon = default_flat_increment(z, cellsize_x, cellsize_y)
    pv = np.pad(valid, 1, constant_values=False)
    interior = np.ones_like(valid)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            interior &= pv[1 + dr:1 + dr + rows, 1 + dc:1 + dc + cols]
    seeds = valid & ~interior
    closed = ~valid
    heap = [(z[r, c], int(r), int(c)) for r, c in zip(*np.nonzero(seeds))]
    closed[seeds] = True
    heapq.heapify(heap)
    push, pop = heapq.heappush, heapq.heappop
    eps = epsilon if flats == "epsilon" else 0.0
    while heap:
        zc, r, c = pop(heap)
        for k in range(8):
            rn, cn = r + _WBT_DR[k], c + _WBT_DC[k]
            if 0 <= rn < rows and 0 <= cn < cols and not closed[rn, cn]:
                closed[rn, cn] = True
                zn = z[rn, cn]
                if zn <= zc:
                    zn = zc + eps
                    z[rn, cn] = zn
                push(heap, (zn, rn, cn))
    if flats == "epsilon":
        return z

    # --- WhiteboxTools-style flat fixing -------------------------------------
    p = np.pad(z, 1, constant_values=np.nan)
    has_lower = np.zeros(z.shape, bool)
    for k in range(8):
        zn = p[1 + _WBT_DR[k]:1 + _WBT_DR[k] + rows, 1 + _WBT_DC[k]:1 + _WBT_DC[k] + cols]
        with np.errstate(invalid="ignore"):
            has_lower |= zn < z
    flat = valid & ~has_lower & ~seeds          # cells that cannot drain yet
    if not flat.any():
        return z
    fixed = np.zeros(z.shape, bool)
    # outlets: draining cells at the same elevation as an adjacent flat cell
    outl = []
    for r, c in zip(*np.nonzero(flat)):
        for k in range(8):
            rn, cn = r + _WBT_DR[k], c + _WBT_DC[k]
            if 0 <= rn < rows and 0 <= cn < cols and valid[rn, cn] and not flat[rn, cn] \
                    and z[rn, cn] == z[r, c]:
                outl.append((z[rn, cn], int(rn), int(cn)))
    heapq.heapify(outl)
    while outl:
        zo, r, c = pop(outl)
        if fixed[r, c]:
            continue
        fixed[r, c] = True
        group = [(r, c)]
        while outl and outl[0][0] == zo:
            _, r2, c2 = pop(outl)
            if not fixed[r2, c2]:
                fixed[r2, c2] = True
                group.append((r2, c2))
        h2 = []
        for r, c in group:
            for k in range(8):
                rn, cn = r + _WBT_DR[k], c + _WBT_DC[k]
                if 0 <= rn < rows and 0 <= cn < cols and not fixed[rn, cn] and valid[rn, cn] \
                        and z[rn, cn] == zo:
                    z[rn, cn] = zo + epsilon
                    fixed[rn, cn] = True
                    push(h2, (orig[rn, cn], rn, cn))
        while h2:
            _, r, c = pop(h2)
            zc = z[r, c]
            for k in range(8):
                rn, cn = r + _WBT_DR[k], c + _WBT_DC[k]
                if 0 <= rn < rows and 0 <= cn < cols and not fixed[rn, cn] and valid[rn, cn]:
                    zn = z[rn, cn]
                    if zo <= zn < zc + epsilon:
                        z[rn, cn] = zc + epsilon
                        fixed[rn, cn] = True
                        push(h2, (orig[rn, cn], rn, cn))
    return z


def flow_direction_d8(filled, cellsize_x=1.0, cellsize_y=1.0):
    """
    D8 receiver (flat index) of every cell, -1 if none: steepest positive drop per
    unit distance, WhiteboxTools neighbour order/distances, first maximum wins.
    """
    rows, cols = filled.shape
    diag = np.hypot(cellsize_x, cellsize_y)
    lengths = (diag, cellsize_x, diag, cellsize_y, diag, cellsize_x, diag, cellsize_y)
    p = np.pad(filled, 1, constant_values=np.nan)
    best = np.full(filled.shape, 0.0)
    tgt = np.full(filled.shape, -1, dtype=np.int64)
    rr, cc = np.indices(filled.shape)
    for k in range(8):
        zn = p[1 + _WBT_DR[k]:1 + _WBT_DR[k] + rows, 1 + _WBT_DC[k]:1 + _WBT_DC[k] + cols]
        with np.errstate(invalid="ignore"):
            s = (filled - zn) / lengths[k]
        better = np.isfinite(s) & (s > best)
        best = np.where(better, s, best)
        tgt = np.where(better, (rr + _WBT_DR[k]) * cols + (cc + _WBT_DC[k]), tgt)
    tgt[~np.isfinite(filled)] = -1
    return tgt


def _accumulate(targets, valid):
    """Topological (Kahn) accumulation of 1 unit per cell along a receiver array."""
    n = targets.size
    acc = np.where(valid, 1.0, 0.0)
    has = targets >= 0
    indeg = np.bincount(targets[has], minlength=n)
    frontier = np.nonzero((indeg == 0) & valid)[0]
    while frontier.size:
        t = targets[frontier]
        m = t >= 0
        src, dst = frontier[m], t[m]
        np.add.at(acc, dst, acc[src])
        np.subtract.at(indeg, dst, 1)
        cand = np.unique(dst)
        frontier = cand[indeg[cand] == 0]
    acc[~valid] = np.nan
    return acc


def compute_flow_accumulation(dem, cellsize_x=1.0, cellsize_y=1.0, epsilon=None):
    """D8 flow accumulation (number of cells, incl. the cell itself) on a
    Priority-Flood+epsilon filled DEM (reproduces WhiteboxTools FillDepressions
    --fix_flats + D8FlowAccumulation --out_type=cells)."""
    dem = np.asarray(dem, dtype=np.float64)
    filled = fill_depressions(dem, cellsize_x, cellsize_y, epsilon)
    tgt = flow_direction_d8(filled, cellsize_x, cellsize_y).ravel()
    return _accumulate(tgt, np.isfinite(filled).ravel()).reshape(dem.shape)


def compute_flow_accumulation_mfd(dem, exponent=1.1, cellsize_x=1.0, cellsize_y=1.0, epsilon=None):
    """
    Multiple-flow-direction accumulation (Freeman 1991 / Quinn et al. 1991):
    each cell's flow is split among all lower neighbours in proportion to
    (drop / distance) ** exponent, on a Priority-Flood+epsilon filled DEM.
    """
    dem = np.asarray(dem, dtype=np.float64)
    filled = fill_depressions(dem, cellsize_x, cellsize_y, epsilon)
    rows, cols = filled.shape
    diag = np.hypot(cellsize_x, cellsize_y)
    lengths = (diag, cellsize_x, diag, cellsize_y, diag, cellsize_x, diag, cellsize_y)
    p = np.pad(filled, 1, constant_values=np.nan)
    W, T = [], []
    rr, cc = np.indices(filled.shape)
    for k in range(8):
        zn = p[1 + _WBT_DR[k]:1 + _WBT_DR[k] + rows, 1 + _WBT_DC[k]:1 + _WBT_DC[k] + cols]
        with np.errstate(invalid="ignore"):
            s = (filled - zn) / lengths[k]
        w = np.where(np.isfinite(s) & (s > 0), s, 0.0) ** exponent
        W.append(w.ravel())
        T.append(((rr + _WBT_DR[k]) * cols + (cc + _WBT_DC[k])).ravel())
    W = np.array(W); T = np.array(T)
    tot = W.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        W = np.where(tot > 0, W / tot, 0.0)
    flat = filled.ravel()
    valid = np.isfinite(flat)
    acc = np.where(valid, 1.0, np.nan)
    order = np.argsort(-np.where(valid, flat, -np.inf), kind="stable")
    for idx in order[:int(valid.sum())]:
        if tot[idx] > 0:
            a = acc[idx]
            for k in range(8):
                if W[k, idx] > 0:
                    acc[T[k, idx]] += a * W[k, idx]
    return acc.reshape(dem.shape)


def compute_twi(dem, cellsize_x, cellsize_y, min_slope_deg=None, flow_method="d8",
                mfd_exponent=1.1, epsilon=None):
    """
    Topographic Wetness Index ln(SCA / tan(slope)) (Beven & Kirkby 1979).

    Reproduces WhiteboxTools: FillDepressions(--fix_flats) -> D8FlowAccumulation
    (out_type = specific contributing area) -> Slope -> WetnessIndex:
      SCA  = contributing cells * (cell area / mean cell size)
      slope from the ORIGINAL DEM (Horn, WhiteboxTools edge rule)
      zero-slope cells -> NaN (min_slope_deg=None, WhiteboxTools behaviour), or
      slope clamped to min_slope_deg if given (pre-1.0 SAMKit used 0.1).
    flow_method: 'd8' (default) or 'mfd'.
    """
    dem = np.asarray(dem, dtype=np.float64)
    _, _, slope_rad, _, _ = compute_slope_aspect(dem, cellsize_x, cellsize_y, edge_mode="whitebox")
    if flow_method == "mfd":
        acc = compute_flow_accumulation_mfd(dem, mfd_exponent, cellsize_x, cellsize_y, epsilon)
    else:
        acc = compute_flow_accumulation(dem, cellsize_x, cellsize_y, epsilon)
    width = (cellsize_x + cellsize_y) / 2.0
    sca = acc * (cellsize_x * cellsize_y) / width
    if min_slope_deg is not None:
        slope_rad = np.maximum(slope_rad, np.radians(min_slope_deg))
    with np.errstate(divide="ignore", invalid="ignore"):
        twi = np.log(sca / np.tan(slope_rad))
    twi[~np.isfinite(twi)] = np.nan
    twi[np.isnan(dem)] = np.nan
    return twi


# ---------------------------------------------------------------------------
# MRVBF / MRRTF  -  faithful port of SAGA GIS 'Multiresolution Index of Valley
# Bottom Flatness (MRVBF)' (ta_morphometry tool 8, O. Conrad 2006), which
# implements Gallant & Dowling (2003).
#
# PROVENANCE / LICENCE: the functions below are a Python translation of the
# algorithm in SAGA's mrvbf.cpp and of the SAGA API routines it relies on
# (CSG_Grid::Get_Gradient, the default B-spline Get_Value incl. its 4x4 nodata
# fill, nearest-neighbour Assign, CSG_Grid_Radius). SAGA is (C) O. Conrad and
# contributors, licensed under the GNU GPL v2 or later; this derived code is
# distributed under the same licence (see LICENSE and NOTICE in this plugin).
# ---------------------------------------------------------------------------

def _saga_T(x, t, p):
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        return 1.0 / (1.0 + np.power(x / t, p))


def _f32(a):
    return np.asarray(a, dtype=np.float32).astype(np.float64)


def _saga_gradient_pct(g, cs):
    """100*tan(slope) with CSG_Grid::Get_Gradient (4 neighbours, one-sided at edges).
    g is in SAGA orientation (row 0 = south)."""
    ny, nx = g.shape
    p = np.pad(g, 1, constant_values=np.nan)
    c = g
    N, S = p[2:, 1:-1], p[:-2, 1:-1]
    E, W = p[1:-1, 2:], p[1:-1, :-2]

    def d(to, frm):
        return np.where(np.isfinite(to), to - c, np.where(np.isfinite(frm), c - frm, 0.0))
    dzN, dzE, dzS, dzW = d(N, S), d(E, W), d(S, N), d(W, E)
    G = (dzN - dzS) / (2.0 * cs)
    H = (dzE - dzW) / (2.0 * cs)
    out = 100.0 * np.hypot(G, H)
    out[~np.isfinite(c)] = np.nan
    return _f32(out)


def _saga_smoothed(g):
    """MRVBF Get_Smoothed: radius-5 kernel 4.3565*exp(-(d/3)^2), weighted mean of
    valid in-grid cells (also defined on nodata cells that have valid neighbours)."""
    yy, xx = np.mgrid[-5:6, -5:6]
    k = 4.3565 * np.exp(-(np.hypot(xx, yy) / 3.0) ** 2)
    v = np.isfinite(g)
    from scipy.ndimage import correlate
    num = correlate(np.where(v, g, 0.0), k, mode="constant", cval=0.0)
    den = correlate(v.astype(float), k, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / den, np.nan)
    return _f32(out)


def _saga_percentiles(g, radius):
    """Fraction of cells within distance < radius that are lower than the centre
    (CSG_Grid_Radius rings 0..radius-1); nLower / (nPoints - 1)."""
    ny, nx = g.shape
    v = np.isfinite(g)
    p = np.pad(g, radius, constant_values=np.nan)
    npts = np.zeros(g.shape); nlow = np.zeros(g.shape)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if int(np.sqrt(dx * dx + dy * dy)) < radius:
                zn = p[radius + dy:radius + dy + ny, radius + dx:radius + dx + nx]
                ok = np.isfinite(zn)
                npts += ok
                with np.errstate(invalid="ignore"):
                    nlow += ok & (zn < g)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(v & (npts > 1), nlow / (npts - 1.0), np.nan)
    return _f32(out)


def _bspline_R(d):
    """SAGA _Get_ValAtPos_BSpline basis weights for offsets -1..+2 (shape (4, n))."""
    R = []
    for i in range(4):
        s = np.zeros_like(d)
        for coef, sh in ((1.0, 1.0), (-4.0, 0.0), (6.0, -1.0), (-4.0, -2.0)):
            dd = i - d + sh
            s += np.where(dd > 0, coef * dd ** 3, 0.0)
        R.append(s / 6.0)
    return np.array(R)


def _fill4x4(g, x, y):
    """SAGA _Get_ValAtPos_Fill4x4Submatrix for block origin (x-1, y-1)."""
    ny, nx = g.shape

    def ingrid(i, j):
        return 0 <= i < nx and 0 <= j < ny and np.isfinite(g[j, i])
    v = np.full((4, 4), np.nan)  # v[ix][iy]
    nnd = 0
    for iy in range(4):
        for ix in range(4):
            jx, jy = x - 1 + ix, y - 1 + iy
            if ingrid(jx, jy):
                v[ix, iy] = g[jy, jx]
            else:
                nnd += 1
    i = 0
    while nnd > 0 and nnd < 16 and i < 16:
        t = v.copy()
        for iy in range(4):
            for ix in range(4):
                if np.isnan(t[ix, iy]):
                    s, n = 0.0, 0
                    for jy in range(iy - 1, iy + 2):
                        for jx in range(ix - 1, ix + 2):
                            if ingrid(jx + x - 1, jy + y - 1):
                                s += g[jy + y - 1, jx + x - 1]; n += 1
                            elif 0 <= jy < 4 and 0 <= jx < 4 and not np.isnan(t[jx, jy]):
                                s += t[jx, jy]; n += 1
                    if n > 0:
                        v[ix, iy] = s / n
                        nnd -= 1
        i += 1
    return v if nnd == 0 else None


def _saga_get_value(g, xmin, ymin, cs, px, py, method="bspline"):
    """Vectorised CSG_Grid::Get_Value(x, y, &Value, Resampling) on grid g (SAGA
    orientation). Returns NaN where SAGA returns false."""
    ny, nx = g.shape
    fx = (px - xmin) / cs
    fy = (py - ymin) / cs
    out = np.full(px.shape, np.nan)
    h = 0.5 * cs
    inside = ((px >= xmin - h) & (px <= (xmin + (nx - 1.0) * cs) + h)
              & (py >= ymin - h) & (py <= (ymin + (ny - 1.0) * cs) + h))
    ix = np.floor(fx).astype(np.int64); dx = fx - ix
    iy = np.floor(fy).astype(np.int64); dy = fy - iy
    nnx = ix + (0.5 + dx).astype(np.int64)
    nny = iy + (0.5 + dy).astype(np.int64)
    inb = inside & (nnx >= 0) & (nnx < nx) & (nny >= 0) & (nny < ny)
    ok = np.zeros(px.shape, bool)
    ok[inb] = np.isfinite(g[nny[inb], nnx[inb]])
    if method == "nearest":
        out[ok] = g[nny[ok], nnx[ok]]
        return out
    idx = np.nonzero(ok)[0]
    if idx.size == 0:
        return out
    bx, by = ix[idx], iy[idx]
    key = by * (nx + 4) + bx
    ukeys, inv = np.unique(key, return_inverse=True)
    blocks = np.full((ukeys.size, 4, 4), np.nan)  # [k, ix, iy]
    gp = np.pad(g, 1, constant_values=np.nan)
    gp = np.pad(gp, ((0, 2), (0, 2)), constant_values=np.nan)
    ubx = ukeys % (nx + 4); uby = ukeys // (nx + 4)
    for jy in range(4):
        for jx in range(4):
            rr = uby - 1 + jy + 1; cc = ubx - 1 + jx + 1
            okk = (rr >= 0) & (cc >= 0) & (rr < gp.shape[0]) & (cc < gp.shape[1])
            vals = np.full(ukeys.size, np.nan)
            vals[okk] = gp[rr[okk], cc[okk]]
            blocks[:, jx, jy] = vals
    good = np.ones(ukeys.size, bool)
    for k in np.nonzero(np.isnan(blocks).any(axis=(1, 2)))[0]:
        b = _fill4x4(g, int(ubx[k]), int(uby[k]))
        if b is None:
            good[k] = False
        else:
            blocks[k] = b
    Rx = _bspline_R(dx[idx]); Ry = _bspline_R(dy[idx])
    B = blocks[inv]
    val = np.einsum("nij,in,jn->n", B, Rx, Ry)
    val[~good[inv]] = np.nan
    out[idx] = val
    return out


def compute_mrvbf_mrrtf(dem, cellsize_x, cellsize_y, t_slope=16.0, t_pctl_v=0.4,
                         t_pctl_r=0.35, p_slope=4.0, p_pctl=3.0, max_res=100.0,
                         classify=False, n_levels=None, origin=None):
    """
    MRVBF / MRRTF (Gallant & Dowling 2003) exactly as computed by SAGA GIS
    ta_morphometry 8 (same parameters and defaults); verified bit-identical to
    SAGA 9.3.1. SAGA requires square cells: the mean of the x and y cell sizes is
    used (rounded to 10 decimals), exactly as the corrected SAGA route writes it.
    origin = (x, y) of the CENTRE of the lower-left cell in map units (SAGA's
    POSITION_XMIN/YMIN); pass it to reproduce SAGA to the last bit - SAGA's grid
    sizes at coarse levels depend on floating-point arithmetic in world coordinates.
    `n_levels` is accepted for backward compatibility and ignored (the number of
    levels follows from max_res, as in SAGA).
    Returns {'mrvbf': array, 'mrrtf': array}.
    """
    cs = float("%.10f" % (0.5 * (float(cellsize_x) + float(cellsize_y))))
    x0, y0 = (0.0, 0.0) if origin is None else (float("%.10f" % origin[0]), float("%.10f" % origin[1]))
    Z = _f32(np.asarray(dem, dtype=np.float64)[::-1].copy())   # SAGA orientation
    ny, nx = Z.shape
    xs_, ys_ = [x0], [y0]
    for _ in range(nx - 1):
        xs_.append(xs_[-1] + cs)                 # SAGA accumulates xp += cellsize
    for _ in range(ny - 1):
        ys_.append(ys_[-1] + cs)
    PX, PY = [a.ravel() for a in np.meshgrid(np.array(xs_), np.array(ys_))]
    xr = (x0 + (nx - 1.0) * cs) - x0; yr = (y0 + (ny - 1.0) * cs) - y0
    max_resolution = max_res / 100.0 * np.hypot(xr, yr)

    CF = np.ones(ny * nx)
    MRVBF = np.full(ny * nx, np.nan); MRRTF = np.full(ny * nx, np.nan)

    def flatness(sl, pc, T):
        slope = _saga_get_value(sl[0], sl[1], sl[2], sl[3], PX, PY)
        pct = _saga_get_value(pc[0], pc[1], pc[2], pc[3], PX, PY)
        ok = np.isfinite(slope) & np.isfinite(pct)
        vf = np.full(ny * nx, np.nan); rf = np.full(ny * nx, np.nan)
        cf = CF[ok] * _saga_T(slope[ok], T, p_slope)
        v = cf * _saga_T(pct[ok], t_pctl_v, p_pctl)
        r = cf * _saga_T(1.0 - pct[ok], t_pctl_r, p_pctl)
        CF[ok] = _f32(cf)
        vf[ok] = _f32(1.0 - _saga_T(v, 0.3, 4.0))
        rf[ok] = _f32(1.0 - _saga_T(r, 0.3, 4.0))
        return vf, rf

    def combine(level, M, F):
        t = 0.4
        p = np.log((level - 0.5) / 0.1) / np.log(1.5)
        ok = np.isfinite(M) & np.isfinite(F)
        w = 1.0 - _saga_T(F[ok], t, p)
        M[ok] = _f32(w * (level - 1 + F[ok]) + (1.0 - w) * M[ok])

    # step 1
    level, res, T = 1, cs, t_slope
    slopes = (_saga_gradient_pct(Z, cs), x0, y0, cs)
    pct = (_saga_percentiles(Z, 3), x0, y0, cs)
    MRVBF[:], MRRTF[:] = flatness(slopes, pct, T)
    # step 2
    T /= 2.0; level += 1
    pct = (_saga_percentiles(Z, 6), x0, y0, cs)
    VF, RF = flatness(slopes, pct, T)
    combine(level, MRVBF, VF); combine(level, MRRTF, RF)
    # further steps on progressively coarser (x3) smoothed DEMs
    cur, cxmin, cymin, ccs = Z, x0, y0, cs
    while res < max_resolution:
        res *= 3.0; T /= 2.0; level += 1
        sm = _saga_smoothed(cur)
        slopes = (_saga_gradient_pct(sm, ccs), cxmin, cymin, ccs)
        cny, cnx = cur.shape
        nnx = 2 + int(((cxmin + (cnx - 1.0) * ccs) - cxmin) / res)
        nny = 2 + int(((cymin + (cny - 1.0) * ccs) - cymin) / res)
        gy, gx = np.mgrid[0:nny, 0:nnx]
        vals = _saga_get_value(sm, cxmin, cymin, ccs, (cxmin + gx * res).ravel(),
                               (cymin + gy * res).ravel(), method="nearest")
        cur, ccs = _f32(vals.reshape(nny, nnx)), res
        pct = (_saga_percentiles(cur, 6), cxmin, cymin, ccs)
        VF, RF = flatness(slopes, pct, T)
        combine(level, MRVBF, VF); combine(level, MRRTF, RF)

    out = {}
    for name, M in (("mrvbf", MRVBF), ("mrrtf", MRRTF)):
        a = M.reshape(ny, nx)[::-1].copy()
        if classify:
            a = np.where(np.isnan(a), np.nan, np.clip(np.floor(a + 0.5), 0, 6))
        a[np.isnan(np.asarray(dem, dtype=float))] = np.nan
        out[name] = a
    return out


def compute_mrvbf_mrrtf_legacy(dem, cellsize_x, cellsize_y, n_levels=6, t_slope_deg=16.0,
                                t_pctl=0.4, p_slope=4.0, p_pctl=3.0):
    """pre-1.0 SAMKit heuristic multi-scale flatness index (NOT MRVBF; kept only
    so that old results can be reproduced). Rescaled to a maximum of 10."""
    from scipy.ndimage import gaussian_filter
    rows, cols = dem.shape
    cellsize = (cellsize_x + cellsize_y) / 2.0
    vbf = np.zeros((rows, cols)); rtf = np.zeros((rows, cols)); wr = np.ones((rows, cols))
    cur_cs = cellsize
    for level in range(n_levels):
        sigma = max(1.0, (3 ** level))
        filled = np.where(np.isnan(dem), np.nanmean(dem), dem)
        sm = gaussian_filter(filled, sigma=sigma, mode="nearest")
        sd, _, _, _, _ = compute_slope_aspect(sm, cur_cs, cur_cs, edge_mode="legacy")
        tl = t_slope_deg / (2.0 ** level)
        flat = 1.0 / (1.0 + (sd / max(tl, 1e-3)) ** p_slope)
        win = int(min(max(3, 3 * (level + 1)), min(rows, cols) // 2 or 3))
        lm = _box_mean_fallback(sm, win // 2); lsq = _box_mean_fallback(sm ** 2, win // 2)
        zz = (sm - lm) / np.sqrt(np.maximum(lsq - lm ** 2, 1e-6))
        low = 1.0 / (1.0 + np.exp((zz / t_pctl) * p_pctl)); high = 1.0 / (1.0 + np.exp((-zz / t_pctl) * p_pctl))
        contrib = np.clip(wr * (1.0 / n_levels) * (level + 1), 0, wr)
        vbf += flat * low * contrib * n_levels / (level + 1)
        rtf += flat * high * contrib * n_levels / (level + 1)
        wr = np.maximum(wr - contrib, 0); cur_cs *= 3.0
    out = {}
    for k, a in (("mrvbf", vbf), ("mrrtf", rtf)):
        a = np.clip(a, 0, None); a = a / (a.max() + 1e-9) * 10.0 if np.nanmax(a) > 0 else a
        a[np.isnan(dem)] = np.nan; out[k] = a
    return out


# backward-compatible aliases used by older code/tests
def _fill_pits(dem, iterations=50):
    return fill_depressions(dem)


def _flow_dir_d8(dem):
    return flow_direction_d8(np.asarray(dem, dtype=np.float64))


# ---------------------------------------------------------------------------
# Point sampling of raster stacks (for DSM tab)
# ---------------------------------------------------------------------------

def sample_stack_at_points(raster_paths, xs, ys, src_epsg=None, dst_epsg=None):
    """
    Sample a list of aligned raster files at coordinate arrays xs, ys (already in the
    raster CRS -- reprojection of points should be handled by the caller before this).
    Returns array of shape (n_points, n_rasters) with np.nan where out of bounds/nodata.
    """
    if not HAS_GDAL:
        raise RuntimeError("GDAL (osgeo) is not available in this Python environment.")
    n_points = len(xs)
    n_rasters = len(raster_paths)
    out = np.full((n_points, n_rasters), np.nan)

    for j, path in enumerate(raster_paths):
        ds = gdal.Open(path)
        gt = ds.GetGeoTransform()
        band = ds.GetRasterBand(1)
        nodata = band.GetNoDataValue()
        arr = band.ReadAsArray().astype(np.float64)
        rows, cols = arr.shape
        inv_det = gt[1] * gt[5] - gt[2] * gt[4]
        for i in range(n_points):
            x, y = xs[i], ys[i]
            col = (x - gt[0]) / gt[1]
            row = (y - gt[3]) / gt[5]
            r, c = int(row), int(col)
            if 0 <= r < rows and 0 <= c < cols:
                val = arr[r, c]
                if nodata is not None and val == nodata:
                    continue
                if np.isnan(val):
                    continue
                out[i, j] = val
        ds = None
    return out


def read_raster_stack(raster_paths):
    """Read multiple aligned rasters into a stack (n_rasters, rows, cols) + reference RasterInfo."""
    ref = read_raster(raster_paths[0])
    stack = [ref.array]
    for p in raster_paths[1:]:
        ri = read_raster(p)
        stack.append(ri.array)
    return np.stack(stack, axis=0), ref
