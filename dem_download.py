# -*- coding: utf-8 -*-
"""
dem_download.py
Fetching a DEM for an area of interest, and converting a "raw" DEM (typically
delivered in geographic degrees, EPSG:4326) into a projected, metre-based grid
suitable for terrain analysis.

Why the reprojection matters
----------------------------
Slope, curvature, TWI and MRVBF are all computed from elevation differences
divided by a horizontal distance. If the DEM's horizontal units are degrees
and the vertical units are metres, that division mixes two incompatible units
and every derived covariate is wrong -- and wrong by a latitude-dependent
factor, so it doesn't even fail consistently. A degree of longitude is about
111 km at the equator but only about 56 km at 60 degrees latitude, so the same
DEM produces different slopes depending purely on where it sits on the globe.
Reprojecting to UTM (metres in both directions) before computing covariates is
therefore not a cosmetic step -- it is a prerequisite. This module picks the
correct UTM zone from the area's own centroid and warps the DEM onto a square
metre grid.

Sources
-------
1. OpenTopography's Global DEM API (SRTM, Copernicus GLO-30/90, NASADEM,
   ALOS AW3D30). Requires a free API key from
   https://portal.opentopography.org/ (register, then Account > myOpenTopo
   Authorizations and API Key).
2. AWS Terrain Tiles ("Terrarium" RGB-encoded PNG tiles, hosted on the AWS
   Open Data registry). No key or registration at all, global coverage, and
   the zoom level sets the resolution. Elevation is encoded in the RGB
   channels and decoded here with the documented formula
   `(R * 256 + G + B / 256) - 32768`.

Everything here is deliberately free of QGIS imports so it can be exercised
outside a QGIS session.
"""

import math
import os
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

try:
    from osgeo import gdal, osr
    gdal.UseExceptions()
    HAS_GDAL = True
except Exception:
    HAS_GDAL = False


USER_AGENT = "SAMKit-QGIS-plugin/1.4 (https://github.com/ssafaee/SamKit)"

OPENTOPOGRAPHY_URL = "https://portal.opentopography.org/API/globaldem"
# USGS 3DEP rasters are served by a separate endpoint and use `datasetName`
# instead of `demtype` (https://opentopography.org/news/api-access-usgs-3dep-rasters-now-available).
OPENTOPOGRAPHY_USGS_URL = "https://portal.opentopography.org/API/usgsdem"
OPENTOPOGRAPHY_USGS_DATASETS = ("USGS1m",)
OPENTOPOGRAPHY_KEY_URL = "https://portal.opentopography.org/requestService?service=api"

# display name -> (demtype code, nominal resolution in metres, coverage note)
OPENTOPOGRAPHY_DEMS = {
    "SRTM GL1 (30 m)": ("SRTMGL1", 30, "60°N to 56°S"),
    "SRTM GL3 (90 m)": ("SRTMGL3", 90, "60°N to 56°S"),
    "Copernicus GLO-30 (30 m)": ("COP30", 30, "global"),
    "Copernicus GLO-90 (90 m)": ("COP90", 90, "global"),
    "NASADEM (30 m)": ("NASADEM", 30, "60°N to 56°S"),
    "ALOS World 3D AW3D30 (30 m)": ("AW3D30", 30, "global"),
    # Fine, lidar-derived bare-earth DEM (USA only), served by OpenTopography's
    # USGS 3DEP API. OpenTopography currently limits it to academic API keys.
    "USGS 3DEP 1 m lidar DEM (1 m)": ("USGS1m", 1,
                                      "USA where 3DEP 1 m lidar exists; academic API key"),
}

TERRAIN_TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"

WEB_MERCATOR_HALF_CIRCUMFERENCE = 20037508.342789244

VERTICAL_UNIT_FACTORS = {
    "metres (no conversion needed)": 1.0,
    "international feet -> metres": 0.3048,
    "US survey feet -> metres": 1200.0 / 3937.0,
    "centimetres -> metres": 0.01,
}


class DemDownloadError(RuntimeError):
    pass


class Cancelled(RuntimeError):
    pass


def _check_cancelled(cancelled):
    if cancelled is not None and cancelled():
        raise Cancelled("Cancelled by user.")


def _require_gdal():
    if not HAS_GDAL:
        raise DemDownloadError("GDAL (osgeo) is not available in this Python environment. "
                               "Inside QGIS it always is -- this tab must be run from QGIS.")


# ---------------------------------------------------------------------------
# Area-of-interest maths
# ---------------------------------------------------------------------------

def normalize_bbox(west, south, east, north):
    """Order-correct a bounding box and clamp it to valid lon/lat ranges."""
    west, east = min(west, east), max(west, east)
    south, north = min(south, north), max(south, north)
    west = max(-180.0, min(180.0, west))
    east = max(-180.0, min(180.0, east))
    south = max(-90.0, min(90.0, south))
    north = max(-90.0, min(90.0, north))
    if east <= west or north <= south:
        raise DemDownloadError("The area of interest is empty. Draw a rectangle with a "
                               "real width and height, or enter a valid bounding box.")
    return west, south, east, north


def bbox_size_km(bbox):
    """Approximate width/height of a lon/lat bbox in kilometres."""
    west, south, east, north = bbox
    mid_lat = math.radians((south + north) / 2.0)
    width_km = (east - west) * 111.320 * math.cos(mid_lat)
    height_km = (north - south) * 110.574
    return abs(width_km), abs(height_km)


def bbox_area_km2(bbox):
    w, h = bbox_size_km(bbox)
    return w * h


def utm_epsg_for(lon, lat):
    """
    EPSG code of the UTM zone containing (lon, lat), including the two
    well-known irregular zones (south-west Norway, and Svalbard) so those
    areas don't silently get the wrong zone.

    Returns (epsg_code, zone_number, hemisphere_letter).
    """
    lon = ((float(lon) + 180.0) % 360.0) - 180.0
    lat = float(lat)
    zone = int((lon + 180.0) // 6.0) + 1

    if 56.0 <= lat < 64.0 and 3.0 <= lon < 12.0:
        zone = 32  # south-west Norway
    if 72.0 <= lat < 84.0:  # Svalbard
        if 0.0 <= lon < 9.0:
            zone = 31
        elif 9.0 <= lon < 21.0:
            zone = 33
        elif 21.0 <= lon < 33.0:
            zone = 35
        elif 33.0 <= lon < 42.0:
            zone = 37

    zone = max(1, min(60, zone))
    hemisphere = "N" if lat >= 0 else "S"
    epsg = (32600 if lat >= 0 else 32700) + zone
    return epsg, zone, hemisphere


def utm_epsg_for_bbox(bbox):
    """UTM zone of the bbox centroid -- the zone the whole AOI is warped into."""
    west, south, east, north = bbox
    return utm_epsg_for((west + east) / 2.0, (south + north) / 2.0)


def crosses_utm_zones(bbox):
    """
    True when an AOI spans more than one UTM zone. Not an error -- everything is
    projected into the centroid's zone regardless -- but distortion grows away
    from that zone's central meridian, so it's worth warning about.
    """
    west, south, east, north = bbox
    mid_lat = (south + north) / 2.0
    return utm_epsg_for(west, mid_lat)[1] != utm_epsg_for(east, mid_lat)[1]


# ---------------------------------------------------------------------------
# Web-Mercator tile maths (for the keyless AWS Terrain Tiles source)
# ---------------------------------------------------------------------------

def deg_to_tile(lon, lat, zoom):
    """Slippy-map tile x/y containing a lon/lat at the given zoom."""
    n = 2 ** zoom
    lat = max(-85.0511, min(85.0511, lat))
    lat_rad = math.radians(lat)
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return max(0, min(n - 1, x)), max(0, min(n - 1, y))


def tile_range_for_bbox(bbox, zoom):
    """(x_min, y_min, x_max, y_max) tile indices covering the bbox."""
    west, south, east, north = bbox
    x_min, y_min = deg_to_tile(west, north, zoom)
    x_max, y_max = deg_to_tile(east, south, zoom)
    return x_min, y_min, x_max, y_max


def tile_count_for_bbox(bbox, zoom):
    x_min, y_min, x_max, y_max = tile_range_for_bbox(bbox, zoom)
    return (x_max - x_min + 1) * (y_max - y_min + 1)


def tile_resolution_m(zoom, lat=0.0):
    """Ground resolution (m/pixel) of a 256 px web-Mercator tile at a latitude."""
    return (2 * WEB_MERCATOR_HALF_CIRCUMFERENCE / (2 ** zoom) / 256.0) * math.cos(math.radians(lat))


def suggest_zoom_for_resolution(target_res_m, lat=0.0, min_zoom=6, max_zoom=14):
    """Coarsest zoom level that still meets a target ground resolution."""
    for z in range(min_zoom, max_zoom + 1):
        if tile_resolution_m(z, lat) <= target_res_m:
            return z
    return max_zoom


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def _http_download(url, out_path, timeout=120, log=print, cancelled=None,
                   chunk=1 << 20):
    """Stream `url` to `out_path` (large DEMs never sit in memory). Cancellable.
    Returns the Content-Type header."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, \
                open(out_path, "wb") as f:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            next_report = 25 << 20
            while True:
                _check_cancelled(cancelled)
                block = response.read(chunk)
                if not block:
                    break
                f.write(block)
                done += len(block)
                if done >= next_report:
                    of = f" of {total / 1e6:.0f}" if total else ""
                    log(f"  … {done / 1e6:.0f}{of} MB received")
                    next_report += 25 << 20
            return response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", "replace").strip()
        except Exception:
            body = ""
        raise DemDownloadError(_explain_http_error(e.code, body, url))
    except urllib.error.URLError as e:
        raise DemDownloadError(
            f"Could not reach the server ({e.reason}).\n\n"
            f"Check your internet connection. If you are behind a corporate proxy or "
            f"firewall, set it in QGIS under Settings > Options > Network, or download "
            f"the DEM manually and use the 'Use a DEM already loaded in QGIS' source "
            f"instead to just reproject it.")


def estimate_download_mb(bbox, resolution_m):
    """Rough uncompressed float32 size of a DEM over `bbox` at `resolution_m`."""
    w_km, h_km = bbox_size_km(bbox)
    cells = (w_km * 1000.0 / resolution_m) * (h_km * 1000.0 / resolution_m)
    return cells * 4 / 1e6


def _http_get(url, timeout=120):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read(), response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", "replace").strip()
        except Exception:
            body = ""
        raise DemDownloadError(_explain_http_error(e.code, body, url))
    except urllib.error.URLError as e:
        raise DemDownloadError(
            f"Could not reach the server ({e.reason}).\n\n"
            f"Check your internet connection. If you are behind a corporate proxy or "
            f"firewall, set it in QGIS under Settings > Options > Network, or download "
            f"the DEM manually and use the 'Use a DEM already loaded in QGIS' source "
            f"instead to just reproject it.")


def _explain_http_error(code, body, url):
    snippet = (body[:400] + "…") if len(body) > 400 else body
    if code in (401, 403):
        return ("The DEM server rejected the request as unauthorised (HTTP "
                f"{code}).\n\nFor OpenTopography this almost always means the API key is "
                f"missing, mistyped, or not yet activated. Get a free key at\n"
                f"{OPENTOPOGRAPHY_KEY_URL}\nand paste it into the API key box.\n\n"
                f"If you requested the USGS 3DEP 1 m DEM: OpenTopography currently "
                f"limits that dataset to academic users. Use a 30 m dataset "
                f"(e.g. Copernicus GLO-30) instead, or request academic access from OpenTopography.\n\n"
                f"Server said: {snippet}")
    if code == 400:
        return ("The DEM server rejected the request as invalid (HTTP 400).\n\n"
                "This usually means the area of interest is outside the chosen "
                "dataset's coverage (SRTM and NASADEM only cover 60°N to 56°S -- use "
                "Copernicus GLO-30 for polar areas), or the area is too large.\n\n"
                f"Server said: {snippet}")
    if code == 404:
        return f"The requested tile or dataset was not found (HTTP 404): {url}"
    if code == 429:
        return ("The DEM server is rate-limiting the request (HTTP 429). Wait a "
                "minute and try again, or request a smaller area.")
    if code >= 500:
        return (f"The DEM server reported a temporary error (HTTP {code}). This is on "
                f"their side -- try again shortly.\n\nServer said: {snippet}")
    return f"Download failed (HTTP {code}).\n\nServer said: {snippet}"


# ---------------------------------------------------------------------------
# Source 1: OpenTopography global DEM API
# ---------------------------------------------------------------------------

def download_opentopography(bbox, demtype, api_key, out_path, log=print, cancelled=None):
    """Fetch a GeoTIFF DEM for `bbox` from OpenTopography's global DEM API."""
    _require_gdal()
    west, south, east, north = normalize_bbox(*bbox)
    if not api_key or not api_key.strip():
        raise DemDownloadError(
            "OpenTopography requires a free API key.\n\n"
            f"Register at {OPENTOPOGRAPHY_KEY_URL} , copy the key into the API key "
            f"box, and try again. Alternatively switch the source to 'AWS Terrain "
            f"Tiles', which needs no key at all.")

    usgs = demtype in OPENTOPOGRAPHY_USGS_DATASETS
    params = {
        ("datasetName" if usgs else "demtype"): demtype,
        "south": f"{south:.6f}", "north": f"{north:.6f}",
        "west": f"{west:.6f}", "east": f"{east:.6f}",
        "outputFormat": "GTiff",
        "API_Key": api_key.strip(),
    }
    base = OPENTOPOGRAPHY_USGS_URL if usgs else OPENTOPOGRAPHY_URL
    url = base + "?" + urllib.parse.urlencode(params)
    safe_url = url.replace(api_key.strip(), "***")
    log(f"Requesting {demtype} from OpenTopography…")
    log(f"  {safe_url}")
    if demtype == "USGS1m":
        mb = estimate_download_mb((west, south, east, north), 1.0)
        log(f"  1 m DEM over this area: roughly {mb:,.0f} MB uncompressed. Large areas take "
            f"a while and may exceed OpenTopography's per-request area limit -- if so, "
            f"split the area or use a 30 m dataset.")

    _check_cancelled(cancelled)
    # 1 m requests are processed server-side before streaming: allow longer.
    content_type = _http_download(url, out_path, timeout=1800 if usgs else 300,
                                  log=log, cancelled=cancelled)
    _check_cancelled(cancelled)

    size = os.path.getsize(out_path) if os.path.exists(out_path) else 0
    if size == 0:
        raise DemDownloadError("The server returned an empty response.")
    # An error page can come back with HTTP 200, so sniff the payload itself
    # rather than trusting the status code (TIFF and BigTIFF magic numbers).
    with open(out_path, "rb") as f:
        head = f.read(500)
    if head[:4] not in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
        text = head.decode("utf-8", "replace").strip()
        try:
            os.remove(out_path)
        except OSError:
            pass
        hint = ("For the USGS 3DEP datasets the area must lie in the USA (and, for 1 m, "
                "where 3DEP lidar has been collected); the 1 m dataset also needs an "
                "academic OpenTopography API key." if usgs else
                "A common cause is an area of interest outside the dataset's coverage, "
                "or an API key that has not been activated yet.")
        raise DemDownloadError(
            "The server returned a message instead of a GeoTIFF "
            f"(content type: {content_type or 'unknown'}):\n\n{text}\n\n{hint}")

    log(f"  downloaded {size / 1e6:.1f} MB -> {out_path}")
    _validate_raster(out_path)
    return out_path


# ---------------------------------------------------------------------------
# Source 2: AWS Terrain Tiles (keyless)
# ---------------------------------------------------------------------------

def _decode_terrarium(rgb):
    """rgb: (3+, 256, 256) uint8 array -> elevation in metres."""
    r = rgb[0].astype(np.float64)
    g = rgb[1].astype(np.float64)
    b = rgb[2].astype(np.float64)
    return (r * 256.0 + g + b / 256.0) - 32768.0


def download_terrain_tiles(bbox, zoom, out_path, log=print, cancelled=None, max_tiles=400):
    """
    Mosaic AWS Terrain Tiles (terrarium PNGs) covering `bbox` into a single
    Web-Mercator (EPSG:3857) GeoTIFF of elevations in metres. No API key.
    """
    _require_gdal()
    west, south, east, north = normalize_bbox(*bbox)
    x_min, y_min, x_max, y_max = tile_range_for_bbox((west, south, east, north), zoom)
    n_x = x_max - x_min + 1
    n_y = y_max - y_min + 1
    n_tiles = n_x * n_y
    if n_tiles > max_tiles:
        raise DemDownloadError(
            f"That area needs {n_tiles} tiles at zoom {zoom}, over the {max_tiles}-tile "
            f"safety limit.\n\nEither draw a smaller area, or lower the zoom level "
            f"(each level down cuts the tile count by four).")

    tile_px = 256
    width, height = n_x * tile_px, n_y * tile_px
    log(f"Downloading {n_tiles} terrain tile(s) at zoom {zoom} "
        f"({width} x {height} pixels, ~{tile_resolution_m(zoom, (south + north) / 2):.0f} m/pixel)…")

    mosaic = np.full((height, width), np.nan, dtype=np.float64)
    done = 0
    failed = 0
    for ty in range(y_min, y_max + 1):
        for tx in range(x_min, x_max + 1):
            _check_cancelled(cancelled)
            url = TERRAIN_TILE_URL.format(z=zoom, x=tx, y=ty)
            try:
                data, _ = _http_get(url, timeout=60)
                arr = _read_png_bytes(data)
                elev = _decode_terrarium(arr)
            except DemDownloadError:
                failed += 1
                elev = None
            except Exception as e:
                log(f"  (tile {tx}/{ty} could not be decoded: {e})")
                failed += 1
                elev = None
            if elev is not None:
                r0 = (ty - y_min) * tile_px
                c0 = (tx - x_min) * tile_px
                mosaic[r0:r0 + tile_px, c0:c0 + tile_px] = elev
            done += 1
            if done % 10 == 0 or done == n_tiles:
                log(f"  {done}/{n_tiles} tiles")

    if failed == n_tiles:
        raise DemDownloadError("Every tile failed to download. Check your internet "
                               "connection or proxy settings and try again.")
    if failed:
        log(f"  WARNING: {failed} tile(s) failed and are left as nodata in the mosaic.")

    tile_span = 2 * WEB_MERCATOR_HALF_CIRCUMFERENCE / (2 ** zoom)
    pixel_size = tile_span / tile_px
    origin_x = -WEB_MERCATOR_HALF_CIRCUMFERENCE + x_min * tile_span
    origin_y = WEB_MERCATOR_HALF_CIRCUMFERENCE - y_min * tile_span
    geotransform = (origin_x, pixel_size, 0.0, origin_y, 0.0, -pixel_size)

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3857)
    _write_gtiff(out_path, mosaic, geotransform, srs.ExportToWkt(), nodata=-9999.0)
    log(f"  mosaicked -> {out_path}")
    return out_path


def _read_png_bytes(data):
    """Decode PNG bytes into a (bands, rows, cols) array using GDAL's in-memory VFS."""
    mem_path = f"/vsimem/samkit_tile_{os.getpid()}_{id(data)}.png"
    gdal.FileFromMemBuffer(mem_path, data)
    try:
        ds = gdal.Open(mem_path)
        if ds is None:
            raise DemDownloadError("GDAL could not open a downloaded tile.")
        arr = ds.ReadAsArray()
        ds = None
        if arr is None or arr.ndim != 3 or arr.shape[0] < 3:
            raise DemDownloadError("A downloaded tile was not a 3-band RGB image.")
        return arr
    finally:
        gdal.Unlink(mem_path)


# ---------------------------------------------------------------------------
# Raster writing / reprojection / unit conversion
# ---------------------------------------------------------------------------

def _write_gtiff(path, array, geotransform, projection_wkt, nodata=-9999.0):
    _require_gdal()
    rows, cols = array.shape
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(path, cols, rows, 1, gdal.GDT_Float32,
                       options=["COMPRESS=LZW", "TILED=YES"])
    ds.SetGeoTransform(geotransform)
    ds.SetProjection(projection_wkt)
    out = np.array(array, dtype=np.float64, copy=True)
    out[~np.isfinite(out)] = nodata
    band = ds.GetRasterBand(1)
    band.WriteArray(out)
    band.SetNoDataValue(nodata)
    band.FlushCache()
    ds = None
    return path


def _validate_raster(path):
    _require_gdal()
    ds = gdal.Open(path)
    if ds is None:
        raise DemDownloadError(f"The downloaded file is not a readable raster: {path}")
    if ds.RasterXSize < 2 or ds.RasterYSize < 2:
        ds = None
        raise DemDownloadError("The downloaded DEM is smaller than 2x2 pixels -- the area "
                               "of interest is probably too small for this dataset's "
                               "resolution. Draw a larger area.")
    ds = None
    return True


def describe_raster(path):
    """A short human-readable description of a raster's grid and CRS."""
    _require_gdal()
    ds = gdal.Open(path)
    if ds is None:
        raise DemDownloadError(f"Could not open raster: {path}")
    gt = ds.GetGeoTransform()
    srs = osr.SpatialReference(wkt=ds.GetProjection()) if ds.GetProjection() else None
    if srs is not None:
        try:
            crs_name = srs.GetName() or "unknown"
        except AttributeError:  # GDAL < 3.0
            crs_name = srs.GetAttrValue("PROJCS") or srs.GetAttrValue("GEOGCS") or "unknown"
        units = srs.GetAttrValue("UNIT") or "unknown"
        authority = srs.GetAuthorityCode(None)
        crs_label = f"{crs_name}" + (f" (EPSG:{authority})" if authority else "")
        is_projected = bool(srs.IsProjected())
    else:
        crs_label, units, is_projected = "undefined", "unknown", False
    info = {
        "cols": ds.RasterXSize, "rows": ds.RasterYSize,
        "pixel_x": abs(gt[1]), "pixel_y": abs(gt[5]),
        "crs": crs_label, "units": units, "projected": is_projected,
    }
    ds = None
    return info


def reproject_to_metres(src_path, out_path, target_epsg, resolution_m=None,
                        resample="bilinear", clip_bbox=None, log=print, nodata=-9999.0):
    """
    Warp `src_path` onto a metre-based grid in `target_epsg` (normally a UTM
    zone). `resolution_m` of None lets GDAL derive an appropriate square pixel
    size from the source; `clip_bbox` is an optional (west, south, east, north)
    in EPSG:4326 to trim the result to the area of interest. Passing
    `target_epsg=None` keeps the source CRS untouched, which makes this usable
    as a plain copy/clip when the user has switched reprojection off.
    """
    _require_gdal()
    resample_map = {
        "bilinear": "bilinear", "nearest": "near", "cubic": "cubic",
        "cubicspline": "cubicspline", "lanczos": "lanczos", "average": "average",
    }
    kwargs = dict(
        format="GTiff",
        resampleAlg=resample_map.get(resample, "bilinear"),
        dstNodata=nodata,
        outputType=gdal.GDT_Float32,
        creationOptions=["COMPRESS=LZW", "TILED=YES"],
        multithread=True,
    )
    if target_epsg is not None:
        kwargs["dstSRS"] = f"EPSG:{int(target_epsg)}"
    if resolution_m:
        kwargs["xRes"] = float(resolution_m)
        kwargs["yRes"] = float(resolution_m)
    elif target_epsg is not None:
        # 'Auto' resolution: GDAL may choose non-square pixels (e.g. 27.31 x 27.41 m),
        # which SAGA cannot represent and which 3x3 terrain operators treat
        # inconsistently across tools. Force square pixels at the finer of the two
        # GDAL-suggested sizes.
        try:
            probe = dict(kwargs)
            probe.pop("creationOptions", None)
            probe["format"] = "VRT"
            vrt = gdal.Warp("", src_path, options=gdal.WarpOptions(**probe))
            gt_s = vrt.GetGeoTransform()
            vrt = None
            sq = min(abs(gt_s[1]), abs(gt_s[5]))
            kwargs["xRes"] = kwargs["yRes"] = sq
            log(f"Auto resolution: using square {sq:.3f} m pixels.")
        except Exception as e:
            log(f"(could not determine a square auto resolution: {e})")
    if clip_bbox:
        west, south, east, north = normalize_bbox(*clip_bbox)
        kwargs["outputBounds"] = (west, south, east, north)
        kwargs["outputBoundsSRS"] = "EPSG:4326"

    target_label = f"EPSG:{int(target_epsg)}" if target_epsg is not None else "the source CRS"
    log(f"Warping to {target_label} "
        f"({'auto' if not resolution_m else f'{resolution_m:g} m'} pixels, {resample})…")
    result = gdal.Warp(out_path, src_path, options=gdal.WarpOptions(**kwargs))
    if result is None:
        raise DemDownloadError("GDAL could not reproject the DEM. Check that the source "
                               "raster has a defined CRS.")
    result = None
    _validate_raster(out_path)
    info = describe_raster(out_path)
    log(f"  -> {info['cols']} x {info['rows']} pixels at "
        f"{info['pixel_x']:.2f} x {info['pixel_y']:.2f} {info['units']}")
    return out_path


def scale_elevation_units(src_path, out_path, factor, log=print, nodata=-9999.0):
    """
    Multiply elevation values by `factor` -- for DEMs whose *vertical* unit is
    not metres (feet is the common case in US LiDAR products). Horizontal units
    are handled by reproject_to_metres; this is the vertical half, and getting
    it wrong inflates every slope by ~3.28x without any obvious symptom.
    """
    _require_gdal()
    if abs(factor - 1.0) < 1e-12:
        return src_path
    ds = gdal.Open(src_path)
    if ds is None:
        raise DemDownloadError(f"Could not open raster: {src_path}")
    band = ds.GetRasterBand(1)
    src_nodata = band.GetNoDataValue()
    arr = band.ReadAsArray().astype(np.float64)
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    ds = None

    mask = ~np.isfinite(arr)
    if src_nodata is not None:
        mask |= (arr == src_nodata)
    arr = arr * float(factor)
    arr[mask] = np.nan

    _write_gtiff(out_path, arr, gt, proj, nodata=nodata)
    log(f"Converted elevation values by x{factor:.6g} (vertical units -> metres).")
    return out_path
