# -*- coding: utf-8 -*-
"""
SAMKit -- publication-quality map export for prediction rasters.

Builds a QGIS print layout (QgsPrintLayout) around a prediction raster and
exports it as PNG (raster, at a chosen dpi), PDF (vector text/linework,
georeferenced) and optionally SVG. No matplotlib is needed: everything is
drawn by QGIS's own layout engine, so what is exported is exactly what QGIS
would print from its Layout Manager -- and the same layout can be registered
in the project for further hand-editing.

Every map carries the elements expected by common cartographic standards
and journal/agency map guidelines:

  * title and subtitle (what is mapped, by which model)
  * the map body inside a neatline, at a rounded representative fraction
  * a coordinate grid with labelled exterior ticks in the map's own CRS
    (eastings/northings in metres for projected grids, D-M-S for geographic)
  * a legend with the variable name and units, drawn as a continuous colour
    bar (or as classed bins with rounded breaks), plus the sample locations
  * a perceptually uniform, colour-vision-deficiency-safe colour ramp
    (viridis by default; cividis, magma, plasma, inferno, greyscale optional)
  * a graphic scale bar in metric units, and the numeric scale
  * a north arrow (true north -- it rotates with meridian convergence)
  * a locator inset showing where the study area is (when QGIS's bundled
    world map is available)
  * a metadata block: coordinate reference system (name, EPSG code,
    projection, ellipsoid), cell size, model, validation accuracy, sample
    size, covariates, author, date and software credit

Public entry points
-------------------
export_prediction_map(raster_path, out_base, ...)   one map, any raster
export_run_maps(...)                               prediction (+ kriging
                                                   variance) maps for a
                                                   SAMKit run
style_prediction_layer(layer, ...)                 apply the map styling to
                                                   a QgsRasterLayer
"""

import os
import math
import datetime
import textwrap

import numpy as np

from qgis.PyQt.QtCore import QPointF, QRectF, QSizeF, QFile, Qt
from qgis.PyQt.QtGui import QColor, QFont, QFontDatabase
from qgis.core import (
    Qgis, QgsApplication, QgsProject, QgsPrintLayout, QgsLayoutItemPage,
    QgsLayoutItemMap, QgsLayoutItemMapGrid, QgsLayoutItemLabel,
    QgsLayoutItemLegend, QgsLayoutItemScaleBar, QgsLayoutItemPicture,
    QgsLayoutPoint, QgsLayoutSize, QgsLayoutExporter, QgsLayoutItem,
    QgsUnitTypes, QgsRasterLayer, QgsVectorLayer, QgsRasterBandStats,
    QgsColorRampShader, QgsRasterShader, QgsSingleBandPseudoColorRenderer,
    QgsGradientColorRamp, QgsGradientStop, QgsLayerTree,
    QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsRectangle,
    QgsPointXY, QgsFillSymbol, QgsMarkerSymbol, QgsSymbol,
    QgsCategorizedSymbolRenderer, QgsRendererCategory, QgsSingleSymbolRenderer,
    QgsTextFormat, QgsDistanceArea, QgsLayoutMeasurement,
    QgsFeature, QgsGeometry, QgsLegendStyle, QgsLegendRenderer,
)

# ---------------------------------------------------------------------------
# Colour ramps. Sampled from the matplotlib/Crameri reference definitions so
# no extra package is needed. All are perceptually uniform and readable by
# viewers with the common forms of colour-vision deficiency, and all but
# "Greyscale" also survive greyscale printing with monotonic lightness.
# ---------------------------------------------------------------------------
COLOR_RAMPS = {
    "Viridis": ["#440154", "#482878", "#3e4989", "#31688e", "#26828e",
                "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725"],
    "Cividis": ["#00224e", "#123570", "#3b496c", "#575d6d", "#707173",
                "#8a8678", "#a59c74", "#c3b369", "#e1cc55", "#fee838"],
    "Magma": ["#000004", "#180f3d", "#440f76", "#721f81", "#9e2f7f",
              "#cd4071", "#f1605d", "#fd9668", "#feca8d", "#fcfdbf"],
    "Plasma": ["#0d0887", "#47039f", "#7301a8", "#9c179e", "#bd3786",
               "#d8576b", "#ed7953", "#fb9f3a", "#fdca26", "#f0f921"],
    "Inferno": ["#000004", "#1b0c41", "#4a0c6b", "#781c6d", "#a52c60",
                "#cf4446", "#ed6925", "#fb9b06", "#f7d13d", "#fcffa4"],
    "Greyscale": ["#f7f7f7", "#252525"],
}
DEFAULT_RAMP = "Viridis"
DEFAULT_VARIANCE_RAMP = "Magma"

PAGE_SIZES_MM = {
    "A4": (297.0, 210.0),
    "A3": (420.0, 297.0),
    "US Letter": (279.4, 215.9),
}

_MARGIN = 10.0          # page margin (mm)
_PANEL_W = 64.0         # right-hand panel width (mm)
_GAP = 6.0              # gap between map annotations and panel (mm)
_ANNO_LEFT = 13.0       # room left of the map frame for northing labels (mm)
_ANNO_BOTTOM = 7.0      # room below the map frame for easting labels (mm)
_TITLE_H = 20.0         # title + subtitle band (mm)
_SCALEBAR_H = 13.0      # room under the map for the scale bar (mm)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def plugin_version():
    """Version string from metadata.txt, so maps always credit the installed build."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metadata.txt")
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("version="):
                    return line.split("=", 1)[1].strip()
    except Exception:
        pass
    return ""


_FONT_FAMILY = None

# Layers that only a layout uses (locator basemap, marker) must outlive the
# Python function that made them, or the layout loses them once Python's
# garbage collector runs. Attributes set on SIP wrapper objects are not safe
# for this (the wrapper itself can be collected), so keep them here.
_KEEP_ALIVE = []


_RELEASE_HOOKED = False


def _keep(*objs):
    """Keep layers alive for layouts registered in the project's Layout Manager.
    Released when the project is cleared (which also deletes those layouts) and
    when the plugin unloads -- layers still alive after QGIS shuts down crash
    the Python interpreter on exit."""
    global _RELEASE_HOOKED
    _KEEP_ALIVE.extend(o for o in objs if o is not None)
    if not _RELEASE_HOOKED:
        try:
            QgsProject.instance().cleared.connect(release_resources)
            _RELEASE_HOOKED = True
        except Exception:
            pass


def release_resources():
    """Drop every layer held for registered layouts. Safe to call repeatedly."""
    del _KEEP_ALIVE[:]


def _font_family():
    """A clean sans-serif that exists on this machine (journal-friendly)."""
    global _FONT_FAMILY
    if _FONT_FAMILY is None:
        try:
            available = set(QFontDatabase().families())
        except Exception:
            available = set()
        for fam in ("Arial", "Helvetica", "Liberation Sans", "Nimbus Sans",
                    "DejaVu Sans", "Noto Sans"):
            if fam in available:
                _FONT_FAMILY = fam
                break
        else:
            _FONT_FAMILY = QFont().defaultFamily() or "Sans Serif"
    return _FONT_FAMILY


def _qfont(size_pt, bold=False, italic=False):
    f = QFont(_font_family())
    f.setPointSizeF(float(size_pt))
    f.setBold(bool(bold))
    f.setItalic(bool(italic))
    return f


def _text_format(size_pt, bold=False, italic=False, color="#1a1a1a"):
    fmt = QgsTextFormat()
    fmt.setFont(_qfont(size_pt, bold, italic))
    fmt.setSize(float(size_pt))
    try:
        fmt.setSizeUnit(QgsUnitTypes.RenderPoints)
    except Exception:
        pass
    fmt.setColor(QColor(color))
    return fmt


def _set_label_text_style(label, size_pt, bold=False, italic=False, color="#1a1a1a"):
    if hasattr(label, "setTextFormat"):               # QGIS >= 3.24
        label.setTextFormat(_text_format(size_pt, bold, italic, color))
    else:                                             # QGIS 3.16 - 3.22
        label.setFont(_qfont(size_pt, bold, italic))
        label.setFontColor(QColor(color))


def _nice_number(x, steps=(1, 2, 2.5, 5, 10)):
    """Smallest 'nice' number >= x (1, 2, 2.5, 5 x 10^k)."""
    if x <= 0 or not math.isfinite(x):
        return 1.0
    mag = 10 ** math.floor(math.log10(x))
    for s in steps:
        if s * mag >= x - 1e-12 * mag:
            return s * mag
    return 10 * mag


def _nice_scale(scale):
    """Round a map scale denominator UP to a conventional value (1:25 000, 1:50 000...)."""
    return _nice_number(scale, steps=(1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10))


def _nice_breaks(vmin, vmax, n_classes):
    """Rounded class breaks spanning [vmin, vmax] with roughly n_classes classes."""
    if not (math.isfinite(vmin) and math.isfinite(vmax)) or vmax <= vmin:
        return [vmin, vmax]
    step = _nice_number((vmax - vmin) / max(1, n_classes))
    lo = math.floor(vmin / step) * step
    hi = math.ceil(vmax / step) * step
    n = int(round((hi - lo) / step))
    return [lo + i * step for i in range(n + 1)]


def _decimals_for(span):
    """Decimals that show a value range meaningfully without false precision."""
    if span <= 0 or not math.isfinite(span):
        return 2
    return int(min(6, max(0, 2 - math.floor(math.log10(span)))))


def _fmt(v, decimals):
    return f"{v:,.{decimals}f}".replace(",", " ")   # thin-space thousands


def _gradient_ramp(name, reverse=False):
    colors = list(COLOR_RAMPS.get(name, COLOR_RAMPS[DEFAULT_RAMP]))
    if reverse:
        colors = colors[::-1]
    qcols = [QColor(c) for c in colors]
    stops = []
    n = len(qcols)
    for i in range(1, n - 1):
        stops.append(QgsGradientStop(i / (n - 1.0), qcols[i]))
    return QgsGradientColorRamp(qcols[0], qcols[-1], False, stops)


def _raster_min_max(layer):
    """Exact min/max of the valid (non-nodata) cells."""
    try:
        from osgeo import gdal
        ds = gdal.Open(layer.source().split("|")[0])
        band = ds.GetRasterBand(1)
        arr = band.ReadAsArray().astype("float64")
        nd = band.GetNoDataValue()
        if nd is not None:
            arr[arr == nd] = np.nan
        arr = arr[np.isfinite(arr)]
        if arr.size:
            return float(arr.min()), float(arr.max())
    except Exception:
        pass
    stats = layer.dataProvider().bandStatistics(1, QgsRasterBandStats.Min | QgsRasterBandStats.Max)
    return float(stats.minimumValue), float(stats.maximumValue)


# ---------------------------------------------------------------------------
# Layer styling
# ---------------------------------------------------------------------------
def style_prediction_layer(layer, ramp_name=DEFAULT_RAMP, reverse=False,
                           classification="continuous", n_classes=7,
                           vmin=None, vmax=None):
    """
    Apply a single-band pseudocolour renderer to ``layer``.

    classification: "continuous" -> smooth colour bar (legend shows min/max)
                    "classed"    -> n_classes-ish bins with rounded breaks
    Returns (vmin, vmax, decimals).
    """
    if vmin is None or vmax is None:
        dmin, dmax = _raster_min_max(layer)
        vmin = dmin if vmin is None else vmin
        vmax = dmax if vmax is None else vmax
    if not (math.isfinite(vmin) and math.isfinite(vmax)):
        raise RuntimeError("the raster has no valid cells to map")
    if vmax <= vmin:
        vmax = vmin + (abs(vmin) * 1e-6 or 1e-6)
    decimals = _decimals_for(vmax - vmin)
    ramp = _gradient_ramp(ramp_name, reverse)

    shader_fn = QgsColorRampShader(vmin, vmax)
    shader_fn.setSourceColorRamp(ramp.clone())
    items = []
    if classification == "classed":
        breaks = _nice_breaks(vmin, vmax, n_classes)
        dec = _tick_decimals(breaks)
        shader_fn.setColorRampType(QgsColorRampShader.Discrete)
        n = len(breaks) - 1
        for i in range(n):
            t = (i + 0.5) / n
            upper = breaks[i + 1] if i < n - 1 else float("inf")
            label = f"{_fmt(breaks[i], dec)} – {_fmt(breaks[i + 1], dec)}"
            items.append(QgsColorRampShader.ColorRampItem(upper, ramp.color(t), label))
        shader_fn.setMinimumValue(breaks[0])
        shader_fn.setMaximumValue(breaks[-1])
    else:
        shader_fn.setColorRampType(QgsColorRampShader.Interpolated)
        n = 32
        for i in range(n + 1):
            t = i / float(n)
            v = vmin + t * (vmax - vmin)
            items.append(QgsColorRampShader.ColorRampItem(v, ramp.color(t), _fmt(v, decimals)))
    shader_fn.setColorRampItemList(items)
    shader_fn.setClip(False)

    # Continuous colour-bar legend (QGIS >= 3.18); 3.16 falls back to entries.
    if classification != "classed" and hasattr(shader_fn, "legendSettings"):
        try:
            from qgis.core import QgsColorRampLegendNodeSettings, QgsBasicNumericFormat
            ls = QgsColorRampLegendNodeSettings()
            ls.setUseContinuousLegend(True)
            ls.setOrientation(Qt.Vertical)
            ls.setDirection(QgsColorRampLegendNodeSettings.MaximumToMinimum)
            nf = QgsBasicNumericFormat()
            nf.setNumberDecimalPlaces(decimals)
            nf.setShowTrailingZeros(True)
            nf.setShowThousandsSeparator(True)
            ls.setNumericFormat(nf)
            shader_fn.setLegendSettings(ls)
        except Exception:
            pass

    shader = QgsRasterShader()
    shader.setRasterShaderFunction(shader_fn)
    renderer = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, shader)
    renderer.setClassificationMin(shader_fn.minimumValue())
    renderer.setClassificationMax(shader_fn.maximumValue())
    layer.setRenderer(renderer)
    try:
        layer.resampleFilter().setZoomedInResampler(None)   # crisp cells, no smoothing
    except Exception:
        pass
    layer.triggerRepaint()
    return vmin, vmax, decimals


def make_sample_points_layer(points, crs, name="Sample locations", path=None):
    """
    points: iterable of (x, y, role) in the raster CRS, role in
            {"sample", "calibration", "validation"}.
    If ``path`` is given the points are written to a GeoPackage there (so an
    editable layout keeps working after the project is reopened), otherwise a
    memory layer is returned. Returns None when there are no points.
    """
    pts = [(float(x), float(y), str(r)) for x, y, r in points
           if math.isfinite(float(x)) and math.isfinite(float(y))]
    if not pts:
        return None
    layer = None
    if path:
        try:
            from osgeo import ogr, osr
            if os.path.exists(path):
                os.remove(path)
            drv = ogr.GetDriverByName("GPKG")
            ds = drv.CreateDataSource(path)
            srs = osr.SpatialReference()
            srs.ImportFromWkt(crs.toWkt())
            lyr = ds.CreateLayer("sample_locations", srs, ogr.wkbPoint)
            lyr.CreateField(ogr.FieldDefn("role", ogr.OFTString))
            for x, y, r in pts:
                f = ogr.Feature(lyr.GetLayerDefn())
                f.SetField("role", r)
                g = ogr.Geometry(ogr.wkbPoint)
                g.AddPoint_2D(x, y)
                f.SetGeometry(g)
                lyr.CreateFeature(f)
            ds = None
            layer = QgsVectorLayer(f"{path}|layername=sample_locations", name, "ogr")
            if not layer.isValid():
                layer = None
        except Exception:
            layer = None
    if layer is None:
        from qgis.core import QgsField
        from qgis.PyQt.QtCore import QVariant
        layer = QgsVectorLayer("Point", name, "memory")
        layer.setCrs(crs)
        pr = layer.dataProvider()
        pr.addAttributes([QgsField("role", QVariant.String)])
        layer.updateFields()
        feats = []
        for x, y, r in pts:
            f = QgsFeature(layer.fields())
            f.setAttribute("role", r)
            f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(x, y)))
            feats.append(f)
        pr.addFeatures(feats)
        layer.updateExtents()

    counts = {}
    for _, _, r in pts:
        counts[r] = counts.get(r, 0) + 1

    def _marker(shape, fill, size=1.9):
        return QgsMarkerSymbol.createSimple({
            "name": shape, "color": fill, "outline_color": "#000000",
            "outline_width": "0.25", "outline_width_unit": "MM",
            "size": str(size), "size_unit": "MM"})

    if set(counts) <= {"sample"}:
        sym = _marker("circle", "#ffffff")
        layer.setRenderer(QgsSingleSymbolRenderer(sym))
        layer.setName(f"Sample location (n = {counts.get('sample', 0)})")
    else:
        cats = []
        if counts.get("calibration"):
            cats.append(QgsRendererCategory(
                "calibration", _marker("circle", "#ffffff"),
                f"Calibration sample (n = {counts['calibration']})"))
        if counts.get("validation"):
            cats.append(QgsRendererCategory(
                "validation", _marker("triangle", "#000000", 2.2),
                f"Validation sample (n = {counts['validation']})"))
        if counts.get("sample"):
            cats.append(QgsRendererCategory(
                "sample", _marker("circle", "#ffffff"),
                f"Sample location (n = {counts['sample']})"))
        layer.setRenderer(QgsCategorizedSymbolRenderer("role", cats))
    return layer


# ---------------------------------------------------------------------------
# Layout construction
# ---------------------------------------------------------------------------
def _available_map_box(page_w, page_h):
    x0 = _MARGIN + _ANNO_LEFT
    y0 = _MARGIN + _TITLE_H
    w = page_w - _MARGIN - _PANEL_W - _GAP - x0
    h = page_h - _MARGIN - _SCALEBAR_H - _ANNO_BOTTOM - y0
    return x0, y0, w, h


def _fit(aspect, box_w, box_h):
    if box_w / box_h > aspect:
        return box_h * aspect, box_h
    return box_w, box_w / aspect


def _choose_orientation(page_name, extent_aspect):
    """Pick landscape or portrait -- whichever gives the larger map body."""
    long_mm, short_mm = PAGE_SIZES_MM.get(page_name, PAGE_SIZES_MM["A4"])
    best = None
    for w, h in ((long_mm, short_mm), (short_mm, long_mm)):
        _, _, bw, bh = _available_map_box(w, h)
        mw, mh = _fit(extent_aspect, bw, bh)
        if best is None or mw * mh > best[0]:
            best = (mw * mh, w, h)
    return best[1], best[2]


GRID_LABELS = {
    "dms": "Latitude / longitude (D\u00b0 M\u2032 S\u2033)",
    "projected": "Projected coordinates (m E / m N)",
}
DEFAULT_GRID_LABELS = "dms"


def _grid_interval(span_units, geographic):
    if geographic:
        cands = [1 / 3600.0 * s for s in (1, 2, 5, 10, 15, 30)] + \
                [1 / 60.0 * s for s in (1, 2, 5, 10, 15, 30)] + \
                [1, 2, 5, 10, 15, 30, 45]
        target = span_units / 4.0
        return min(cands, key=lambda c: abs(math.log(c / target)))
    return _nice_number(span_units / 5.0)


def _add_grid(map_item, crs, extent, labels=DEFAULT_GRID_LABELS):
    """
    Coordinate grid with exterior ticks and labels on the left and bottom.
    labels = "dms": a WGS 84 latitude/longitude graticule labelled D\u00b0M\u2032S\u2033
             (e.g. 39\u00b002\u203231\u2033N, 85\u00b031\u203255\u2033W), drawn over the map in
             whatever CRS the map uses (on a UTM map the graticule is slightly curved/rotated);
    labels = "projected": the map's own CRS, eastings/northings in metres
             (falls back to D\u00b0M\u2032S\u2033 if the map itself is in degrees).
    A label that would run past a corner of the neatline is left out (its tick
    is still drawn).
    """
    wgs84 = QgsCoordinateReferenceSystem("EPSG:4326")
    use_dms = labels == "dms" or crs.isGeographic()
    grid = QgsLayoutItemMapGrid("Coordinate grid", map_item)
    map_item.grids().addGrid(grid)
    grid.setUnits(QgsLayoutItemMapGrid.MapUnit)
    clear_mm = 9.0
    if use_dms:
        grid.setCrs(wgs84)
        if crs.isGeographic():
            ll = QgsRectangle(extent)
        else:
            ll = QgsCoordinateTransform(crs, wgs84, QgsProject.instance()).transformBoundingBox(extent)
        interval = _grid_interval(max(ll.width(), ll.height()), True)
        # corner clearance in degrees (from mm on paper at the map scale)
        clear_m = clear_mm * map_item.scale() / 1000.0
        lat = math.radians(ll.center().y())
        cx = clear_m / (111320.0 * max(math.cos(lat), 0.01))
        cy = clear_m / 110540.0
        grid.setIntervalX(interval)
        grid.setIntervalY(interval)
        try:
            from qgis.core import QgsExpression
            has_to_dms = QgsExpression.isFunctionName("to_dms")
        except Exception:
            has_to_dms = False
        if has_to_dms:
            grid.setAnnotationFormat(QgsLayoutItemMapGrid.CustomFormat)
            grid.setAnnotationExpression(
                f"CASE WHEN @grid_axis = 'x' AND (@grid_number < {ll.xMinimum() + cx!r} "
                f"OR @grid_number > {ll.xMaximum() - cx!r}) THEN '' "
                f"WHEN @grid_axis = 'y' AND (@grid_number < {ll.yMinimum() + cy!r} "
                f"OR @grid_number > {ll.yMaximum() - cy!r}) THEN '' "
                "ELSE to_dms(@grid_number, @grid_axis, 0, 'aligned') END")
        else:                                   # QGIS < 3.18: built-in D-M-S labels
            grid.setAnnotationFormat(QgsLayoutItemMapGrid.DegreeMinuteSecond)
            grid.setAnnotationPrecision(0)
    else:
        grid.setCrs(crs)
        interval = _grid_interval(min(extent.width(), extent.height()) * 1.25, False)
        grid.setIntervalX(interval)
        grid.setIntervalY(interval)
        clear = clear_mm * map_item.scale() / 1000.0 * QgsUnitTypes.fromUnitToUnitFactor(
            QgsUnitTypes.DistanceMeters, crs.mapUnits())
        e = extent
        grid.setAnnotationFormat(QgsLayoutItemMapGrid.CustomFormat)
        grid.setAnnotationExpression(
            f"CASE WHEN @grid_axis = 'x' AND (@grid_number < {e.xMinimum() + clear!r} "
            f"OR @grid_number > {e.xMaximum() - clear!r}) THEN '' "
            f"WHEN @grid_axis = 'y' AND (@grid_number < {e.yMinimum() + clear!r} "
            f"OR @grid_number > {e.yMaximum() - clear!r}) THEN '' "
            "ELSE format_number(@grid_number, 0) || if(@grid_axis = 'x', ' m E', ' m N') END")
    grid.setStyle(QgsLayoutItemMapGrid.Cross)
    grid.setCrossLength(1.6)
    line = grid.lineSymbol()
    if line is not None:
        line.setColor(QColor(60, 60, 60, 200))
        line.setWidth(0.15)
    grid.setFrameStyle(QgsLayoutItemMapGrid.ExteriorTicks)
    grid.setFrameWidth(1.4)
    try:
        grid.setFramePenSize(0.25)
        grid.setFramePenColor(QColor("#000000"))
    except Exception:
        pass
    grid.setAnnotationEnabled(True)
    if hasattr(grid, "setAnnotationTextFormat"):
        grid.setAnnotationTextFormat(_text_format(6.5, color="#1a1a1a"))
    else:
        grid.setAnnotationFont(_qfont(6.5))
    grid.setAnnotationFrameDistance(1.2)
    L, R, T, B = (QgsLayoutItemMapGrid.Left, QgsLayoutItemMapGrid.Right,
                  QgsLayoutItemMapGrid.Top, QgsLayoutItemMapGrid.Bottom)
    for side in (L, R, T, B):
        grid.setAnnotationPosition(QgsLayoutItemMapGrid.OutsideMapFrame, side)
    # Labels on the left and bottom only (clean, and room for the side panel);
    # ticks on all four sides.
    grid.setAnnotationDisplay(QgsLayoutItemMapGrid.ShowAll, L)
    grid.setAnnotationDisplay(QgsLayoutItemMapGrid.ShowAll, B)
    grid.setAnnotationDisplay(QgsLayoutItemMapGrid.HideAll, R)
    grid.setAnnotationDisplay(QgsLayoutItemMapGrid.HideAll, T)
    # Longitude/easting along the bottom, latitude/northing up the left only.
    try:
        grid.setAnnotationDisplay(QgsLayoutItemMapGrid.LongitudeOnly, B)
        grid.setAnnotationDisplay(QgsLayoutItemMapGrid.LatitudeOnly, L)
    except Exception:
        pass
    grid.setAnnotationDirection(QgsLayoutItemMapGrid.Vertical, L)   # reads bottom-to-top
    grid.setAnnotationDirection(QgsLayoutItemMapGrid.Horizontal, B)
    grid.samkit_dms = use_dms
    return grid


def _north_arrow_path():
    res = ":/images/north_arrows/layout_default_north_arrow.svg"
    if QFile.exists(res):
        return res
    for d in QgsApplication.svgPaths():
        p = os.path.join(d, "arrows", "NorthArrow_02.svg")
        if os.path.exists(p):
            return p
    return None


def _world_map_path():
    p = os.path.join(QgsApplication.pkgDataPath(), "resources", "data", "world_map.gpkg")
    return p if os.path.exists(p) else None


def _crs_lines(crs):
    lines = []
    desc = crs.description() or "unknown"
    auth = crs.authid()
    lines.append(f"Coordinate reference system: {desc}" + (f" ({auth})" if auth else ""))
    parts = []
    try:
        proj = crs.projectionAcronym()
        if proj:
            names = {"utm": "Universal Transverse Mercator", "tmerc": "Transverse Mercator",
                     "longlat": "none (geographic coordinates)", "lcc": "Lambert Conformal Conic",
                     "aea": "Albers Equal Area", "laea": "Lambert Azimuthal Equal Area",
                     "merc": "Mercator", "stere": "Stereographic", "sterea": "Oblique Stereographic"}
            parts.append(f"Projection: {names.get(proj, proj)}")
    except Exception:
        pass
    try:
        ell = crs.ellipsoidAcronym()
        if ell:
            parts.append(f"ellipsoid: {ell.replace('EPSG:7030', 'WGS 84')}")
    except Exception:
        pass
    if parts:
        lines.append("; ".join(parts))
    return lines


def _wrap(lines, width_mm, size_pt):
    """Word-wrap plain-text lines to a physical width (labels don't wrap by themselves)."""
    char_mm = size_pt * 0.3528 * 0.50          # average glyph width of a sans font
    n = max(20, int(width_mm / char_mm))
    out = []
    for line in lines:
        if not line:
            out.append("")
            continue
        out.extend(textwrap.wrap(line, n, subsequent_indent="   ") or [""])
    return "\n".join(out)


def build_map_layout(raster_layer, *, title, subtitle="", legend_title="",
                     points_layer=None, metadata_lines=(), page="A4",
                     project=None, layout_name="SAMKit map", locator=True,
                     colorbar=None, grid_labels=DEFAULT_GRID_LABELS):
    """
    Build and return (layout, info) for ``raster_layer`` (already styled).
    colorbar: {"svg": path, "w": mm, "h": mm} from write_colorbar_svg(), or
              None to use QGIS's own legend (classed maps).
    Nothing is exported and the layout is not registered with the project.
    """
    project = project or QgsProject.instance()
    crs = raster_layer.crs()
    extent = QgsRectangle(raster_layer.extent())
    if extent.isEmpty():
        raise RuntimeError("the raster has an empty extent")
    aspect = extent.width() / extent.height()

    page_w, page_h = _choose_orientation(page, aspect)

    layout = QgsPrintLayout(project)
    layout.initializeDefaults()
    layout.setName(layout_name)
    pg = layout.pageCollection().page(0)
    pg.setPageSize(QgsLayoutSize(page_w, page_h, QgsUnitTypes.LayoutMillimeters))

    # --- title + subtitle ---
    t = QgsLayoutItemLabel(layout)
    t.setText(title)
    _set_label_text_style(t, 15, bold=True)
    t.attemptMove(QgsLayoutPoint(_MARGIN, _MARGIN, QgsUnitTypes.LayoutMillimeters))
    t.attemptResize(QgsLayoutSize(page_w - 2 * _MARGIN, 8, QgsUnitTypes.LayoutMillimeters))
    t.setId("title")
    layout.addLayoutItem(t)
    if subtitle:
        st = QgsLayoutItemLabel(layout)
        st.setText(subtitle)
        _set_label_text_style(st, 9, color="#404040")
        st.attemptMove(QgsLayoutPoint(_MARGIN, _MARGIN + 8.5, QgsUnitTypes.LayoutMillimeters))
        st.attemptResize(QgsLayoutSize(page_w - 2 * _MARGIN, 6, QgsUnitTypes.LayoutMillimeters))
        st.setId("subtitle")
        layout.addLayoutItem(st)

    # --- main map body ---
    # Fit the raster extent into the available box, round the scale UP to a
    # conventional representative fraction, then shrink the frame to the
    # extent at that scale -- a round scale with a tight neatline.
    bx, by, bw, bh = _available_map_box(page_w, page_h)
    pad = extent.buffered(max(extent.width(), extent.height()) * 0.015)
    p_aspect = pad.width() / pad.height()
    mw, mh = _fit(p_aspect, bw, bh)
    mp = QgsLayoutItemMap(layout)
    mp.setId("main map")
    mp.attemptMove(QgsLayoutPoint(bx, by, QgsUnitTypes.LayoutMillimeters))
    mp.attemptResize(QgsLayoutSize(mw, mh, QgsUnitTypes.LayoutMillimeters))
    mp.setCrs(crs)
    layers = ([points_layer] if points_layer is not None else []) + [raster_layer]
    mp.setLayers(layers)
    mp.setKeepLayerSet(True)
    mp.setBackgroundColor(QColor("#ffffff"))
    mp.setFrameEnabled(True)                     # neatline
    mp.setFrameStrokeWidth(QgsLayoutMeasurement(0.3))
    mp.setFrameStrokeColor(QColor("#000000"))
    layout.addLayoutItem(mp)
    mp.zoomToExtent(pad)
    raw_scale = mp.scale()
    scale = _nice_scale(raw_scale)
    shrink = raw_scale / scale
    mw, mh = mw * shrink, mh * shrink
    mx = bx + (bw - mw) / 2.0                    # centred across, top-aligned
    my = by
    mp.attemptMove(QgsLayoutPoint(mx, my, QgsUnitTypes.LayoutMillimeters))
    mp.attemptResize(QgsLayoutSize(mw, mh, QgsUnitTypes.LayoutMillimeters))
    mp.zoomToExtent(pad)
    mp.setScale(scale)
    _add_grid(mp, crs, mp.extent(), labels=grid_labels)

    # --- scale bar (under the map, left-aligned with the frame) ---
    sb = QgsLayoutItemScaleBar(layout)
    sb.setId("scale bar")
    sb.setStyle("Single Box")
    sb.setLinkedMap(mp)
    map_width_m = _map_width_metres(mp, crs)
    use_km = map_width_m >= 5000
    sb.setUnits(QgsUnitTypes.DistanceKilometers if use_km else QgsUnitTypes.DistanceMeters)
    sb.setUnitLabel("km" if use_km else "m")
    sb.setMapUnitsPerScaleBarUnit(1.0)
    sb.setNumberOfSegmentsLeft(0)
    sb.setNumberOfSegments(4)
    seg_len = _nice_number((map_width_m / (1000.0 if use_km else 1.0)) / 12.0)
    sb.setSegmentSizeMode(_segment_size_fixed())
    sb.setUnitsPerSegment(seg_len)
    sb.setHeight(1.6)
    sb.setLineWidth(0.25)
    if hasattr(sb, "setTextFormat"):
        sb.setTextFormat(_text_format(7))
    else:
        sb.setFont(_qfont(7))
    sb.setLabelBarSpace(1.2)
    sb.setBoxContentSpace(0.8)
    try:
        from qgis.core import QgsBasicNumericFormat
        nf = QgsBasicNumericFormat()
        nf.setNumberDecimalPlaces(3)
        nf.setShowTrailingZeros(False)
        sb.setNumericFormat(nf)
    except Exception:
        pass
    layout.addLayoutItem(sb)
    sb.update()
    sb.attemptMove(QgsLayoutPoint(mx, my + mh + _ANNO_BOTTOM + 2.0,
                                  QgsUnitTypes.LayoutMillimeters))
    # numeric scale (representative fraction) at the right end of the same row
    ns = QgsLayoutItemLabel(layout)
    ns.setId("numeric scale")
    ns.setText(f"Scale 1 : {_fmt(scale, 0)} (at {page} print size)")
    _set_label_text_style(ns, 7, color="#404040")
    ns.setMarginX(0)
    ns.attemptResize(QgsLayoutSize(70, 4, QgsUnitTypes.LayoutMillimeters))
    ns.setHAlign(Qt.AlignRight)
    ns.setReferencePoint(QgsLayoutItem.UpperRight)
    ns.attemptMove(QgsLayoutPoint(mx + mw, my + mh + _ANNO_BOTTOM + 3.8,
                                  QgsUnitTypes.LayoutMillimeters))
    layout.addLayoutItem(ns)

    # --- right-hand panel ---
    px = page_w - _MARGIN - _PANEL_W
    py = by

    # north arrow (top right of the panel, true north)
    na_path = _north_arrow_path()
    if na_path:
        na = QgsLayoutItemPicture(layout)
        na.setId("north arrow")
        na.setPicturePath(na_path)
        na.setLinkedMap(mp)
        try:
            na.setNorthMode(QgsLayoutItemPicture.TrueNorth)
        except Exception:
            pass
        na.setResizeMode(QgsLayoutItemPicture.Zoom)
        na.attemptResize(QgsLayoutSize(10, 15, QgsUnitTypes.LayoutMillimeters))
        na.attemptMove(QgsLayoutPoint(px + _PANEL_W - 11, py, QgsUnitTypes.LayoutMillimeters))
        layout.addLayoutItem(na)
        na_label = QgsLayoutItemLabel(layout)
        na_label.setId("north arrow caption")
        na_label.setText("N")
        _set_label_text_style(na_label, 8, bold=True)
        na_label.setHAlign(Qt.AlignHCenter)
        na_label.setMarginX(0)
        na_label.setMarginY(0)
        na_label.attemptResize(QgsLayoutSize(10, 4, QgsUnitTypes.LayoutMillimeters))
        na_label.attemptMove(QgsLayoutPoint(px + _PANEL_W - 11, py + 15.5,
                                            QgsUnitTypes.LayoutMillimeters))
        layout.addLayoutItem(na_label)

    # legend title (variable + units)
    lt = QgsLayoutItemLabel(layout)
    lt.setId("legend title")
    lt.setText(legend_title or raster_layer.name())
    _set_label_text_style(lt, 9, bold=True)
    lt.setMarginX(0)
    lt.setMarginY(0)
    lt.attemptResize(QgsLayoutSize(_PANEL_W - 14, 5, QgsUnitTypes.LayoutMillimeters))
    lt.attemptMove(QgsLayoutPoint(px, py, QgsUnitTypes.LayoutMillimeters))
    layout.addLayoutItem(lt)
    y_cursor = py + 6.5

    if colorbar is not None and colorbar.get("svg"):
        # continuous colour bar with rounded tick values (vector SVG)
        cb = QgsLayoutItemPicture(layout)
        cb.setId("colour bar")
        cb.setPicturePath(colorbar["svg"])
        cb.setResizeMode(QgsLayoutItemPicture.Zoom)
        cb.attemptResize(QgsLayoutSize(colorbar["w"], colorbar["h"],
                                       QgsUnitTypes.LayoutMillimeters))
        cb.attemptMove(QgsLayoutPoint(px, y_cursor, QgsUnitTypes.LayoutMillimeters))
        layout.addLayoutItem(cb)
        y_cursor += colorbar["h"] + 4.0
    else:
        # classed: QGIS's own legend of the colour classes
        lg = _make_legend(layout, mp, raster_layer, symbol=(7.0, 4.0))
        lg.setId("class legend")
        lg.attemptMove(QgsLayoutPoint(px, y_cursor, QgsUnitTypes.LayoutMillimeters))
        y_cursor += _legend_height(layout, lg) + 4.0

    if points_layer is not None:
        plg = _make_legend(layout, mp, points_layer, symbol=(4.0, 3.2))
        plg.setId("sample legend")
        plg.attemptMove(QgsLayoutPoint(px, y_cursor, QgsUnitTypes.LayoutMillimeters))
        y_cursor += _legend_height(layout, plg) + 3.0
    y_cursor = max(y_cursor, py + 21.0) + 4.0

    # metadata block, anchored to the bottom of the panel
    meta_text = _wrap(list(metadata_lines), _PANEL_W, 6.5)
    md = QgsLayoutItemLabel(layout)
    md.setId("metadata")
    md.setText(meta_text)
    _set_label_text_style(md, 6.5, color="#303030")
    md.setMarginX(0)
    md.setMarginY(0)
    n_lines = meta_text.count("\n") + 1
    md_h = n_lines * 6.5 * 0.3528 * 1.22 + 1.0
    md.attemptResize(QgsLayoutSize(_PANEL_W, md_h, QgsUnitTypes.LayoutMillimeters))
    md.setReferencePoint(QgsLayoutItem.LowerLeft)
    md.attemptMove(QgsLayoutPoint(px, page_h - _MARGIN, QgsUnitTypes.LayoutMillimeters))
    layout.addLayoutItem(md)
    md_top = page_h - _MARGIN - md_h

    # locator inset in the space left between the legends and the metadata
    inset_layers = []
    avail = md_top - 5.0 - y_cursor
    wm = _world_map_path() if locator else None
    if wm and avail >= 26:
        inset_h = min(avail, _PANEL_W * 0.66)
        inset_layers = _add_locator(layout, project, wm, crs, extent, px, y_cursor,
                                    _PANEL_W, inset_h) or []

    info = {"page_w": page_w, "page_h": page_h, "scale": scale,
            "locator": bool(inset_layers), "aux_layers": inset_layers, "map_item": mp}
    return layout, info


def _map_width_metres(map_item, crs):
    ext = map_item.extent()
    if not crs.isGeographic():
        f = QgsUnitTypes.fromUnitToUnitFactor(crs.mapUnits(), QgsUnitTypes.DistanceMeters)
        return ext.width() * f
    da = QgsDistanceArea()
    da.setSourceCrs(crs, QgsProject.instance().transformContext())
    da.setEllipsoid("WGS84")
    ym = ext.center().y()
    return da.measureLine(QgsPointXY(ext.xMinimum(), ym), QgsPointXY(ext.xMaximum(), ym))


def _segment_size_fixed():
    try:
        from qgis.core import QgsScaleBarSettings
        return QgsScaleBarSettings.SegmentSizeFixed
    except Exception:
        return 0


def _make_legend(layout, map_item, layer, symbol=(4.0, 3.2)):
    """A legend item showing only ``layer``'s entries (its own title hidden)."""
    lg = QgsLayoutItemLegend(layout)
    lg.setTitle("")
    lg.setLinkedMap(map_item)
    # setAutoUpdateModel(False) gives the legend its own private layer tree,
    # owned and freed by the legend item in C++. Edit that tree in place.
    # (A QgsLayerTree created in Python and handed to setRootGroup() is freed
    # by Python's garbage collector while the legend still points at it; the
    # next layer rename then crashes QGIS with an access violation.)
    lg.setAutoUpdateModel(False)
    root = lg.model().rootGroup()
    root.removeAllChildren()
    node = root.addLayer(layer)
    try:
        QgsLegendRenderer.setNodeLegendStyle(node, QgsLegendStyle.Hidden)
    except Exception:
        pass
    _strip_band_name_node(lg, node)
    _style_legend(lg)
    lg.setSymbolWidth(symbol[0])
    lg.setSymbolHeight(symbol[1])
    layout.addLayoutItem(lg)
    return lg


def _legend_height(layout, legend):
    """Rendered height (mm) of a legend -- the item only knows it after painting."""
    try:
        from qgis.PyQt.QtGui import QImage, QPainter
        from qgis.core import QgsLayoutUtils
        img = QImage(8, 8, QImage.Format_ARGB32)
        painter = QPainter(img)
        try:
            ctx = QgsLayoutUtils.createRenderContextForLayout(layout, painter)
            size = QgsLegendRenderer(legend.model(), legend.legendSettings()).minimumSize(ctx)
        finally:
            painter.end()
        return float(size.height())
    except Exception:
        return 12.0


def _tick_decimals(ticks):
    for d in range(0, 7):
        if all(abs(round(t, d) - t) < 1e-9 * max(1.0, abs(t)) for t in ticks):
            return d
    return 6


def write_colorbar_svg(path, colors, vmin, vmax, bar_h=38.0, bar_w=4.5, font_pt=7.0):
    """
    Vertical colour bar (maximum at the top) with rounded, evenly spaced tick
    values -- the style journals expect, and which QGIS's own continuous
    legend (min/max labels only) doesn't provide. Written as SVG so it stays
    vector in the PDF. Returns {"svg": path, "w": mm, "h": mm}.
    """
    tol = 1e-9 * max(abs(vmin), abs(vmax), 1e-300)
    ticks = []
    for target in (5, 7, 10):                  # aim for 4-7 labelled ticks
        ticks = [t for t in _nice_breaks(vmin, vmax, target)
                 if vmin - tol <= t <= vmax + tol]
        if len(ticks) >= 4:
            break
    if len(ticks) < 2:
        ticks = [vmin, vmax]
    dec = _tick_decimals(ticks) if len(ticks) > 2 or ticks != [vmin, vmax] else _decimals_for(vmax - vmin)
    labels = [_fmt(t, dec) for t in ticks]
    fs = font_pt * 0.3528                                   # font size in mm
    pad = fs * 0.75
    label_x = bar_w + 1.6
    w = label_x + max(len(l) for l in labels) * fs * 0.58 + 1.0
    h = bar_h + 2 * pad
    stops = "".join(
        f'<stop offset="{i / (len(colors) - 1.0):.4f}" stop-color="{c}"/>'
        for i, c in enumerate(colors))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="{w:.3f}mm" '
        f'height="{h:.3f}mm" viewBox="0 0 {w:.3f} {h:.3f}">',
        f'<defs><linearGradient id="ramp" x1="0" y1="1" x2="0" y2="0">{stops}'
        '</linearGradient></defs>',
        f'<rect x="0" y="{pad:.3f}" width="{bar_w:.3f}" height="{bar_h:.3f}" '
        'fill="url(#ramp)" stroke="#333333" stroke-width="0.15"/>',
    ]
    span = (vmax - vmin) or 1.0
    for t, lab in zip(ticks, labels):
        yy = pad + bar_h * (1.0 - (t - vmin) / span)
        parts.append(f'<line x1="{bar_w - 0.9:.3f}" y1="{yy:.3f}" x2="{bar_w + 0.9:.3f}" '
                     f'y2="{yy:.3f}" stroke="#1a1a1a" stroke-width="0.18"/>')
        parts.append(f'<text x="{label_x:.3f}" y="{yy + fs * 0.35:.3f}" '
                     f'font-family="{_font_family()}" font-size="{fs:.3f}" '
                     f'fill="#1a1a1a">{lab}</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    return {"svg": path, "w": w, "h": h}


def _is_continuous(layer):
    try:
        fn = layer.renderer().shader().rasterShaderFunction()
        return fn.colorRampType() == QgsColorRampShader.Interpolated
    except Exception:
        return False


def _strip_band_name_node(legend, node):
    """Hide the 'Band 1' caption QGIS puts above a raster's legend entries."""
    try:
        from qgis.core import QgsMapLayerLegendUtils
        model = legend.model()
        nodes = model.layerLegendNodes(node, True)
        keep = []
        for i, n in enumerate(nodes):
            txt = str(n.data(Qt.DisplayRole) or "")
            if txt.lower().startswith("band ") and n.__class__.__name__ == "QgsSimpleLegendNode":
                continue
            keep.append(i)
        if len(keep) != len(nodes):
            QgsMapLayerLegendUtils.setLegendNodeOrder(node, keep)
            model.refreshLayerLegend(node)
    except Exception:
        pass


def _style_legend(lg):
    specs = {
        QgsLegendStyle.Title: (9, True),
        QgsLegendStyle.Group: (8, True),
        QgsLegendStyle.Subgroup: (8, True),
        QgsLegendStyle.SymbolLabel: (7.5, False),
    }
    for style, (size, bold) in specs.items():
        try:
            st = lg.rstyle(style)
            if hasattr(st, "setTextFormat"):        # QGIS >= 3.30
                st.setTextFormat(_text_format(size, bold))
            else:
                st.setFont(_qfont(size, bold))
        except Exception:
            try:
                lg.setStyleFont(style, _qfont(size, bold))
            except Exception:
                pass
    try:
        lg.rstyle(QgsLegendStyle.Subgroup).setMargin(QgsLegendStyle.Bottom, 2.0)
        lg.rstyle(QgsLegendStyle.Subgroup).setMargin(QgsLegendStyle.Top, 3.5)
        lg.rstyle(QgsLegendStyle.Symbol).setMargin(QgsLegendStyle.Top, 1.8)
    except Exception:
        pass
    lg.setBoxSpace(0.0)
    lg.setBackgroundEnabled(False)
    lg.setFrameEnabled(False)
    try:
        lg.setWrapString("")
    except Exception:
        pass


def _add_locator(layout, project, wm_path, crs, extent, x, y, w, h):
    land = QgsVectorLayer(f"{wm_path}|layername=countries", "land", "ogr")
    if not land.isValid():
        return []
    land.setRenderer(QgsSingleSymbolRenderer(QgsFillSymbol.createSimple({
        "color": "#ececec", "outline_style": "no"})))
    borders = QgsVectorLayer(f"{wm_path}|layername=countries", "country borders", "ogr")
    borders.setRenderer(QgsSingleSymbolRenderer(QgsFillSymbol.createSimple({
        "style": "no", "outline_color": "#8c8c8c", "outline_width": "0.15",
        "outline_width_unit": "MM"})))
    wgs84 = QgsCoordinateReferenceSystem("EPSG:4326")
    try:
        tr = QgsCoordinateTransform(crs, wgs84, project)
        ext_ll = tr.transformBoundingBox(extent)
    except Exception:
        return []
    c = ext_ll.center()
    span = max(ext_ll.width(), ext_ll.height(), 1e-6)
    view_h = min(170.0, max(12.0, span * 12.0))   # at least ~12 degrees of context
    view_w = view_h * (w / h)
    cx = min(max(c.x(), -180 + view_w / 2), 180 - view_w / 2) if view_w < 360 else 0.0
    cy = min(max(c.y(), -90 + view_h / 2), 90 - view_h / 2)
    view = QgsRectangle(cx - view_w / 2, cy - view_h / 2, cx + view_w / 2, cy + view_h / 2)

    # State/province borders, thinner and lighter, once the view is regional
    # enough for them to read as detail rather than clutter.
    states = None
    if view_h <= 60:
        states = QgsVectorLayer(f"{wm_path}|layername=states_provinces", "state borders", "ogr")
        if states.isValid():
            states.setRenderer(QgsSingleSymbolRenderer(QgsFillSymbol.createSimple({
                "style": "no", "outline_color": "#d0d0d0", "outline_width": "0.06",
                "outline_width_unit": "MM"})))
        else:
            states = None

    marker_layer = QgsVectorLayer("Point?crs=EPSG:4326", "study area", "memory")
    f = QgsFeature()
    f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(c.x(), c.y())))
    marker_layer.dataProvider().addFeatures([f])
    marker_layer.setRenderer(QgsSingleSymbolRenderer(QgsMarkerSymbol.createSimple({
        "name": "circle", "color": "#d7191c", "outline_color": "#ffffff",
        "outline_width": "0.35", "outline_width_unit": "MM", "size": "2.4", "size_unit": "MM"})))

    lab = QgsLayoutItemLabel(layout)
    lab.setId("locator caption")
    lab.setText("Location of the study area")
    _set_label_text_style(lab, 7, italic=True, color="#404040")
    lab.attemptMove(QgsLayoutPoint(x, y, QgsUnitTypes.LayoutMillimeters))
    lab.attemptResize(QgsLayoutSize(w, 4, QgsUnitTypes.LayoutMillimeters))
    lab.setMarginX(0)
    layout.addLayoutItem(lab)

    inset = QgsLayoutItemMap(layout)
    inset.setId("locator map")
    inset.attemptMove(QgsLayoutPoint(x, y + 4.5, QgsUnitTypes.LayoutMillimeters))
    inset.attemptResize(QgsLayoutSize(w, h - 4.5, QgsUnitTypes.LayoutMillimeters))
    inset.setCrs(wgs84)
    layers = [marker_layer, borders] + ([states] if states is not None else []) + [land]
    inset.setLayers(layers)
    inset.setKeepLayerSet(True)
    inset.setBackgroundColor(QColor("#e6eff6"))        # water
    inset.setFrameEnabled(True)
    inset.setFrameStrokeWidth(QgsLayoutMeasurement(0.2))
    inset.setFrameStrokeColor(QColor("#606060"))
    layout.addLayoutItem(inset)
    inset.zoomToExtent(view)
    return layers


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def export_layout(layout, out_base, formats=("png", "pdf"), dpi=300):
    """Export ``layout`` to out_base.<ext> for each format. Returns written paths."""
    written = []
    exporter = QgsLayoutExporter(layout)
    for fmt in formats:
        fmt = fmt.lower()
        path = f"{out_base}.{fmt}"
        if fmt == "png":
            s = QgsLayoutExporter.ImageExportSettings()
            s.dpi = float(dpi)
            s.generateWorldFile = False
            res = exporter.exportToImage(path, s)
        elif fmt == "pdf":
            s = QgsLayoutExporter.PdfExportSettings()
            s.dpi = float(dpi)
            s.rasterizeWholeImage = False
            s.forceVectorOutput = False
            s.exportMetadata = True
            try:
                s.appendGeoreference = True
            except Exception:
                pass
            try:
                s.simplifyGeometries = False
            except Exception:
                pass
            res = exporter.exportToPdf(path, s)
        elif fmt == "svg":
            s = QgsLayoutExporter.SvgExportSettings()
            s.dpi = float(dpi)
            s.forceVectorOutput = False
            s.exportAsLayers = False
            try:
                s.simplifyGeometries = False
            except Exception:
                pass
            res = exporter.exportToSvg(path, s)
        else:
            raise ValueError(f"unsupported map format: {fmt}")
        if res != QgsLayoutExporter.Success:
            raise RuntimeError(f"QGIS could not export {path} (code {res}): "
                               f"{exporter.errorMessage() if hasattr(exporter, 'errorMessage') else ''}")
        written.append(path)
    return written


def export_prediction_map(raster_path, out_base, *, title=None, subtitle="",
                          variable="", units="", points=None, points_gpkg=None,
                          metadata_lines=(), ramp=DEFAULT_RAMP, reverse_ramp=False,
                          classification="continuous", n_classes=7, page="A4",
                          dpi=300, formats=("png", "pdf"), author="",
                          raster_layer=None, points_layer=None, locator=True,
                          register_layout=False, layout_name=None, project=None,
                          grid_labels=DEFAULT_GRID_LABELS):
    """
    Export one publication-quality map of a single-band raster.

    raster_path     GeoTIFF to map (ignored when raster_layer is given)
    out_base        output path without extension ("…/prediction_map")
    points          optional [(x, y, role), ...] in the raster CRS
    raster_layer    optional pre-loaded QgsRasterLayer to map (it is restyled)
    register_layout add the layout to the project's Layout Manager for editing

    Returns {"files": [...], "layout": QgsPrintLayout, "layers": (...), "scale": float}.
    """
    project = project or QgsProject.instance()
    if raster_layer is None:
        raster_layer = QgsRasterLayer(raster_path, os.path.splitext(os.path.basename(raster_path))[0])
    if not raster_layer.isValid():
        raise RuntimeError(f"could not open raster for mapping: {raster_path}")
    if not raster_layer.crs().isValid():
        raise RuntimeError("the raster has no coordinate reference system -- "
                           "a map without one cannot carry a grid or scale bar")
    vmin, vmax, dec = style_prediction_layer(raster_layer, ramp, reverse_ramp,
                                             classification, n_classes)
    if points_layer is None and points:
        points_layer = make_sample_points_layer(points, raster_layer.crs(), path=points_gpkg)

    res_dir = os.path.join(os.path.dirname(os.path.abspath(out_base)), "layout_resources")
    os.makedirs(res_dir, exist_ok=True)
    colorbar = None
    if classification != "classed":
        cols = list(COLOR_RAMPS.get(ramp, COLOR_RAMPS[DEFAULT_RAMP]))
        if reverse_ramp:
            cols = cols[::-1]
        colorbar = write_colorbar_svg(
            os.path.join(res_dir, os.path.basename(out_base) + "_colour_bar.svg"),
            cols, vmin, vmax)

    var = variable or raster_layer.name()
    legend_title = f"{var} ({units})" if units else var
    title = title or f"Predicted {var}"

    lines = list(metadata_lines)
    res_x = raster_layer.rasterUnitsPerPixelX()
    res_y = raster_layer.rasterUnitsPerPixelY()
    crs = raster_layer.crs()
    if crs.isGeographic():
        lines.append(f"Cell size: {res_x * 3600:.3g}″ × {res_y * 3600:.3g}″")
    else:
        u = QgsUnitTypes.toAbbreviatedString(crs.mapUnits())
        lines.append(f"Cell size: {res_x:.4g} {u} × {res_y:.4g} {u}")
    lines.append(f"Mapped range: {_fmt(vmin, dec)} – {_fmt(vmax, dec)}"
                 + (f" {units}" if units else ""))
    lines.extend(_crs_lines(crs))
    if grid_labels == "dms" and not crs.isGeographic():
        lines.append("Grid labels: WGS 84 latitude/longitude graticule (EPSG:4326).")
    who = f"Map: {author}.  " if author else ""
    ver = plugin_version()
    lines.append(f"{who}Date: {datetime.date.today().isoformat()}.  "
                 f"Produced with SAMKit{(' ' + ver) if ver else ''} and QGIS {Qgis.QGIS_VERSION.split('-')[0]}.")

    name = layout_name or f"SAMKit - {os.path.basename(out_base)}"
    layout, info = build_map_layout(raster_layer, title=title, subtitle=subtitle,
                                    legend_title=legend_title, points_layer=points_layer,
                                    metadata_lines=lines, page=page, project=project,
                                    layout_name=name, locator=locator, colorbar=colorbar,
                                    grid_labels=grid_labels)
    os.makedirs(os.path.dirname(os.path.abspath(out_base)), exist_ok=True)
    files = export_layout(layout, out_base, formats, dpi)

    if register_layout:
        # the layout outlives this function: keep the layers only it uses
        _keep(raster_layer, points_layer, *info.get("aux_layers", []))
        mgr = project.layoutManager()
        old = mgr.layoutByName(name)
        if old is not None:
            mgr.removeLayout(old)
        mgr.addLayout(layout)
    return {"files": files, "layout": layout, "layers": (raster_layer, points_layer),
            "scale": info["scale"], "range": (vmin, vmax)}


class _ExportedFiles(list):
    """A list of written paths that can also carry the layers it used."""
    points_layer = None


def export_run_maps(pred_path, maps_dir, *, variable, units="", model_name="",
                    validation_label="", metrics=None, n_samples=None,
                    feature_names=(), points=None, variance_path=None,
                    ramp=DEFAULT_RAMP, reverse_ramp=False, classification="continuous",
                    page="A4", dpi=300, formats=("png", "pdf"), author="",
                    title=None, pred_layer=None, var_layer=None,
                    register_layouts=False, log=print, grid_labels=DEFAULT_GRID_LABELS):
    """
    Maps for one SAMKit modelling run: the prediction and, for kriging models,
    the kriging variance. ``points`` is [(x, y, role), ...] in the raster CRS.
    Returns a list of written file paths; its ``points_layer`` attribute is
    the sample-location layer the maps draw (None if no points), so a caller
    that registers the layouts can add that layer to the project as well.
    """
    os.makedirs(maps_dir, exist_ok=True)
    common = []
    if model_name:
        common.append(f"Model: {model_name}")
    if validation_label:
        common.append(f"Validation: {validation_label}")
    if metrics:
        r2 = metrics.get("r2", float("nan"))
        common.append(
            "Accuracy on held-out points: "
            f"R² = {r2:.2f}, RMSE = {metrics.get('rmse', float('nan')):.3g}"
            + (f" {units}" if units else "")
            + f", MAE = {metrics.get('mae', float('nan')):.3g}"
            + f", Lin's CCC = {metrics.get('ccc', float('nan')):.2f}")
    if n_samples:
        common.append(f"Sample points: n = {n_samples}")
    if feature_names:
        common.append("Covariates: " + ", ".join(feature_names))

    base = os.path.splitext(os.path.basename(pred_path))[0]
    pts_gpkg = (os.path.join(maps_dir, "layout_resources", "sample_locations.gpkg")
                if points else None)
    if pts_gpkg:
        os.makedirs(os.path.dirname(pts_gpkg), exist_ok=True)
    pts_layer = None
    if points:
        crs_src = pred_layer.crs() if pred_layer is not None else QgsRasterLayer(pred_path, "tmp").crs()
        pts_layer = make_sample_points_layer(points, crs_src, name=f"{base} sample locations",
                                             path=pts_gpkg)

    written = _ExportedFiles()
    written.points_layer = pts_layer
    subtitle = " · ".join(p for p in (model_name, validation_label) if p)
    res = export_prediction_map(
        pred_path, os.path.join(maps_dir, f"{base}_map"),
        title=title or f"Predicted {variable}", subtitle=subtitle, variable=variable,
        units=units, points_layer=pts_layer, metadata_lines=common, ramp=ramp,
        reverse_ramp=reverse_ramp, classification=classification, page=page, dpi=dpi,
        formats=formats, author=author, raster_layer=pred_layer,
        register_layout=register_layouts, layout_name=f"SAMKit - {base} - prediction map",
        grid_labels=grid_labels)
    written += res["files"]
    log(f"Wrote prediction map ({', '.join(f.upper() for f in formats)}, {dpi} dpi, "
        f"scale 1:{res['scale']:,.0f}) to: {res['files'][0]}")

    if variance_path and os.path.exists(variance_path):
        vunits = f"{units}²" if units else ""
        vres = export_prediction_map(
            variance_path, os.path.join(maps_dir, f"{base}_kriging_variance_map"),
            title=f"Kriging variance of predicted {variable}",
            subtitle=(subtitle + " · " if subtitle else "") + "prediction uncertainty",
            variable="Kriging variance", units=vunits, points_layer=pts_layer,
            metadata_lines=common + ["Low variance = more reliable prediction; variance "
                                     "grows with distance from the sample points."]
            + (["For regression kriging this is the residual-kriging variance only "
                "(the trend model's own uncertainty is not included)."]
               if "Regression" in (model_name or "") else []),
            ramp=DEFAULT_VARIANCE_RAMP, reverse_ramp=True, classification=classification,
            page=page, dpi=dpi, formats=formats, author=author, raster_layer=var_layer,
            register_layout=register_layouts,
            layout_name=f"SAMKit - {base} - kriging variance map",
            grid_labels=grid_labels)
        written += vres["files"]
        log(f"Wrote kriging variance map to: {vres['files'][0]}")
    return written
