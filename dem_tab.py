# -*- coding: utf-8 -*-
"""
dem_tab.py
Tab 1: get a DEM for an area of interest -- either downloaded from a global
open dataset, or an existing raster already loaded in QGIS -- and convert it
from "raw" geographic coordinates (degrees) into a projected, metre-based UTM
grid ready for the covariate calculations in the next tab.

The area of interest can be drawn directly on the map canvas, taken from any
loaded layer's extent, taken from the current canvas view, or typed in as a
bounding box. The UTM zone is detected automatically from the area's centroid.
"""

import os
import traceback

from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QCheckBox,
    QPushButton, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox, QPlainTextEdit,
    QProgressBar, QFileDialog, QMessageBox, QComboBox, QStackedWidget, QRadioButton,
    QButtonGroup
)
from qgis.PyQt.QtCore import QThread, pyqtSignal, QSettings, QUrl
from qgis.PyQt.QtGui import QDesktopServices
from qgis.gui import QgsMapLayerComboBox
from qgis.core import (
    QgsMapLayerProxyModel, QgsProject, QgsRasterLayer, QgsCoordinateTransform,
    QgsCoordinateReferenceSystem, QgsRectangle
)

from . import dem_download
from . import map_tools

WGS84 = "EPSG:4326"

SOURCE_OPENTOPO = 0
SOURCE_TERRAIN_TILES = 1
SOURCE_EXISTING_LAYER = 2


class _DemWorker(QThread):
    """
    Downloads and reprojects in a background thread. Doing this on the GUI
    thread would freeze QGIS for the whole transfer -- on a slow connection
    long enough that the window manager paints it as "Not Responding" and
    users force-quit mid-write.
    """

    message = pyqtSignal(str)
    progress = pyqtSignal(int)
    finished_ok = pyqtSignal(bool, str, str)  # ok, final_path, error_message

    def __init__(self, job, parent=None):
        super().__init__(parent)
        self._job = job
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def _is_cancelled(self):
        return self._cancel

    def run(self):
        job = self._job
        log = lambda m: self.message.emit(str(m))
        try:
            raw_path = job["raw_path"]

            if job["source"] == SOURCE_OPENTOPO:
                self.progress.emit(10)
                dem_download.download_opentopography(
                    job["bbox"], job["demtype"], job["api_key"], raw_path,
                    log=log, cancelled=self._is_cancelled)
            elif job["source"] == SOURCE_TERRAIN_TILES:
                self.progress.emit(10)
                dem_download.download_terrain_tiles(
                    job["bbox"], job["zoom"], raw_path,
                    log=log, cancelled=self._is_cancelled)
            else:
                raw_path = job["existing_path"]
                log(f"Using existing raster: {raw_path}")

            self.progress.emit(55)
            if self._cancel:
                raise dem_download.Cancelled("Cancelled by user.")

            info = dem_download.describe_raster(raw_path)
            log(f"Source grid: {info['cols']} x {info['rows']} pixels, "
                f"{info['pixel_x']:.6g} x {info['pixel_y']:.6g} {info['units']}, "
                f"CRS: {info['crs']}")
            if not info["projected"]:
                log("Source is in geographic coordinates (degrees) -- converting to "
                    "metres is required before computing slope or any other covariate.")

            current = raw_path
            factor = job["vertical_factor"]
            if abs(factor - 1.0) > 1e-12:
                scaled = job["scaled_path"]
                dem_download.scale_elevation_units(current, scaled, factor, log=log)
                current = scaled

            self.progress.emit(70)
            if self._cancel:
                raise dem_download.Cancelled("Cancelled by user.")

            if job["reproject"]:
                dem_download.reproject_to_metres(
                    current, job["out_path"], job["target_epsg"],
                    resolution_m=job["resolution"], resample=job["resample"],
                    clip_bbox=job["clip_bbox"], log=log)
            else:
                # Keep the source CRS untouched -- only copy (and optionally clip).
                log("Reprojection is switched off: keeping the source CRS as-is.")
                dem_download.reproject_to_metres(
                    current, job["out_path"], None,
                    resolution_m=job["resolution"], resample=job["resample"],
                    clip_bbox=job["clip_bbox"], log=log)

            self.progress.emit(95)
            final_info = dem_download.describe_raster(job["out_path"])
            log("")
            log(f"Final DEM: {final_info['cols']} x {final_info['rows']} pixels at "
                f"{final_info['pixel_x']:.3f} x {final_info['pixel_y']:.3f} "
                f"{final_info['units']}")
            log(f"CRS: {final_info['crs']}")
            self.progress.emit(100)
            self.finished_ok.emit(True, job["out_path"], "")
        except dem_download.Cancelled:
            self.finished_ok.emit(False, "", "Cancelled.")
        except Exception as e:
            self.message.emit(traceback.format_exc())
            self.finished_ok.emit(False, "", str(e))


class DemTab(QWidget):
    """Download or import a DEM, then project it onto a metre-based UTM grid."""

    demReady = pyqtSignal(object)  # emits the loaded QgsRasterLayer

    def __init__(self, iface, parent=None):
        super().__init__(parent)
        self.iface = iface
        self._bbox = None            # (west, south, east, north) in EPSG:4326
        self._rect_map_tool = None
        self._prev_map_tool = None
        self._worker = None
        self._build_ui()
        self._restore_api_key()
        self._on_bbox_edited()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout(self)

        intro = QLabel(
            "Download a DEM for your study area, or convert a DEM you already have, onto a "
            "metre-based (UTM) grid. Slope, curvature, TWI and MRVBF all divide elevation "
            "differences by a "
            "horizontal distance, so a DEM whose coordinates are in degrees produces "
            "covariates that are wrong by a latitude-dependent factor. Projecting to "
            "UTM first is a prerequisite, not a formality."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #444; font-size: 11px;")
        layout.addWidget(intro)

        # --- What do you want to do? ---
        mode_box = QGroupBox("What do you want to do?")
        mode_layout = QVBoxLayout(mode_box)
        self.mode_download_rb = QRadioButton(
            "Download a DEM for an area (OpenTopography or AWS Terrain Tiles)")
        self.mode_convert_rb = QRadioButton(
            "Convert my own DEM to UTM (no download) -- a DEM file on my computer or a "
            "raster already loaded in QGIS")
        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self.mode_download_rb)
        self._mode_group.addButton(self.mode_convert_rb)
        mode_layout.addWidget(self.mode_download_rb)
        mode_layout.addWidget(self.mode_convert_rb)
        layout.addWidget(mode_box)

        # --- Your DEM (convert mode) ---
        self.own_box = QGroupBox("1. Your DEM")
        own_form = QFormLayout(self.own_box)
        file_row = QHBoxLayout()
        self.own_path_edit = QLineEdit()
        self.own_path_edit.setReadOnly(True)
        self.own_path_edit.setPlaceholderText("click 'Browse…' to open a DEM file "
                                              "(GeoTIFF, IMG, ASCII grid, VRT, HGT, …)")
        file_row.addWidget(self.own_path_edit, 1)
        browse_dem_btn = QPushButton("Browse…")
        browse_dem_btn.setToolTip("Open a DEM from disk. It is added to QGIS and selected.")
        browse_dem_btn.clicked.connect(self._browse_own_dem)
        file_row.addWidget(browse_dem_btn)
        own_form.addRow("DEM file:", file_row)
        self.existing_combo = QgsMapLayerComboBox()
        self.existing_combo.setFilters(QgsMapLayerProxyModel.RasterLayer)
        self.existing_combo.layerChanged.connect(self._on_own_layer_changed)
        own_form.addRow("or a raster in QGIS:", self.existing_combo)
        own_note = QLabel(
            "Nothing is downloaded. Your DEM is converted onto a metre-based UTM grid (zone "
            "detected from the DEM itself), with its elevations converted to metres if they "
            "are in feet or centimetres (section 2). Use this for a DEM in degrees, one with "
            "elevations in feet, or any LiDAR/agency DEM you want on a clean square-metre grid. "
            "No area of interest is needed; tick 'Clip the result to an area of interest' "
            "below only if you want to cut it.")
        own_note.setWordWrap(True)
        own_note.setStyleSheet("color: gray; font-size: 10px;")
        own_form.addRow("", own_note)
        layout.addWidget(self.own_box)

        # --- Area of interest ---
        aoi_box = QGroupBox("1. Area of interest")
        self.aoi_box = aoi_box
        aoi_layout = QVBoxLayout(aoi_box)

        btn_row = QHBoxLayout()
        self.draw_btn = QPushButton("Draw rectangle on map…")
        self.draw_btn.clicked.connect(self._on_draw_rectangle)
        btn_row.addWidget(self.draw_btn)
        self.canvas_extent_btn = QPushButton("Use current map view")
        self.canvas_extent_btn.clicked.connect(self._on_use_canvas_extent)
        btn_row.addWidget(self.canvas_extent_btn)
        btn_row.addStretch()
        aoi_layout.addLayout(btn_row)

        layer_row = QHBoxLayout()
        layer_row.addWidget(QLabel("or use the extent of a layer:"))
        self.extent_layer_combo = QgsMapLayerComboBox()
        self.extent_layer_combo.setFilters(QgsMapLayerProxyModel.All)
        self.extent_layer_combo.setAllowEmptyLayer(True)
        layer_row.addWidget(self.extent_layer_combo)
        self.use_layer_btn = QPushButton("Use layer extent")
        self.use_layer_btn.clicked.connect(self._on_use_layer_extent)
        layer_row.addWidget(self.use_layer_btn)
        aoi_layout.addLayout(layer_row)

        bbox_form = QFormLayout()
        self.west_spin = self._coord_spin(-180, 180, 0.0)
        self.south_spin = self._coord_spin(-90, 90, 0.0)
        self.east_spin = self._coord_spin(-180, 180, 0.0)
        self.north_spin = self._coord_spin(-90, 90, 0.0)
        for w in (self.west_spin, self.south_spin, self.east_spin, self.north_spin):
            w.valueChanged.connect(self._on_bbox_edited)
        wr = QHBoxLayout()
        for label, spin in (("W", self.west_spin), ("S", self.south_spin),
                            ("E", self.east_spin), ("N", self.north_spin)):
            wr.addWidget(QLabel(label))
            wr.addWidget(spin)
        bbox_form.addRow("Bounding box (lon/lat, WGS84):", wr)
        aoi_layout.addLayout(bbox_form)

        self.aoi_label = QLabel("no area selected yet")
        self.aoi_label.setWordWrap(True)
        self.aoi_label.setStyleSheet("color: #205020; font-size: 11px;")
        aoi_layout.addWidget(self.aoi_label)

        aoi_note = QLabel(
            "Click 'Draw rectangle on map…' then click-drag-release on the QGIS canvas "
            "(right-click or Esc cancels). Whatever you draw, pick, or type ends up in "
            "the boxes above, so you can always fine-tune the numbers by hand."
        )
        aoi_note.setWordWrap(True)
        aoi_note.setStyleSheet("color: gray; font-size: 10px;")
        aoi_layout.addWidget(aoi_note)
        layout.addWidget(aoi_box)

        # --- Source ---
        src_box = QGroupBox("2. DEM source")
        self.src_box = src_box
        src_layout = QVBoxLayout(src_box)

        src_row = QHBoxLayout()
        src_row.addWidget(QLabel("Source:"))
        self.source_combo = QComboBox()
        self.source_combo.addItems([
            "OpenTopography global DEMs (free API key required)",
            "AWS Terrain Tiles (no key needed)",
        ])
        self.source_combo.setCurrentIndex(SOURCE_TERRAIN_TILES)
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        src_row.addWidget(self.source_combo)
        src_row.addStretch()
        src_layout.addLayout(src_row)

        self.source_stack = QStackedWidget()

        # OpenTopography page
        ot_w = QWidget()
        ot_form = QFormLayout(ot_w)
        self.ot_dem_combo = QComboBox()
        for name, (_code, res, coverage) in dem_download.OPENTOPOGRAPHY_DEMS.items():
            self.ot_dem_combo.addItem(f"{name}, {coverage}")
        # Copernicus GLO-30 (global coverage) stays the default; found by name so
        # adding datasets to the list can never change the default silently.
        _names = list(dem_download.OPENTOPOGRAPHY_DEMS.keys())
        self.ot_dem_combo.setCurrentIndex(
            next((i for i, n in enumerate(_names) if n.startswith("Copernicus GLO-30")), 0))
        ot_form.addRow("Dataset:", self.ot_dem_combo)
        self.ot_info = QLabel("")
        self.ot_info.setWordWrap(True)
        self.ot_info.setStyleSheet("color: gray; font-size: 10px;")
        ot_form.addRow("", self.ot_info)
        self.ot_dem_combo.currentIndexChanged.connect(self._update_ot_info)
        key_row = QHBoxLayout()
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setPlaceholderText("paste your OpenTopography API key here")
        key_row.addWidget(self.api_key_edit)
        self.save_key_chk = QCheckBox("Remember")
        self.save_key_chk.setChecked(True)
        key_row.addWidget(self.save_key_chk)
        get_key_btn = QPushButton("Get a free key…")
        get_key_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl(dem_download.OPENTOPOGRAPHY_KEY_URL)))
        key_row.addWidget(get_key_btn)
        ot_form.addRow("API key:", key_row)
        ot_note = QLabel(
            "Registration is free and takes a minute. SRTM and NASADEM only cover 60°N "
            "to 56°S, so for anything closer to the poles use Copernicus GLO-30/90. The "
            "key is stored in your QGIS settings on this machine only."
        )
        ot_note.setWordWrap(True)
        ot_note.setStyleSheet("color: gray; font-size: 10px;")
        ot_form.addRow("", ot_note)
        self.source_stack.addWidget(ot_w)

        # Terrain tiles page
        tt_w = QWidget()
        tt_form = QFormLayout(tt_w)
        self.zoom_spin = QSpinBox()
        self.zoom_spin.setRange(6, 14)
        self.zoom_spin.setValue(12)
        self.zoom_spin.valueChanged.connect(self._update_aoi_label)
        tt_form.addRow("Zoom level (detail):", self.zoom_spin)
        self.zoom_info = QLabel("")
        self.zoom_info.setWordWrap(True)
        self.zoom_info.setStyleSheet("color: #205020; font-size: 10px;")
        tt_form.addRow("", self.zoom_info)
        tt_note = QLabel(
            "AWS Terrain Tiles need no registration and cover the whole globe. They are "
            "a merged product (SRTM, NED, and others) delivered as RGB-encoded tiles at "
            "roughly 30 m detail near the equator at zoom 12; each extra zoom level "
            "halves the pixel size but quadruples the number of tiles to download. The "
            "tiles also include ocean bathymetry, so values below sea level offshore are "
            "expected rather than errors. For a citable, single-source DEM with known "
            "provenance, prefer OpenTopography above."
        )
        tt_note.setWordWrap(True)
        tt_note.setStyleSheet("color: gray; font-size: 10px;")
        tt_form.addRow("", tt_note)
        self.source_stack.addWidget(tt_w)

        src_layout.addWidget(self.source_stack)
        layout.addWidget(src_box)

        # --- Projection ---
        proj_box = QGroupBox("3. Convert to metres (projection)")
        self.proj_box = proj_box
        proj_form = QFormLayout(proj_box)

        self.reproject_chk = QCheckBox("Reproject to UTM (metres), strongly recommended")
        self.reproject_chk.setChecked(True)
        self.reproject_chk.toggled.connect(self._on_reproject_toggled)
        proj_form.addRow("", self.reproject_chk)

        crs_row = QHBoxLayout()
        self.auto_utm_chk = QCheckBox("Auto-detect UTM zone from the area")
        self.auto_utm_chk.setChecked(True)
        self.auto_utm_chk.toggled.connect(self._on_auto_utm_toggled)
        crs_row.addWidget(self.auto_utm_chk)
        self.epsg_edit = QLineEdit()
        self.epsg_edit.setPlaceholderText("EPSG code, e.g. 32610")
        self.epsg_edit.setEnabled(False)
        self.epsg_edit.setMaximumWidth(160)
        crs_row.addWidget(self.epsg_edit)
        crs_row.addStretch()
        proj_form.addRow("Target CRS:", crs_row)

        self.detected_crs_label = QLabel("")
        self.detected_crs_label.setStyleSheet("color: #205020; font-size: 11px;")
        proj_form.addRow("", self.detected_crs_label)

        res_row = QHBoxLayout()
        self.auto_res_chk = QCheckBox("Auto (keep the source's native detail)")
        self.auto_res_chk.setChecked(True)
        self.auto_res_chk.toggled.connect(
            lambda checked: self.res_spin.setEnabled(not checked))
        res_row.addWidget(self.auto_res_chk)
        self.res_spin = QDoubleSpinBox()
        self.res_spin.setRange(0.1, 10000.0)
        self.res_spin.setDecimals(2)
        self.res_spin.setValue(30.0)
        self.res_spin.setSuffix(" m")
        self.res_spin.setEnabled(False)
        res_row.addWidget(self.res_spin)
        res_row.addStretch()
        proj_form.addRow("Output pixel size:", res_row)

        self.resample_combo = QComboBox()
        self.resample_combo.addItems(["bilinear", "cubic", "cubicspline", "lanczos",
                                       "average", "nearest"])
        proj_form.addRow("Resampling:", self.resample_combo)

        self.vertical_combo = QComboBox()
        self.vertical_combo.addItems(list(dem_download.VERTICAL_UNIT_FACTORS.keys()))
        proj_form.addRow("Source elevation units:", self.vertical_combo)
        vert_note = QLabel(
            "Reprojection fixes the horizontal units only. If the elevation values "
            "themselves are in feet (common in US LiDAR products), convert them here too, "
            "otherwise every slope comes out about 3.3x too steep, with no obvious "
            "symptom in the output."
        )
        vert_note.setWordWrap(True)
        vert_note.setStyleSheet("color: gray; font-size: 10px;")
        proj_form.addRow("", vert_note)

        self.clip_chk = QCheckBox("Clip the result to the area of interest")
        self.clip_chk.setChecked(True)
        self.clip_chk.toggled.connect(lambda *_: self._apply_mode())
        proj_form.addRow("", self.clip_chk)

        layout.addWidget(proj_box)

        # --- Output ---
        out_box = QGroupBox("4. Output")
        self.out_box = out_box
        out_form = QFormLayout(out_box)
        out_row = QHBoxLayout()
        self.out_path_edit = QLineEdit()
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_output)
        out_row.addWidget(self.out_path_edit)
        out_row.addWidget(browse)
        out_form.addRow("Output DEM (GeoTIFF):", out_row)
        self.load_chk = QCheckBox("Load into QGIS when finished")
        self.load_chk.setChecked(True)
        out_form.addRow("", self.load_chk)
        self.set_as_dem_chk = QCheckBox(
            "Select it as the DEM in the Environmental Covariates tab")
        self.set_as_dem_chk.setChecked(True)
        out_form.addRow("", self.set_as_dem_chk)
        self.keep_raw_chk = QCheckBox("Also keep the raw (unprojected) download")
        self.keep_raw_chk.setChecked(False)
        out_form.addRow("", self.keep_raw_chk)
        layout.addWidget(out_box)

        run_row = QHBoxLayout()
        self.run_btn = QPushButton("Get DEM")
        self.run_btn.clicked.connect(self.run)
        run_row.addWidget(self.run_btn)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._on_cancel)
        run_row.addWidget(self.cancel_btn)
        run_row.addStretch()
        layout.addLayout(run_row)

        self.progress = QProgressBar()
        layout.addWidget(self.progress)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(4000)
        layout.addWidget(self.log)

        self._on_source_changed(self.source_combo.currentIndex())
        convert = QSettings().value("SAMKit/dem_mode", "download", type=str) == "convert"
        (self.mode_convert_rb if convert else self.mode_download_rb).setChecked(True)
        self.mode_download_rb.toggled.connect(self._on_mode_toggled)
        if convert:
            self.clip_chk.setChecked(False)
        self._apply_mode()

    @staticmethod
    def _coord_spin(minimum, maximum, value):
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(6)
        spin.setSingleStep(0.01)
        spin.setValue(value)
        spin.setMinimumWidth(105)
        return spin

    def _log(self, msg):
        self.log.appendPlainText(str(msg))
        self.log.repaint()

    # ------------------------------------------------------------------
    # Area of interest
    # ------------------------------------------------------------------
    def _on_draw_rectangle(self):
        canvas = self.iface.mapCanvas() if self.iface is not None else None
        if canvas is None:
            QMessageBox.warning(self, "Draw area", "No map canvas is available.")
            return
        self._prev_map_tool = canvas.mapTool()
        self._rect_map_tool = map_tools.RectangleMapTool(canvas)
        self._rect_map_tool.rectangleFinished.connect(self._on_rectangle_drawn)
        canvas.setMapTool(self._rect_map_tool)
        self.aoi_label.setText("drawing… click-drag-release a rectangle on the map "
                               "(right-click or Esc cancels)")
        self.aoi_label.setStyleSheet("color: #a04000; font-size: 11px;")

    def _on_rectangle_drawn(self, rect):
        canvas = self.iface.mapCanvas()
        try:
            canvas_crs = canvas.mapSettings().destinationCrs()
            self._set_bbox_from_rect(rect, canvas_crs)
        except Exception as e:
            QMessageBox.critical(self, "Draw area", f"Could not read the drawn area: {e}")
        finally:
            if canvas is not None and self._prev_map_tool is not None:
                canvas.setMapTool(self._prev_map_tool)
            self._rect_map_tool = None

    def _on_use_canvas_extent(self):
        canvas = self.iface.mapCanvas() if self.iface is not None else None
        if canvas is None:
            QMessageBox.warning(self, "Use map view", "No map canvas is available.")
            return
        self._set_bbox_from_rect(canvas.extent(), canvas.mapSettings().destinationCrs())

    def _on_use_layer_extent(self):
        layer = self.extent_layer_combo.currentLayer()
        if layer is None:
            QMessageBox.warning(self, "Use layer extent",
                                 "Pick a layer first, its extent will be used as the "
                                 "area of interest.")
            return
        extent = layer.extent()
        if extent.isEmpty():
            QMessageBox.warning(self, "Use layer extent",
                                 f"'{layer.name()}' has an empty extent.")
            return
        self._set_bbox_from_rect(extent, layer.crs())

    def _set_bbox_from_rect(self, rect, crs):
        """Transform any rectangle into WGS84 lon/lat and push it into the boxes."""
        wgs84 = QgsCoordinateReferenceSystem(WGS84)
        if crs is not None and crs.isValid() and crs != wgs84:
            transform = QgsCoordinateTransform(crs, wgs84, QgsProject.instance())
            rect = transform.transformBoundingBox(QgsRectangle(rect))
        for spin, value in ((self.west_spin, rect.xMinimum()),
                            (self.south_spin, rect.yMinimum()),
                            (self.east_spin, rect.xMaximum()),
                            (self.north_spin, rect.yMaximum())):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self._on_bbox_edited()

    def _on_bbox_edited(self):
        try:
            self._bbox = dem_download.normalize_bbox(
                self.west_spin.value(), self.south_spin.value(),
                self.east_spin.value(), self.north_spin.value())
        except Exception:
            self._bbox = None
        self._update_aoi_label()

    def _current_source(self):
        return (SOURCE_EXISTING_LAYER if self.mode_convert_rb.isChecked()
                else self.source_combo.currentIndex())

    def _on_mode_toggled(self, download_checked):
        # Converting your own DEM: never clip silently to a left-over area of interest.
        if not download_checked:
            self.clip_chk.setChecked(False)
        self._apply_mode()

    def _apply_mode(self):
        """Show what the chosen job needs: download (area + source) or convert (your DEM)."""
        convert = self.mode_convert_rb.isChecked()
        QSettings().setValue("SAMKit/dem_mode", "convert" if convert else "download")
        self.own_box.setVisible(convert)
        self.src_box.setVisible(not convert)
        # the area of interest is only needed to download, or to clip your own DEM
        self.aoi_box.setVisible(not convert or self.clip_chk.isChecked())
        if convert:
            self.own_box.setTitle("1. Your DEM")
            self.aoi_box.setTitle("Area to clip to (optional)")
            self.proj_box.setTitle("2. Convert to metres (projection)")
            self.out_box.setTitle("3. Output")
            self.clip_chk.setText("Clip the result to an area of interest")
            self.auto_utm_chk.setText("Auto-detect UTM zone from the DEM")
            self.keep_raw_chk.setVisible(False)     # nothing is downloaded
            self.run_btn.setText("Convert DEM")
        else:
            self.aoi_box.setTitle("1. Area of interest")
            self.proj_box.setTitle("3. Convert to metres (projection)")
            self.out_box.setTitle("4. Output")
            self.clip_chk.setText("Clip the result to the area of interest")
            self.auto_utm_chk.setText("Auto-detect UTM zone from the area")
            self.keep_raw_chk.setVisible(True)
            self.run_btn.setText("Get DEM")

    def _on_own_layer_changed(self, layer):
        """Keep the file box in step with the selected raster; suggest an output name."""
        src = self._resolve_raster_source(layer) if layer is not None else None
        self.own_path_edit.setText(src or "")
        if src and not self.out_path_edit.text().strip():
            stem = os.path.splitext(src)[0]
            self.out_path_edit.setText(stem + "_utm.tif")

    def _browse_own_dem(self):
        """Pick a DEM file from disk, add it to QGIS and select it as the source."""
        start = QSettings().value("SAMKit/last_dem_dir", "", type=str)
        path, _ = QFileDialog.getOpenFileName(
            self, "Open your DEM", start,
            "Raster DEMs (*.tif *.tiff *.img *.asc *.vrt *.hgt *.bil *.dem *.adf *.sdat "
            "*.nc *.jp2 *.flt);;All files (*)")
        if not path:
            return
        QSettings().setValue("SAMKit/last_dem_dir", os.path.dirname(path))
        layer = QgsRasterLayer(path, os.path.splitext(os.path.basename(path))[0])
        if not layer.isValid():
            QMessageBox.warning(self, "DEM", f"QGIS could not open this file as a raster:\n{path}")
            return
        if not layer.crs().isValid():
            QMessageBox.warning(
                self, "DEM",
                "This DEM has no coordinate reference system, so it cannot be converted to "
                "UTM. Assign its CRS in QGIS (Layer Properties > Source) and pick it again.")
        QgsProject.instance().addMapLayer(layer)
        self.mode_convert_rb.setChecked(True)
        self.existing_combo.setLayer(layer)
        self.own_path_edit.setText(path)
        self._log(f"Loaded your DEM: {path}  (CRS {layer.crs().authid() or 'not set'}, "
                  f"{layer.width()} x {layer.height()} cells)")

    def _selected_ot_code(self):
        names = list(dem_download.OPENTOPOGRAPHY_DEMS.keys())
        i = self.ot_dem_combo.currentIndex()
        return dem_download.OPENTOPOGRAPHY_DEMS[names[i]] if 0 <= i < len(names) else None

    def _update_ot_info(self, *_args):
        sel = self._selected_ot_code()
        if not sel:
            return
        code, res, _cov = sel
        if code in dem_download.OPENTOPOGRAPHY_USGS_DATASETS:
            txt = ("USGS 3DEP 1 m lidar bare-earth DEM (USA only), served by OpenTopography's "
                   "USGS 3DEP API. It exists only where 3DEP 1 m lidar has been collected, and "
                   "OpenTopography currently limits it to academic API keys. Keep the area "
                   "small (a few km²): 1 m data grows by 1 million cells per km². Pixel size "
                   "'Auto' keeps the native 1 m.")
            if self._bbox is not None:
                mb = dem_download.estimate_download_mb(self._bbox, float(res))
                txt += f"\nThis area at {res} m: about {mb:,.0f} MB."
        else:
            txt = ""
        self.ot_info.setText(txt)

    def _update_aoi_label(self):
        if self._bbox is None:
            self.aoi_label.setText("No valid area yet. Draw a rectangle, use a layer "
                                    "extent, or type a bounding box above.")
            self.aoi_label.setStyleSheet("color: #a04000; font-size: 11px;")
            self.detected_crs_label.setText("")
            self.zoom_info.setText("")
            return

        west, south, east, north = self._bbox
        width_km, height_km = dem_download.bbox_size_km(self._bbox)
        epsg, zone, hemi = dem_download.utm_epsg_for_bbox(self._bbox)
        text = (f"Area: {width_km:.1f} x {height_km:.1f} km "
                f"(~{width_km * height_km:,.0f} km²)   |   "
                f"W {west:.5f}, S {south:.5f}, E {east:.5f}, N {north:.5f}")
        if dem_download.crosses_utm_zones(self._bbox):
            text += ("\nNote: this area spans more than one UTM zone. Everything is "
                     "projected into the centroid's zone, which is fine, but distortion "
                     "grows towards the edges of very wide areas.")
        self.aoi_label.setText(text)
        self.aoi_label.setStyleSheet("color: #205020; font-size: 11px;")

        if self.auto_utm_chk.isChecked():
            self.detected_crs_label.setText(
                f"Detected: UTM zone {zone}{hemi}, EPSG:{epsg} (metres)")

        mid_lat = (south + north) / 2.0
        zoom = self.zoom_spin.value()
        n_tiles = dem_download.tile_count_for_bbox(self._bbox, zoom)
        res = dem_download.tile_resolution_m(zoom, mid_lat)
        warn = ", over the 400-tile limit, lower the zoom" if n_tiles > 400 else ""
        self.zoom_info.setText(f"≈ {res:.0f} m per pixel here; {n_tiles} tile(s) to "
                                f"download{warn}")
        if hasattr(self, "ot_info"):
            self._update_ot_info()

    # ------------------------------------------------------------------
    # Option toggles
    # ------------------------------------------------------------------
    def _on_source_changed(self, idx):
        self.source_stack.setCurrentIndex(idx)
        using_layer = idx == SOURCE_EXISTING_LAYER
        for w in (self.draw_btn, self.canvas_extent_btn, self.use_layer_btn):
            w.setToolTip("Optional when reprojecting an existing layer, only used if "
                          "'Clip to the area of interest' is ticked." if using_layer else "")

    def _on_reproject_toggled(self, checked):
        for w in (self.auto_utm_chk, self.res_spin, self.auto_res_chk,
                  self.resample_combo):
            w.setEnabled(checked)
        self.epsg_edit.setEnabled(checked and not self.auto_utm_chk.isChecked())
        if not checked:
            self.detected_crs_label.setText(
                "Leaving the DEM unprojected. Remember that slope, TWI and MRVBF "
                "computed from a DEM in degrees are not meaningful.")
            self.detected_crs_label.setStyleSheet("color: #a04000; font-size: 11px;")
        else:
            self.detected_crs_label.setStyleSheet("color: #205020; font-size: 11px;")
            self._update_aoi_label()

    def _on_auto_utm_toggled(self, checked):
        self.epsg_edit.setEnabled(not checked and self.reproject_chk.isChecked())
        if checked:
            self._update_aoi_label()
        else:
            self.detected_crs_label.setText("Using the EPSG code entered above.")

    def _browse_output(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save DEM as", "dem_utm.tif",
                                               filter="GeoTIFF (*.tif)")
        if path:
            if not path.lower().endswith((".tif", ".tiff")):
                path += ".tif"
            self.out_path_edit.setText(path)

    # ------------------------------------------------------------------
    # API key persistence
    # ------------------------------------------------------------------
    def _restore_api_key(self):
        key = QSettings().value("SAMKit/opentopography_api_key", "", type=str)
        if key:
            self.api_key_edit.setText(key)

    def _store_api_key(self):
        if self.save_key_chk.isChecked():
            QSettings().setValue("SAMKit/opentopography_api_key",
                                  self.api_key_edit.text().strip())
        else:
            QSettings().remove("SAMKit/opentopography_api_key")

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_raster_source(layer):
        """
        Return a source string GDAL can open for a raster already loaded in
        QGIS, or None if it genuinely can't be read.

        A raster layer's source() is usually a plain file path, but it can also
        carry GDAL open-options, a subdataset selector, a query string, or a
        /vsi* prefix -- none of which os.path.exists() understands. Rejecting on
        os.path.exists alone wrongly refuses perfectly valid local DEMs (the
        old "export it to GeoTIFF first" error). Here we take the layer's own
        source, strip only the QGIS "|..." decorations, and let GDAL be the
        judge of whether it's readable, falling back to the provider URI.
        """
        candidates = []
        src = layer.source() or ""
        candidates.append(src.split("|")[0].strip())
        candidates.append(src.strip())
        try:
            candidates.append(layer.dataProvider().dataSourceUri().split("|")[0].strip())
        except Exception:
            pass

        # A plain existing file is the happy path -- return it as-is.
        for cand in candidates:
            if cand and os.path.exists(cand):
                return cand

        # Not a plain file (subdataset, /vsi*, open-options, remote): trust GDAL.
        for cand in candidates:
            if not cand:
                continue
            try:
                dem_download.describe_raster(cand)
                return cand
            except Exception:
                continue
        return None

    def _validate_and_build_job(self):
        source = self._current_source()
        out_path = self.out_path_edit.text().strip()
        if not out_path:
            raise RuntimeError("Choose where to save the output DEM.")
        if not out_path.lower().endswith((".tif", ".tiff")):
            out_path += ".tif"
            self.out_path_edit.setText(out_path)
        out_dir = os.path.dirname(out_path) or "."
        if not os.path.isdir(out_dir):
            raise RuntimeError(f"The output folder does not exist:\n{out_dir}")

        existing_path = None
        if source == SOURCE_EXISTING_LAYER:
            layer = self.existing_combo.currentLayer()
            if layer is None:
                raise RuntimeError("Choose your DEM first: click 'Browse…' to open a DEM file, "
                                   "or pick a raster already loaded in QGIS.")
            existing_path = self._resolve_raster_source(layer)
            if existing_path is None:
                raise RuntimeError(
                    f"'{layer.name()}' is not backed by a raster GDAL can read. "
                    f"Export it to GeoTIFF first, then pick the exported file.")
        elif self._bbox is None:
            raise RuntimeError("Set a valid area of interest first: draw a rectangle, "
                                "use a layer extent, or type a bounding box.")

        if source == SOURCE_TERRAIN_TILES:
            n_tiles = dem_download.tile_count_for_bbox(self._bbox, self.zoom_spin.value())
            if n_tiles > 400:
                raise RuntimeError(
                    f"That area needs {n_tiles} tiles at zoom {self.zoom_spin.value()}, "
                    f"over the 400-tile limit. Lower the zoom level or draw a smaller "
                    f"area.")

        # Target CRS
        target_epsg = None
        if self.reproject_chk.isChecked():
            if self.auto_utm_chk.isChecked():
                # Your own DEM: take the zone from the DEM itself, not from a
                # (possibly unrelated, left-over) area of interest.
                if self._bbox is not None and source != SOURCE_EXISTING_LAYER:
                    target_epsg = dem_download.utm_epsg_for_bbox(self._bbox)[0]
                else:
                    layer = self.existing_combo.currentLayer()
                    extent = layer.extent()
                    crs = layer.crs()
                    wgs84 = QgsCoordinateReferenceSystem(WGS84)
                    if crs.isValid() and crs != wgs84:
                        extent = QgsCoordinateTransform(
                            crs, wgs84, QgsProject.instance()).transformBoundingBox(extent)
                    target_epsg = dem_download.utm_epsg_for(
                        extent.center().x(), extent.center().y())[0]
            else:
                text = self.epsg_edit.text().strip().upper().replace("EPSG:", "")
                if not text.isdigit():
                    raise RuntimeError("Enter a numeric EPSG code for the target CRS "
                                        "(for example 32610), or tick auto-detect.")
                target_epsg = int(text)

        demtype = None
        if source == SOURCE_OPENTOPO:
            names = list(dem_download.OPENTOPOGRAPHY_DEMS.keys())
            demtype = dem_download.OPENTOPOGRAPHY_DEMS[
                names[self.ot_dem_combo.currentIndex()]][0]

        stem = os.path.splitext(out_path)[0]
        return {
            "source": source,
            "bbox": self._bbox,
            "demtype": demtype,
            "api_key": self.api_key_edit.text().strip(),
            "zoom": self.zoom_spin.value(),
            "existing_path": existing_path,
            "raw_path": stem + "_raw.tif",
            "scaled_path": stem + "_raw_metres.tif",
            "out_path": out_path,
            "reproject": self.reproject_chk.isChecked(),
            "target_epsg": target_epsg,
            "resolution": None if self.auto_res_chk.isChecked() else self.res_spin.value(),
            "resample": self.resample_combo.currentText(),
            "vertical_factor": dem_download.VERTICAL_UNIT_FACTORS[
                self.vertical_combo.currentText()],
            "clip_bbox": self._bbox if (self.clip_chk.isChecked() and self._bbox) else None,
        }

    def run(self):
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(self, "DEM", "A download is already running.")
            return
        try:
            job = self._validate_and_build_job()
        except Exception as e:
            QMessageBox.warning(self, "DEM", str(e))
            return

        self._store_api_key()
        self.log.clear()
        self.progress.setValue(0)
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self._job = job

        if job["source"] != SOURCE_EXISTING_LAYER:
            self._log(f"Area: W {job['bbox'][0]:.5f}, S {job['bbox'][1]:.5f}, "
                      f"E {job['bbox'][2]:.5f}, N {job['bbox'][3]:.5f}")
        if job["reproject"]:
            self._log(f"Target CRS: EPSG:{job['target_epsg']} (metres)")
        self._log("")

        worker = _DemWorker(job, parent=self)
        worker.message.connect(self._log)
        worker.progress.connect(self.progress.setValue)
        worker.finished_ok.connect(self._on_finished)
        self._worker = worker
        worker.start()

    def _on_cancel(self):
        if self._worker is not None and self._worker.isRunning():
            self._log("Cancelling…")
            self._worker.cancel()
            self.cancel_btn.setEnabled(False)

    def _on_finished(self, ok, out_path, error):
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.deleteLater()

        job = getattr(self, "_job", {})
        if not self.keep_raw_chk.isChecked():
            for key in ("raw_path", "scaled_path"):
                path = job.get(key)
                if path and os.path.exists(path) and os.path.abspath(path) != \
                        os.path.abspath(job.get("out_path", "")):
                    try:
                        os.remove(path)
                    except Exception:
                        pass

        if not ok:
            self.progress.setValue(0)
            if error and error != "Cancelled.":
                self._log("ERROR: " + error)
                QMessageBox.critical(self, "DEM", error)
            else:
                self._log("Cancelled.")
            return

        self._log("")
        self._log(f"Done: {out_path}")

        layer = None
        if self.load_chk.isChecked():
            layer = QgsRasterLayer(out_path, os.path.splitext(os.path.basename(out_path))[0])
            if layer.isValid():
                QgsProject.instance().addMapLayer(layer)
            else:
                self._log("  (warning: could not load the DEM into QGIS)")
                layer = None

        if layer is not None and self.set_as_dem_chk.isChecked():
            self.demReady.emit(layer)
            self._log("Selected as the DEM in the Environmental Covariates tab.")

        QMessageBox.information(
            self, "DEM",
            f"DEM ready:\n{out_path}\n\n"
            f"Next step: the Environmental Covariates tab, to compute slope, curvature, "
            f"TPI, TWI and MRVBF from it.")
