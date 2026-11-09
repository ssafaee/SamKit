# -*- coding: utf-8 -*-
import os
import traceback

from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QCheckBox,
    QPushButton, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox, QPlainTextEdit,
    QProgressBar, QFileDialog, QMessageBox, QComboBox
)
from qgis.PyQt.QtCore import Qt, QThread, pyqtSignal, QUrl
from qgis.PyQt.QtGui import QDesktopServices
from qgis.gui import QgsMapLayerComboBox
from qgis.core import QgsMapLayerProxyModel, QgsRasterLayer, QgsProject

from . import utils
from . import qgis_algorithms
from . import deps


class _WhiteboxInstallWorker(QThread):
    """Installs the WhiteboxTools engine in the background.

    Two steps, either of which may be skipped: pip-install the 'whitebox'
    wrapper into the plugin-private libs/ folder, then download the
    whitebox_tools binary it drives. Both run off the GUI thread so QGIS stays
    responsive and the install stays cancellable.
    """

    message = pyqtSignal(str)
    finished_ok = pyqtSignal(bool)

    def __init__(self, need_package, need_binary, parent=None):
        super().__init__(parent)
        self._need_package = need_package
        self._need_binary = need_binary
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def _is_cancelled(self):
        return self._cancel

    def run(self):
        ok = True
        try:
            if self._need_package and not self._cancel:
                ok = deps.pip_install(
                    ["whitebox"],
                    log=lambda m: self.message.emit(str(m)),
                    cancelled=self._is_cancelled,
                )
            if ok and self._need_binary and not self._cancel:
                ok = deps.download_whitebox_binary(
                    log=lambda m: self.message.emit(str(m)),
                    cancelled=self._is_cancelled,
                )
        except Exception as e:
            self.message.emit("Unexpected error while installing WhiteboxTools: {}".format(e))
            self.message.emit(traceback.format_exc())
            ok = False
        self.finished_ok.emit(bool(ok) and not self._cancel)


class TerrainAttributesTab(QWidget):

    def __init__(self, iface, parent=None):
        super().__init__(parent)
        self.iface = iface
        self._wbt_worker = None
        self._build_ui()
        self._refresh_terrain_deps()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        form_box = QGroupBox("Input")
        form = QFormLayout(form_box)

        self.dem_combo = QgsMapLayerComboBox()
        self.dem_combo.setFilters(QgsMapLayerProxyModel.RasterLayer)
        form.addRow("DEM raster:", self.dem_combo)

        out_row = QHBoxLayout()
        self.out_dir_edit = QLineEdit()
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_output_dir)
        out_row.addWidget(self.out_dir_edit)
        out_row.addWidget(browse_btn)
        form.addRow("Output folder:", out_row)

        self.prefix_edit = QLineEdit("dem")
        form.addRow("Output filename prefix:", self.prefix_edit)

        layout.addWidget(form_box)

        self._build_dependency_panel(layout)

        method_box = QGroupBox("Computation method")
        method_layout = QVBoxLayout(method_box)
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method for slope / aspect / TPI:"))
        self.computation_method = QComboBox()
        self.computation_method.addItems([
            "SAMKit built-in (self-contained, no dependencies, adjustable)",
            "Exact GDAL via QGIS Processing (matches gdaldem exactly)",
        ])
        self.computation_method.currentIndexChanged.connect(self._on_method_changed)
        method_row.addWidget(self.computation_method)
        method_row.addStretch()
        method_layout.addLayout(method_row)
        edge_row = QHBoxLayout()
        edge_row.addWidget(QLabel("Edge / nodata rule (built-in 3x3 operators):"))
        self.edge_combo = QComboBox()
        self.edge_combo.addItems([
            "GDAL (== gdaldem -compute_edges)",
            "WhiteboxTools",
            "SAGA GIS",
            "Legacy (pre-1.0 SAMKit)",
        ])
        edge_row.addWidget(self.edge_combo)
        edge_row.addStretch()
        method_layout.addLayout(edge_row)
        method_note = QLabel(
            "In v1.0.0 every built-in covariate reproduces an established tool: slope/aspect "
            "equal gdaldem, WhiteboxTools or SAGA depending on the edge rule (interior cells are "
            "identical in all three); curvature can follow SAGA exactly; TPI matches "
            "WhiteboxTools/SAGA/GDAL; MRVBF/MRRTF are a verified port of SAGA's tool; TWI follows "
            "WhiteboxTools. 'Exact GDAL' mode calls gdaldem through QGIS Processing instead."
        )
        method_note.setWordWrap(True)
        method_note.setStyleSheet("color: gray; font-size: 10px;")
        method_layout.addWidget(method_note)
        layout.addWidget(method_box)

        attrs_box = QGroupBox("Environmental covariates to compute")
        attrs_layout = QVBoxLayout(attrs_box)

        self.chk_slope = QCheckBox("Slope (degrees)")
        self.chk_slope.setChecked(True)
        attrs_layout.addWidget(self.chk_slope)

        self.chk_aspect = QCheckBox("Aspect (degrees, 0-360, North=0 clockwise)")
        self.chk_aspect.setChecked(False)
        attrs_layout.addWidget(self.chk_aspect)

        curv_row = QHBoxLayout()
        self.chk_curv = QCheckBox("Curvature (general / profile / plan)")
        self.chk_curv.setChecked(True)
        self.curv_conv = QComboBox()
        self.curv_conv.addItems([
            "SAMKit / Esri-style (x100)",
            "SAGA-identical (C_GENE, C_PROF, C_PLAN, C_TANG; 1/m)",
            "Florinsky exact geometric (1/m)",
        ])
        curv_row.addWidget(self.chk_curv)
        curv_row.addWidget(QLabel("   convention:"))
        curv_row.addWidget(self.curv_conv)
        curv_row.addStretch()
        attrs_layout.addLayout(curv_row)

        tpi_row = QHBoxLayout()
        self.chk_tpi = QCheckBox("TPI (Topographic Position Index)")
        self.chk_tpi.setChecked(True)
        self.tpi_radius = QSpinBox()
        self.tpi_radius.setRange(1, 200)
        self.tpi_radius.setValue(5)
        tpi_row.addWidget(self.chk_tpi)
        tpi_row.addWidget(QLabel("   window radius (cells):"))
        tpi_row.addWidget(self.tpi_radius)
        self.tpi_kernel = QComboBox()
        self.tpi_kernel.addItems(["square window (WhiteboxTools DiffFromMeanElev)",
                                  "circular window (SAGA TPI)",
                                  "WhiteboxTools-exact (incl. its edge/binning behaviour)"])
        tpi_row.addWidget(self.tpi_kernel)
        tpi_row.addStretch()
        attrs_layout.addLayout(tpi_row)
        self.tpi_exact_note = QLabel(
            "In 'Exact GDAL' mode, TPI always uses GDAL's fixed 3x3 (immediate 8-neighbour) "
            "window -- the radius above is ignored and ships disabled in that mode."
        )
        self.tpi_exact_note.setWordWrap(True)
        self.tpi_exact_note.setStyleSheet("color: gray; font-size: 10px;")
        self.tpi_exact_note.setVisible(False)
        attrs_layout.addWidget(self.tpi_exact_note)

        twi_row = QHBoxLayout()
        self.chk_twi = QCheckBox("TWI (Topographic Wetness Index)")
        self.chk_twi.setChecked(True)
        self.twi_flow_method = QComboBox()
        self.twi_flow_method.addItems(["D8 (default)", "MFD (multiple flow direction)"])
        twi_row.addWidget(self.chk_twi)
        twi_row.addWidget(QLabel("   flow routing:"))
        twi_row.addWidget(self.twi_flow_method)
        twi_row.addStretch()
        attrs_layout.addLayout(twi_row)
        self.chk_twi_wbt = QCheckBox("Use WhiteboxTools for exact TWI (if installed; no plugin needed)")
        attrs_layout.addWidget(self.chk_twi_wbt)
        twi_note = QLabel(
            "Built-in TWI: Priority-Flood+epsilon depression filling (Barnes et al. 2014), "
            "WhiteboxTools D8 rules and SCA definition. "
            "D8 routes each cell's flow to a single steepest-descent neighbour (sharper, "
            "thread-like channels). MFD spreads flow across all downslope neighbours weighted "
            "by slope, generally giving smoother, often more realistic catchment areas on "
            "hillslopes at the cost of some dispersion in narrow valleys."
        )
        twi_note.setWordWrap(True)
        twi_note.setStyleSheet("color: gray; font-size: 10px;")
        attrs_layout.addWidget(twi_note)

        mrvbf_row = QHBoxLayout()
        self.chk_mrvbf = QCheckBox("MRVBF && MRRTF (multiresolution valley bottom / ridge top flatness)")
        self.chk_mrvbf.setChecked(True)
        self.mrvbf_levels = QDoubleSpinBox()   # 'maximum resolution' (% of DEM diameter), as in SAGA
        self.mrvbf_levels.setRange(1.0, 100.0)
        self.mrvbf_levels.setValue(100.0)
        mrvbf_row.addWidget(self.chk_mrvbf)
        mrvbf_row.addWidget(QLabel("   max. resolution (% of DEM diameter):"))
        mrvbf_row.addWidget(self.mrvbf_levels)
        mrvbf_row.addStretch()
        attrs_layout.addLayout(mrvbf_row)

        self.chk_mrvbf_saga = QCheckBox(
            "Use SAGA for exact MRVBF/MRRTF (requires a SAGA GIS install; no provider plugin needed)")
        self.chk_mrvbf_saga.setChecked(False)
        self.chk_mrvbf_saga.toggled.connect(self._on_mrvbf_saga_toggled)
        attrs_layout.addWidget(self.chk_mrvbf_saga)
        mrvbf_saga_note = QLabel(
            "GDAL has no MRVBF/MRRTF algorithm at all, in any form -- SAGA GIS is the only "
            "source of the genuine Gallant & Dowling (2003) reference implementation. "
            "Checking this calls SAGA's own MRVBF tool for exact reference output. "
            "SAMKit uses a registered QGIS SAGA provider if one exists, otherwise it "
            "calls SAGA's command-line program (saga_cmd) directly, which works with "
            "any installed SAGA version and needs no provider plugin (the DEM is handed "
            "to SAGA with a verified square cell size). Set the SAGA "
            "folder below if it isn't auto-detected. Leave unchecked to use SAMKit's "
            "built-in implementation, which is a verified port of the same SAGA tool and "
            "gives identical results without any SAGA install."
        )
        mrvbf_saga_note.setWordWrap(True)
        mrvbf_saga_note.setStyleSheet("color: gray; font-size: 10px;")
        attrs_layout.addWidget(mrvbf_saga_note)

        saga_path_row = QHBoxLayout()
        self.saga_path_label = QLabel("   SAGA folder (optional):")
        self.saga_cmd_path = QLineEdit()
        self.saga_cmd_path.setPlaceholderText(
            "auto-detected if blank -- folder containing saga_cmd(.exe)")
        self.btn_saga_browse = QPushButton("Browse…")
        self.btn_saga_browse.clicked.connect(self._on_browse_saga_folder)
        saga_path_row.addWidget(self.saga_path_label)
        saga_path_row.addWidget(self.saga_cmd_path)
        saga_path_row.addWidget(self.btn_saga_browse)
        attrs_layout.addLayout(saga_path_row)
        # path field only relevant when the SAGA option is on
        for w in (self.saga_path_label, self.saga_cmd_path, self.btn_saga_browse):
            w.setEnabled(self.chk_mrvbf_saga.isChecked())

        note = QLabel(
            "Note: no SAGA or WhiteboxTools installation is ever required in this tab. Slope/aspect/TPI can "
            "be computed either by SAMKit's self-contained numpy/GDAL implementation, or "
            "(see 'Computation method' above) via QGIS's own Processing framework calling "
            "GDAL directly. The built-in MRVBF/MRRTF are bit-identical to SAGA 9.3.1 and the "
            "built-in TWI follows WhiteboxTools; the optional SAGA/WhiteboxTools routes call "
            "those programs directly."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: gray; font-size: 10px;")
        attrs_layout.addWidget(note)

        dem_quality_note = QLabel(
            "IMPORTANT, input DEM quality: every covariate here is only as good as the DEM "
            "itself. Noisy, artifact-laden, or unfiltered DEMs (sensor noise, interpolation "
            "artifacts, striping) will propagate directly into slope/curvature/TWI/MRVBF and can "
            "produce misleading covariates and weak model results. Use a cleaned/filtered DEM "
            "appropriate to your study, and consider whether man-made structures (buildings, "
            "berms, roads, levees) should be removed depending on your application -- e.g. for "
            "natural terrain/soil-forming process analysis a bare-earth DEM with structures "
            "removed is usually preferable, whereas some engineering/hydrology studies may "
            "intentionally want them retained."
        )
        dem_quality_note.setWordWrap(True)
        dem_quality_note.setStyleSheet("color: #a04000; font-size: 10px; font-weight: bold;")
        attrs_layout.addWidget(dem_quality_note)

        layout.addWidget(attrs_box)

        self.load_outputs_chk = QCheckBox("Load output rasters into QGIS after processing")
        self.load_outputs_chk.setChecked(True)
        layout.addWidget(self.load_outputs_chk)

        run_row = QHBoxLayout()
        self.run_btn = QPushButton("Calculate environmental covariates")
        self.run_btn.clicked.connect(self.run)
        run_row.addWidget(self.run_btn)
        layout.addLayout(run_row)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        layout.addWidget(self.log)

    def _build_dependency_panel(self, layout):
        """Optional exact-match engines: WhiteboxTools (installable here) and
        SAGA GIS (detected here; installed separately as it is a full GIS)."""
        dep_box = QGroupBox("Optional exact-match engines (WhiteboxTools / SAGA)")
        dep_layout = QVBoxLayout(dep_box)

        intro = QLabel(
            "Optional. SAMKit's built-in methods need nothing installed and cover "
            "every covariate. Install WhiteboxTools only for the exact WhiteboxTools "
            "TWI route, and SAGA only for the exact SAGA MRVBF/MRRTF route. "
            "WhiteboxTools installs into SAMKit's own folder (no admin rights, QGIS "
            "is not modified) and works right away, no restart needed."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #444; font-size: 11px;")
        dep_layout.addWidget(intro)

        # --- WhiteboxTools (pip-installable from here) ---
        wbt_row = QHBoxLayout()
        wbt_row.addWidget(QLabel("WhiteboxTools:"))
        self.wbt_status = QLabel("checking…")
        wbt_row.addWidget(self.wbt_status)
        wbt_row.addStretch()
        self.wbt_install_btn = QPushButton("Install / download WhiteboxTools")
        self.wbt_install_btn.clicked.connect(self._install_whitebox)
        wbt_row.addWidget(self.wbt_install_btn)
        self.wbt_cancel_btn = QPushButton("Cancel")
        self.wbt_cancel_btn.setEnabled(False)
        self.wbt_cancel_btn.clicked.connect(self._cancel_whitebox_install)
        wbt_row.addWidget(self.wbt_cancel_btn)
        dep_layout.addLayout(wbt_row)

        # --- SAGA (external GIS; detected, not pip-installable) ---
        saga_row = QHBoxLayout()
        saga_row.addWidget(QLabel("SAGA GIS:"))
        self.saga_status = QLabel("checking…")
        saga_row.addWidget(self.saga_status)
        saga_row.addStretch()
        self.saga_get_btn = QPushButton("Get SAGA GIS…")
        self.saga_get_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(
                QUrl("https://saga-gis.sourceforge.io/en/index.html")))
        saga_row.addWidget(self.saga_get_btn)
        dep_layout.addLayout(saga_row)

        saga_note = QLabel(
            "SAGA is a separate GIS and can't be installed with one click. Install "
            "any SAGA build, then set its folder in the SAGA path field below (or add "
            "it to PATH) and click 'Check again'. saga_cmd works with any SAGA version."
        )
        saga_note.setWordWrap(True)
        saga_note.setStyleSheet("color: gray; font-size: 10px;")
        dep_layout.addWidget(saga_note)

        refresh_row = QHBoxLayout()
        self.dep_refresh_btn = QPushButton("Check again")
        self.dep_refresh_btn.clicked.connect(self._refresh_terrain_deps)
        refresh_row.addWidget(self.dep_refresh_btn)
        refresh_row.addStretch()
        dep_layout.addLayout(refresh_row)

        layout.addWidget(dep_box)

    def _refresh_terrain_deps(self):
        """Detect WhiteboxTools and SAGA and update the status labels.

        Detection only -- never installs anything and never blocks. The
        WhiteboxTools binary is what actually matters (the wrapper alone can't
        run), so the status reflects the binary, not just the pip package.
        """
        # WhiteboxTools
        pkg_ok, pkg_ver = deps.check_package("whitebox", "whitebox")
        try:
            binpath = qgis_algorithms._find_whitebox_tools()
        except Exception:
            binpath = None
        if binpath:
            self.wbt_status.setText("installed and ready")
            self.wbt_status.setStyleSheet("color: green;")
            self.wbt_install_btn.setText("Reinstall / re-download")
        elif pkg_ok:
            self.wbt_status.setText(
                "wrapper installed (v{}); binary not downloaded yet".format(pkg_ver or "?"))
            self.wbt_status.setStyleSheet("color: #b36b00;")
            self.wbt_install_btn.setText("Download WhiteboxTools binary")
        else:
            self.wbt_status.setText("not installed")
            self.wbt_status.setStyleSheet("color: red;")
            self.wbt_install_btn.setText("Install / download WhiteboxTools")

        # SAGA
        saga_path = ""
        w = getattr(self, "saga_cmd_path", None)
        if w is not None:
            saga_path = w.text().strip()
        saga_cmd = provider = None
        try:
            saga_cmd = qgis_algorithms._find_saga_cmd(saga_path or None, log=lambda *_a, **_k: None)
            provider = qgis_algorithms._resolve_saga_mrvbf_algorithm_id()
        except Exception:
            pass
        if saga_cmd:
            self.saga_status.setText("found: {}".format(saga_cmd))
            self.saga_status.setStyleSheet("color: green;")
        elif provider:
            self.saga_status.setText("found (QGIS SAGA provider: {})".format(provider))
            self.saga_status.setStyleSheet("color: green;")
        else:
            self.saga_status.setText("not found, install SAGA GIS separately")
            self.saga_status.setStyleSheet("color: red;")

    def _set_wbt_busy(self, busy):
        self.wbt_install_btn.setEnabled(not busy)
        self.dep_refresh_btn.setEnabled(not busy)
        self.wbt_cancel_btn.setEnabled(busy)

    def _install_whitebox(self):
        if self._wbt_worker is not None and self._wbt_worker.isRunning():
            QMessageBox.information(self, "WhiteboxTools", "An install is already running.")
            return
        pkg_ok, _ = deps.check_package("whitebox", "whitebox")
        self._log("Installing WhiteboxTools … this runs in the background; "
                  "QGIS stays usable and you can cancel it.")
        self._set_wbt_busy(True)
        worker = _WhiteboxInstallWorker(need_package=not pkg_ok, need_binary=True, parent=self)
        worker.message.connect(self._log)
        worker.finished_ok.connect(self._on_whitebox_installed)
        self._wbt_worker = worker
        worker.start()

    def _cancel_whitebox_install(self):
        if self._wbt_worker is not None and self._wbt_worker.isRunning():
            self._log("Cancelling WhiteboxTools install …")
            self._wbt_worker.cancel()
            self.wbt_cancel_btn.setEnabled(False)

    def _on_whitebox_installed(self, ok):
        self._set_wbt_busy(False)
        worker = self._wbt_worker
        self._wbt_worker = None
        if worker is not None:
            worker.deleteLater()
        self._refresh_terrain_deps()
        if ok:
            self.chk_twi_wbt.setChecked(True)
            QMessageBox.information(
                self, "WhiteboxTools",
                "WhiteboxTools is installed and ready.\n\nThe 'Use WhiteboxTools "
                "for exact TWI' option has been ticked for you, no QGIS restart "
                "needed.")
        else:
            QMessageBox.warning(
                self, "WhiteboxTools",
                "WhiteboxTools did not finish installing. See the log below for "
                "details, including the manual pip command you can run.")

    def _browse_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder")
        if d:
            self.out_dir_edit.setText(d)

    def _on_method_changed(self, idx):
        exact_mode = idx == 1
        self.tpi_radius.setEnabled(not exact_mode)
        self.tpi_exact_note.setVisible(exact_mode)

    def _edge_mode(self):
        return ("gdal", "whitebox", "saga", "legacy")[self.edge_combo.currentIndex()]

    def _on_mrvbf_saga_toggled(self, checked):
        for w in (getattr(self, "saga_path_label", None),
                  getattr(self, "saga_cmd_path", None),
                  getattr(self, "btn_saga_browse", None)):
            if w is not None:
                w.setEnabled(checked)

    def _on_browse_saga_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select the SAGA folder (containing saga_cmd)")
        if folder:
            self.saga_cmd_path.setText(folder)

    def _log(self, msg):
        self.log.appendPlainText(str(msg))
        self.log.repaint()

    def _offer_projection(self, layer, dem_path, out_dir, prefix):
        """The DEM is in degrees: offer to reproject it to its UTM zone first.
        Returns the path of the DEM to use, or None to stop."""
        from qgis.PyQt.QtWidgets import QMessageBox as _MB
        from qgis.core import QgsCoordinateReferenceSystem, QgsCoordinateTransform
        from . import dem_download
        ext = layer.extent()
        wgs84 = QgsCoordinateReferenceSystem("EPSG:4326")
        if layer.crs() != wgs84:
            ext = QgsCoordinateTransform(layer.crs(), wgs84, QgsProject.instance()) \
                .transformBoundingBox(ext)
        epsg, zone, hemi = dem_download.utm_epsg_for(ext.center().x(), ext.center().y())
        answer = _MB.question(
            self, "Environmental Covariates: DEM is in degrees",
            f"'{layer.name()}' is in a geographic CRS ({layer.crs().authid()}), so its cell "
            f"size is in degrees while elevations are in metres.\n\n"
            f"Slope, curvature, TPI, TWI and MRVBF/MRRTF computed on it are wrong (slope "
            f"comes out near 90 degrees everywhere; MRVBF/MRRTF are 0 or nodata).\n\n"
            f"Reproject it to UTM zone {zone}{hemi} (EPSG:{epsg}, metres) now and compute "
            f"the covariates on that?",
            _MB.Yes | _MB.Cancel, _MB.Yes)
        if answer != _MB.Yes:
            self._log("Stopped: the DEM is in degrees. Reproject it (tab 1, source 'Use a "
                      "DEM already loaded in QGIS') before computing covariates.")
            return None
        out = os.path.join(out_dir, f"{prefix}_dem_utm{zone}{hemi}.tif")
        self.log.clear()
        self._log(f"DEM is in degrees: reprojecting to EPSG:{epsg} (UTM zone {zone}{hemi})…")
        try:
            dem_download.reproject_to_metres(dem_path, out, epsg, log=self._log)
        except Exception as e:
            QMessageBox.critical(self, "Environmental Covariates",
                                 f"Could not reproject the DEM:\n{e}")
            return None
        new_layer = QgsRasterLayer(out, os.path.splitext(os.path.basename(out))[0])
        if new_layer.isValid():
            QgsProject.instance().addMapLayer(new_layer)
            self.dem_combo.setLayer(new_layer)
        self._log(f"Projected DEM written to {out}; covariates are computed from it.")
        return out

    def run(self):
        layer = self.dem_combo.currentLayer()
        out_dir = self.out_dir_edit.text().strip()
        prefix = self.prefix_edit.text().strip() or "dem"

        if layer is None or not isinstance(layer, QgsRasterLayer):
            QMessageBox.warning(self, "Environmental Covariates", "Please select a valid DEM raster layer.")
            return
        if not out_dir or not os.path.isdir(out_dir):
            QMessageBox.warning(self, "Environmental Covariates", "Please choose a valid output folder.")
            return

        dem_path = layer.source().split("|")[0]

        # Every covariate divides an elevation difference by a horizontal distance,
        # so a DEM in degrees gives meaningless results (slope ~90 deg everywhere,
        # MRVBF/MRRTF all 0 -- or all nodata via SAGA). Never compute on one.
        crs = layer.crs()
        if not crs.isValid():
            QMessageBox.warning(
                self, "Environmental Covariates",
                "The selected DEM has no coordinate reference system, so SAMKit cannot "
                "tell whether its cell size is in metres.\n\nAssign its CRS (Layer "
                "Properties > Source), or prepare it in tab 1 'DEM (download / project)'.")
            return
        if crs.isGeographic():
            dem_path = self._offer_projection(layer, dem_path, out_dir, prefix)
            if not dem_path:
                return

        n_tasks = sum([self.chk_slope.isChecked(), self.chk_aspect.isChecked(),
                       self.chk_curv.isChecked(), self.chk_tpi.isChecked(),
                       self.chk_twi.isChecked(), self.chk_mrvbf.isChecked()])
        if n_tasks == 0:
            QMessageBox.information(self, "Environmental Covariates", "Select at least one attribute to compute.")
            return

        self.run_btn.setEnabled(False)
        self.progress.setValue(0)
        if not layer.crs().isGeographic():
            self.log.clear()
        outputs = []

        try:
            self._log(f"Reading DEM: {dem_path}")
            ri = utils.read_raster(dem_path)
            cellsize_x, cellsize_y = ri.cellsize_x, ri.cellsize_y
            self._log(f"Grid: {ri.shape[0]} rows x {ri.shape[1]} cols, "
                      f"cell size ~ {cellsize_x:.3f} x {cellsize_y:.3f}")

            done = 0

            if self.chk_slope.isChecked() or self.chk_aspect.isChecked():
                exact_mode = self.computation_method.currentIndex() == 1
                if exact_mode:
                    self._log("Computing slope/aspect via QGIS Processing (exact GDAL)…")
                    if self.chk_slope.isChecked():
                        p = os.path.join(out_dir, f"{prefix}_slope.tif")
                        qgis_algorithms.compute_slope_gdal(dem_path, p, log=self._log)
                        outputs.append(("Slope", p))
                        self._log(f"  -> wrote {p}")
                        done += 1
                        self.progress.setValue(int(done / n_tasks * 100))
                    if self.chk_aspect.isChecked():
                        p = os.path.join(out_dir, f"{prefix}_aspect.tif")
                        qgis_algorithms.compute_aspect_gdal(dem_path, p, log=self._log)
                        outputs.append(("Aspect", p))
                        self._log(f"  -> wrote {p}")
                        done += 1
                        self.progress.setValue(int(done / n_tasks * 100))
                else:
                    self._log("Computing slope/aspect (SAMKit built-in)…")
                    slope_deg, aspect_deg, _, _, _ = utils.compute_slope_aspect(
                        ri.array, cellsize_x, cellsize_y, edge_mode=self._edge_mode())
                    if self.chk_slope.isChecked():
                        p = os.path.join(out_dir, f"{prefix}_slope.tif")
                        utils.write_raster(p, slope_deg, ri)
                        outputs.append(("Slope", p))
                        self._log(f"  -> wrote {p}")
                        done += 1
                        self.progress.setValue(int(done / n_tasks * 100))
                    if self.chk_aspect.isChecked():
                        p = os.path.join(out_dir, f"{prefix}_aspect.tif")
                        utils.write_raster(p, aspect_deg, ri)
                        outputs.append(("Aspect", p))
                        self._log(f"  -> wrote {p}")
                        done += 1
                        self.progress.setValue(int(done / n_tasks * 100))

            if self.chk_curv.isChecked():
                self._log("Computing curvature (general/profile/plan)…")
                conv = ("samkit", "saga", "florinsky")[self.curv_conv.currentIndex()]
                curv = utils.compute_curvature(
                    ri.array, cellsize_x, cellsize_y,
                    edge_mode=None if conv == "saga" else self._edge_mode(), convention=conv)
                for name, arr in curv.items():
                    p = os.path.join(out_dir, f"{prefix}_curvature_{name}.tif")
                    utils.write_raster(p, arr, ri)
                    outputs.append((f"Curvature ({name})", p))
                    self._log(f"  -> wrote {p}")
                done += 1
                self.progress.setValue(int(done / n_tasks * 100))

            if self.chk_tpi.isChecked():
                exact_mode = self.computation_method.currentIndex() == 1
                p = os.path.join(out_dir, f"{prefix}_tpi.tif")
                if exact_mode:
                    self._log("Computing TPI via QGIS Processing (exact GDAL, fixed 3x3 window)…")
                    qgis_algorithms.compute_tpi_gdal(dem_path, p, log=self._log)
                else:
                    self._log(f"Computing TPI (SAMKit built-in, radius={self.tpi_radius.value()} cells)…")
                    kidx = self.tpi_kernel.currentIndex()
                    tpi = utils.compute_tpi(ri.array, radius_cells=self.tpi_radius.value(),
                                            kernel="circle" if kidx == 1 else "square",
                                            compat="whitebox" if kidx == 2 else None)
                    utils.write_raster(p, tpi, ri)
                outputs.append(("TPI", p))
                self._log(f"  -> wrote {p}")
                done += 1
                self.progress.setValue(int(done / n_tasks * 100))

            if self.chk_twi.isChecked():
                flow_method = "mfd" if self.twi_flow_method.currentIndex() == 1 else "d8"
                self._log(f"Computing TWI ({flow_method.upper()} flow accumulation, "
                          f"may take a while on large DEMs)…")
                p = os.path.join(out_dir, f"{prefix}_twi.tif")
                if self.chk_twi_wbt.isChecked():
                    qgis_algorithms.compute_twi_whitebox(dem_path, p, log=self._log)
                else:
                    twi = utils.compute_twi(ri.array, cellsize_x, cellsize_y, flow_method=flow_method)
                    utils.write_raster(p, twi, ri)
                outputs.append(("TWI", p))
                self._log(f"  -> wrote {p}")
                done += 1
                self.progress.setValue(int(done / n_tasks * 100))

            if self.chk_mrvbf.isChecked():
                p1 = os.path.join(out_dir, f"{prefix}_mrvbf.tif")
                p2 = os.path.join(out_dir, f"{prefix}_mrrtf.tif")
                if self.chk_mrvbf_saga.isChecked():
                    self._log("Computing MRVBF/MRRTF via SAGA GIS (exact reference "
                              "implementation, QGIS SAGA provider if present, else "
                              "saga_cmd directly)…")
                    saga_path = self.saga_cmd_path.text().strip() or None
                    qgis_algorithms.compute_mrvbf_mrrtf_saga(
                        dem_path, p1, p2, saga_cmd_path=saga_path,
                        max_res=self.mrvbf_levels.value(), log=self._log)
                else:
                    self._log("Computing MRVBF/MRRTF (built-in port of SAGA ta_morphometry 8)…")
                    gt = ri.geotransform
                    origin = (gt[0] + 0.5 * cellsize_x, gt[3] + ri.shape[0] * gt[5] + 0.5 * cellsize_y)
                    res = utils.compute_mrvbf_mrrtf(ri.array, cellsize_x, cellsize_y,
                                                     max_res=self.mrvbf_levels.value(), origin=origin)
                    utils.write_raster(p1, res["mrvbf"], ri)
                    utils.write_raster(p2, res["mrrtf"], ri)
                outputs.append(("MRVBF", p1))
                outputs.append(("MRRTF", p2))
                self._log(f"  -> wrote {p1}")
                self._log(f"  -> wrote {p2}")
                done += 1
                self.progress.setValue(int(done / n_tasks * 100))

            self.progress.setValue(100)
            self._log("Done.")

            if self.load_outputs_chk.isChecked():
                for name, p in outputs:
                    rl = QgsRasterLayer(p, os.path.basename(p))
                    if rl.isValid():
                        QgsProject.instance().addMapLayer(rl)
                    else:
                        self._log(f"  (warning: could not load {p} into QGIS)")

            QMessageBox.information(self, "Environmental Covariates",
                                     f"Finished. {len(outputs)} raster(s) written to:\n{out_dir}")

        except Exception as e:
            self._log("ERROR: " + str(e))
            self._log(traceback.format_exc())
            QMessageBox.critical(self, "Environmental Covariates", f"Processing failed:\n{e}")
        finally:
            self.run_btn.setEnabled(True)
