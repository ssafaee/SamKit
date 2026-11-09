# -*- coding: utf-8 -*-
import os
import re
import sys
import csv
import json
import random
import datetime
import tempfile
import traceback

import numpy as np

from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QCheckBox,
    QPushButton, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox, QPlainTextEdit,
    QProgressBar, QFileDialog, QMessageBox, QComboBox, QListWidget,
    QListWidgetItem, QStackedWidget, QApplication, QDialog, QScrollArea,
    QTabWidget, QTableWidget, QTableWidgetItem
)
from qgis.PyQt.QtCore import Qt, QRectF, QThread, pyqtSignal, QUrl
from qgis.PyQt.QtGui import (QCursor, QPixmap, QPainter, QColor, QFont, QPen,
                              QDesktopServices)
from qgis.gui import QgsMapLayerComboBox
from qgis.core import (
    QgsMapLayerProxyModel, QgsVectorLayer, QgsRasterLayer, QgsProject,
    QgsWkbTypes, QgsCoordinateTransform, QgsCoordinateReferenceSystem,
    QgsPointXY, QgsGeometry, QgsFeature, QgsFeatureRequest, QgsVectorFileWriter,
    QgsMemoryProviderUtils
)

from . import utils
from . import dsm_models
from . import deps
from . import map_tools


class _DependencyInstallWorker(QThread):
    """Runs pip in a background thread.

    pip used to be called directly from the button's clicked handler, which
    blocked the Qt event loop for the whole download+install. QGIS was then
    painted as "Not Responding" by the window manager and users force-killed
    it. Running it here keeps the UI alive and cancellable.
    """

    message = pyqtSignal(str)
    finished_ok = pyqtSignal(bool)

    def __init__(self, pip_names, parent=None):
        super().__init__(parent)
        self._pip_names = list(pip_names)
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def _is_cancelled(self):
        return self._cancel

    def run(self):
        try:
            ok = deps.pip_install(
                self._pip_names,
                log=lambda msg: self.message.emit(str(msg)),
                cancelled=self._is_cancelled,
            )
        except Exception as e:
            self.message.emit(f"Unexpected error while installing: {e}")
            self.message.emit(traceback.format_exc())
            ok = False
        self.finished_ok.emit(bool(ok))


class DSMTab(QWidget):

    def __init__(self, iface, parent=None):
        super().__init__(parent)
        self.iface = iface
        self._preview_dialog = None
        self._results_dialog = None
        self._poly_map_tool = None
        self._prev_map_tool = None
        self._drawn_calib_layer_id = None
        self._install_worker = None
        self._restart_required = False
        self._scen_dir_user_set = False
        self._scen_eligible_count = 0
        self._build_ui()
        self._refresh_raster_list()
        self._refresh_dependency_status()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout(self)

        # --- Python dependencies group ---
        dep_box = QGroupBox("Python dependencies (for the models below)")
        dep_layout = QVBoxLayout(dep_box)

        interp_label = QLabel(
            f"SAMKit is running under this Python interpreter:\n{sys.executable}\n"
            f"Packages are installed into SAMKit's own folder:\n{deps.LIBS_DIR}"
        )
        interp_label.setWordWrap(True)
        interp_label.setStyleSheet("color: gray; font-size: 10px;")
        dep_layout.addWidget(interp_label)

        self.dep_checks = {}
        for display_name in deps.PACKAGES:
            row = QHBoxLayout()
            chk = QCheckBox(display_name)
            status = QLabel("checking…")
            status.setStyleSheet("color: gray;")
            row.addWidget(chk)
            row.addWidget(status)
            row.addStretch()
            dep_layout.addLayout(row)
            self.dep_checks[display_name] = (chk, status)

        dep_btn_row = QHBoxLayout()
        self.refresh_dep_btn = QPushButton("Refresh status")
        self.refresh_dep_btn.clicked.connect(self._refresh_dependency_status)
        self.install_dep_btn = QPushButton("Install checked packages")
        self.install_dep_btn.clicked.connect(self._install_checked_dependencies)
        self.cancel_dep_btn = QPushButton("Cancel install")
        self.cancel_dep_btn.clicked.connect(self._cancel_dependency_install)
        self.cancel_dep_btn.setEnabled(False)
        dep_btn_row.addWidget(self.refresh_dep_btn)
        dep_btn_row.addWidget(self.install_dep_btn)
        dep_btn_row.addWidget(self.cancel_dep_btn)
        dep_btn_row.addStretch()
        dep_layout.addLayout(dep_btn_row)

        self.dep_restart_label = QLabel("")
        self.dep_restart_label.setWordWrap(True)
        self.dep_restart_label.setStyleSheet("color: #b36b00; font-weight: bold;")
        self.dep_restart_label.setVisible(False)
        dep_layout.addWidget(self.dep_restart_label)

        dep_note = QLabel(
            "Tip: only check the package(s) needed for the model you plan to use "
            "(see table in the README). Packages are installed into SAMKit's own "
            "'libs' folder, not into QGIS itself, so no administrator rights are "
            "needed and your QGIS installation (including its numpy and scipy) is "
            "never modified. The download runs in the background -- QGIS stays "
            "usable, and you can cancel it. After a successful install you must "
            "restart QGIS once before the new packages can be used; Python cannot "
            "safely load newly installed compiled packages into a session that is "
            "already running. If the button fails on a given machine, there is also "
            "a standalone 'install_dependencies.py' script in the SAMKit plugin "
            "folder -- open it in QGIS's own Python Console (Plugins > Python "
            "Console) and run it there. See the README for details."
        )
        dep_note.setWordWrap(True)
        dep_note.setStyleSheet("color: gray; font-size: 10px;")
        dep_layout.addWidget(dep_note)

        layout.addWidget(dep_box)

        # --- Input data group ---
        data_box = QGroupBox("Input data")
        form = QFormLayout(data_box)

        self.points_combo = QgsMapLayerComboBox()
        self.points_combo.setFilters(QgsMapLayerProxyModel.PointLayer)
        self.points_combo.layerChanged.connect(self._on_points_layer_changed)
        form.addRow("Georeferenced points\n(measured samples with X,Y):", self.points_combo)

        self.field_combo = QComboBox()
        form.addRow("Attribute of interest (field):", self.field_combo)

        raster_row = QVBoxLayout()
        self.raster_list = QListWidget()
        self.raster_list.setMaximumHeight(140)
        raster_row.addWidget(self.raster_list)
        refresh_btn = QPushButton("Refresh raster list")
        refresh_btn.clicked.connect(self._refresh_raster_list)
        raster_row.addWidget(refresh_btn)
        form.addRow("Covariate rasters\n(check all that apply):", raster_row)

        self.align_chk = QCheckBox("Align covariate rasters to a common grid before modelling "
                                    "(resamples all to match the first checked raster)")
        self.align_chk.setChecked(True)
        form.addRow("", self.align_chk)

        # --- Point-data scenario: use a random portion of the points ---
        scen_box = QGroupBox("Point-data scenario (use a random portion of the points)")
        scen_form = QFormLayout(scen_box)

        scen_note = QLabel(
            "Randomly keep a chosen percentage of the points above (only points with a "
            "valid geometry and a non-null value in the selected field are counted). The "
            "subset is saved locally as a GeoPackage (plus a .json file describing it), "
            "added to QGIS under the 'SAMKit scenarios' group and, if ticked, used as the "
            "input points for modelling. The same percentage and seed always give the same "
            "subset, so scenarios are reproducible."
        )
        scen_note.setWordWrap(True)
        scen_note.setStyleSheet("color: gray; font-size: 10px;")
        scen_form.addRow(scen_note)

        pct_row = QHBoxLayout()
        self.scen_pct_spin = QSpinBox()
        self.scen_pct_spin.setRange(1, 100)
        self.scen_pct_spin.setValue(100)
        self.scen_pct_spin.setSuffix(" %")
        self.scen_pct_spin.valueChanged.connect(self._update_scenario_count_label)
        pct_row.addWidget(self.scen_pct_spin)
        for preset in (100, 75, 50, 25):
            b = QPushButton(f"{preset}%")
            b.setMaximumWidth(52)
            b.clicked.connect(lambda _=False, v=preset: self.scen_pct_spin.setValue(v))
            pct_row.addWidget(b)
        pct_row.addStretch()
        scen_form.addRow("Percentage of points to keep:", pct_row)

        self.scen_count_label = QLabel("Select a point layer and field.")
        self.scen_count_label.setStyleSheet("color: gray;")
        scen_form.addRow("", self.scen_count_label)

        seed_row = QHBoxLayout()
        self.scen_seed_spin = QSpinBox()
        self.scen_seed_spin.setRange(0, 999999)
        self.scen_seed_spin.setValue(42)
        seed_row.addWidget(self.scen_seed_spin)
        new_seed_btn = QPushButton("New random seed")
        new_seed_btn.clicked.connect(
            lambda: self.scen_seed_spin.setValue(random.randint(0, 999999)))
        seed_row.addWidget(new_seed_btn)
        seed_row.addStretch()
        scen_form.addRow("Random seed:", seed_row)

        self.scen_name_edit = QLineEdit()
        self.scen_name_edit.setPlaceholderText("automatic, e.g. <layer>_50pct_seed42")
        scen_form.addRow("Scenario name (optional):", self.scen_name_edit)

        dir_row = QHBoxLayout()
        self.scen_dir_edit = QLineEdit()
        self.scen_dir_edit.textEdited.connect(self._on_scenario_dir_edited)
        self.scen_dir_edit.editingFinished.connect(self._refresh_scenario_list)
        dir_row.addWidget(self.scen_dir_edit)
        scen_dir_btn = QPushButton("Browse…")
        scen_dir_btn.clicked.connect(self._browse_scenario_dir)
        dir_row.addWidget(scen_dir_btn)
        scen_form.addRow("Save scenarios in folder:", dir_row)

        self.scen_heldout_chk = QCheckBox(
            "Also save the points that were NOT kept (e.g. for use as external validation)")
        scen_form.addRow("", self.scen_heldout_chk)

        self.scen_use_chk = QCheckBox("Use the scenario as the input points layer above")
        self.scen_use_chk.setChecked(True)
        scen_form.addRow("", self.scen_use_chk)

        create_row = QHBoxLayout()
        self.scen_create_btn = QPushButton("Create scenario, save and load in QGIS")
        self.scen_create_btn.clicked.connect(self._on_create_scenario)
        create_row.addWidget(self.scen_create_btn)
        create_row.addStretch()
        scen_form.addRow("", create_row)

        saved_row = QHBoxLayout()
        self.scen_saved_combo = QComboBox()
        self.scen_saved_combo.setMinimumWidth(220)
        saved_row.addWidget(self.scen_saved_combo, 1)
        scen_refresh_btn = QPushButton("Refresh")
        scen_refresh_btn.clicked.connect(self._refresh_scenario_list)
        saved_row.addWidget(scen_refresh_btn)
        scen_load_btn = QPushButton("Load in QGIS")
        scen_load_btn.clicked.connect(self._on_load_saved_scenario)
        saved_row.addWidget(scen_load_btn)
        scen_form.addRow("Saved scenarios:", saved_row)

        form.addRow(scen_box)
        self.field_combo.currentTextChanged.connect(self._on_scenario_field_changed)

        layout.addWidget(data_box)

        # --- Covariate correlation analysis group ---
        corr_box = QGroupBox("Covariate correlation analysis (explore before modelling)")
        corr_layout = QVBoxLayout(corr_box)
        corr_note = QLabel(
            "Computes Pearson correlation between the target attribute and each checked "
            "covariate, and between the covariates themselves (multicollinearity check), "
            "using the values sampled at your sample points above. Use this to help decide "
            "which covariates are actually worth including before running a model."
        )
        corr_note.setWordWrap(True)
        corr_note.setStyleSheet("color: gray; font-size: 10px;")
        corr_layout.addWidget(corr_note)
        corr_btn_row = QHBoxLayout()
        self.corr_btn = QPushButton("Compute correlation table && heatmap")
        self.corr_btn.clicked.connect(self._on_compute_correlation)
        corr_btn_row.addWidget(self.corr_btn)
        corr_btn_row.addStretch()
        corr_layout.addLayout(corr_btn_row)
        layout.addWidget(corr_box)

        # --- Model group ---
        model_box = QGroupBox("Model")
        model_layout = QVBoxLayout(model_box)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Spatial predictive model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "Random Forest",
            "Support Vector Machine (SVM)",
            "Regression Kriging",
            "Ordinary Kriging",
        ])
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        model_row.addWidget(self.model_combo)
        model_row.addStretch()
        model_layout.addLayout(model_row)

        self.params_stack = QStackedWidget()
        model_layout.addWidget(self.params_stack)

        # RF params
        rf_w = QWidget()
        rf_form = QFormLayout(rf_w)
        self.rf_n_estimators = QSpinBox()
        self.rf_n_estimators.setRange(10, 5000)
        self.rf_n_estimators.setValue(500)
        self.rf_preset_combo = QComboBox()
        self.rf_preset_combo.addItems([
            "R randomForest defaults (mtry = p/3, nodesize 5)",
            "R ranger defaults (mtry = sqrt(p), min.node.size 5)",
            "pre-1.0 SAMKit (all features per split, min leaf 2)",
        ])
        self.rf_preset_combo.currentIndexChanged.connect(
            lambda i: self.rf_n_estimators.setValue(300 if i == 2 else 500))
        rf_form.addRow("Settings preset:", self.rf_preset_combo)
        rf_form.addRow("n_estimators:", self.rf_n_estimators)
        self.rf_max_depth = QSpinBox()
        self.rf_max_depth.setRange(0, 100)
        self.rf_max_depth.setValue(0)
        self.rf_max_depth.setToolTip("0 = no limit")
        rf_form.addRow("max_depth (0=none):", self.rf_max_depth)
        self.params_stack.addWidget(rf_w)

        # SVM params
        svm_w = QWidget()
        svm_form = QFormLayout(svm_w)
        self.svm_kernel = QComboBox()
        self.svm_kernel.addItems(["rbf", "linear", "poly", "sigmoid"])
        svm_form.addRow("kernel:", self.svm_kernel)
        self.svm_C = QDoubleSpinBox()
        self.svm_C.setRange(0.001, 10000)
        self.svm_C.setDecimals(3)
        self.svm_C.setValue(1.0)
        svm_form.addRow("C (regularization):", self.svm_C)
        self.svm_epsilon = QDoubleSpinBox()
        self.svm_epsilon.setRange(0.0, 100.0)
        self.svm_epsilon.setDecimals(3)
        self.svm_epsilon.setValue(0.1)
        svm_form.addRow("epsilon:", self.svm_epsilon)
        self.svm_gamma = QComboBox()
        self.svm_gamma.addItems(["scale", "auto"])
        svm_form.addRow("gamma:", self.svm_gamma)
        svm_note = QLabel("Predictors are automatically standardized before fitting "
                          "(SVMs are sensitive to feature scale).")
        svm_note.setWordWrap(True)
        svm_note.setStyleSheet("color: gray; font-size: 10px;")
        svm_form.addRow("", svm_note)
        self.params_stack.addWidget(svm_w)

        # Regression Kriging params
        rk_w = QWidget()
        rk_form = QFormLayout(rk_w)
        self.rk_trend_combo = QComboBox()
        self.rk_trend_combo.addItems(["random_forest", "linear"])
        rk_form.addRow("Trend model:", self.rk_trend_combo)
        self.rk_resid_combo = QComboBox()
        self.rk_resid_combo.addItems(["out-of-bag RF residuals (recommended; R practice)",
                                      "in-bag RF residuals (pre-1.0 SAMKit)"])
        rk_form.addRow("RF trend residuals:", self.rk_resid_combo)
        rk_note0 = QLabel("2-step: fits the trend model above, then krige-interpolates its "
                          "residuals and adds them back (a manual regression-kriging pipeline).")
        rk_note0.setWordWrap(True)
        rk_note0.setStyleSheet("color: gray; font-size: 10px;")
        rk_form.addRow("", rk_note0)
        self._build_variogram_controls("rk", rk_form, label_suffix=" (residuals)")
        self.params_stack.addWidget(rk_w)

        # Ordinary Kriging params
        ok_w = QWidget()
        ok_form = QFormLayout(ok_w)
        self._build_variogram_controls("ok", ok_form)
        self.params_stack.addWidget(ok_w)

        layout.addWidget(model_box)

        # --- Validation strategy group ---
        val_box = QGroupBox("Validation strategy")
        val_layout = QVBoxLayout(val_box)

        val_method_row = QHBoxLayout()
        val_method_row.addWidget(QLabel("Split calibration/validation data using:"))
        self.val_method_combo = QComboBox()
        self.val_method_combo.addItems([
            "k-fold cross-validation (random)",
            "Spatial block (calibration polygon)",
            "External validation layer (separate calibration/validation point layers)",
        ])
        self.val_method_combo.currentIndexChanged.connect(self._on_val_method_changed)
        val_method_row.addWidget(self.val_method_combo)
        val_method_row.addStretch()
        val_layout.addLayout(val_method_row)

        self.val_stack = QStackedWidget()
        val_layout.addWidget(self.val_stack)

        # k-fold page
        kfold_w = QWidget()
        kfold_layout = QVBoxLayout(kfold_w)
        eval_row = QHBoxLayout()
        self.eval_chk = QCheckBox("Validate with k-fold cross-validation")
        self.eval_chk.setChecked(True)
        self.eval_folds = QSpinBox()
        self.eval_folds.setRange(2, 20)
        self.eval_folds.setValue(10)
        eval_row.addWidget(self.eval_chk)
        eval_row.addWidget(QLabel("folds (k):"))
        eval_row.addWidget(self.eval_folds)
        eval_row.addStretch()
        kfold_layout.addLayout(eval_row)

        # Optional: fixed calibration % per fold (repeated random splits)
        cal_row = QHBoxLayout()
        self.eval_cal_pct_chk = QCheckBox("Use a fixed calibration percentage in every fold:")
        self.eval_cal_pct_chk.setChecked(False)
        self.eval_cal_pct_chk.setToolTip(
            "When ticked, each of the k folds randomly picks this percentage of the points "
            "for calibration (training) and uses the rest for validation (testing), "
            "independently each time (repeated random splits / Monte Carlo CV). "
            "When unticked, standard k-fold is used (1/k of the points held out per fold).")
        self.eval_cal_pct = QSpinBox()
        self.eval_cal_pct.setRange(10, 95)
        self.eval_cal_pct.setSingleStep(5)
        self.eval_cal_pct.setValue(70)
        self.eval_cal_pct.setSuffix(" %")
        self.eval_cal_pct.setEnabled(False)
        self.eval_cal_split_lbl = QLabel()
        self.eval_cal_split_lbl.setStyleSheet("color: gray;")
        cal_row.addWidget(self.eval_cal_pct_chk)
        cal_row.addWidget(self.eval_cal_pct)
        cal_row.addWidget(self.eval_cal_split_lbl)
        cal_row.addStretch()
        kfold_layout.addLayout(cal_row)
        self.eval_cal_pct_chk.toggled.connect(self._update_eval_split_label)
        self.eval_cal_pct.valueChanged.connect(self._update_eval_split_label)
        self.eval_folds.valueChanged.connect(self._update_eval_split_label)
        self._update_eval_split_label()

        eval_note = QLabel(
            "k-fold cross-validation trains on k-1 folds and predicts the held-out fold, "
            "repeated for every fold, then pools all held-out predictions into one R2/RMSE/MAE. "
            "This is much more stable than a single random train/test split, especially for the "
            "small sample sizes typical of field-collected soil/environmental data. Every point "
            "gets an honest observed/predicted pair this way -- the prediction made while that "
            "point was held out -- so the observed-vs-predicted table covers your whole dataset. "
            "The exact calibration/validation membership of every fold is saved as well.\n"
            "With a fixed calibration percentage ticked, each fold instead draws its own random "
            "calibration set of that size (e.g. 70% calibration / 30% validation, repeated k "
            "times); a point can then be validated in several folds, and every one of those "
            "held-out predictions is pooled and listed with its fold number."
        )
        eval_note.setWordWrap(True)
        eval_note.setStyleSheet("color: gray; font-size: 10px;")
        kfold_layout.addWidget(eval_note)
        self.val_stack.addWidget(kfold_w)

        # spatial block page
        spatial_w = QWidget()
        spatial_layout = QVBoxLayout(spatial_w)

        draw_row = QHBoxLayout()
        self.draw_poly_btn = QPushButton("Draw calibration polygon on map…")
        self.draw_poly_btn.clicked.connect(self._on_draw_calibration_polygon)
        draw_row.addWidget(self.draw_poly_btn)
        self.draw_status_label = QLabel("no polygon drawn yet")
        self.draw_status_label.setStyleSheet("color: gray;")
        draw_row.addWidget(self.draw_status_label)
        draw_row.addStretch()
        spatial_layout.addLayout(draw_row)

        draw_note = QLabel(
            "Click the button, then left-click on the QGIS map canvas to add polygon vertices. "
            "Double-click, right-click or press Enter to finish (at least 3 vertices); "
            "Backspace removes the last vertex; Esc cancels. This creates a small layer named "
            "'SAMKit_calibration_region' and selects it below automatically. Drawing again "
            "replaces it."
        )
        draw_note.setWordWrap(True)
        draw_note.setStyleSheet("color: gray; font-size: 10px;")
        spatial_layout.addWidget(draw_note)

        spatial_form = QFormLayout()
        self.calib_polygon_combo = QgsMapLayerComboBox()
        self.calib_polygon_combo.setFilters(QgsMapLayerProxyModel.PolygonLayer)
        spatial_form.addRow("Calibration region (polygon layer):", self.calib_polygon_combo)
        self.calib_inside_combo = QComboBox()
        self.calib_inside_combo.addItems([
            "Points INSIDE polygon = calibration, rest = validation",
            "Points OUTSIDE polygon = calibration, rest = validation",
        ])
        spatial_form.addRow("", self.calib_inside_combo)
        spatial_layout.addLayout(spatial_form)
        spatial_note = QLabel(
            "Either draw a polygon above, or choose any existing/already-digitized polygon "
            "layer here directly (e.g. a shapefile of management zones, or several polygons "
            "digitized with QGIS's own editing tools) and select it here. "
            "Points inside the polygon are used to fit the model; the rest are held out to "
            "compute R2/RMSE/MAE, giving a more honest estimate of how well the model "
            "extrapolates to unsampled areas than random k-fold CV does for spatially "
            "clustered data. Both subsets, and the observed vs predicted values for the "
            "held-out points, are saved to the run's results folder."
        )
        spatial_note.setWordWrap(True)
        spatial_note.setStyleSheet("color: gray; font-size: 10px;")
        spatial_layout.addWidget(spatial_note)
        self.val_stack.addWidget(spatial_w)

        # external validation layer page
        extval_w = QWidget()
        extval_layout = QVBoxLayout(extval_w)
        extval_form = QFormLayout()
        self.extval_points_combo = QgsMapLayerComboBox()
        self.extval_points_combo.setFilters(QgsMapLayerProxyModel.PointLayer)
        self.extval_points_combo.layerChanged.connect(self._on_extval_layer_changed)
        extval_form.addRow("Validation points (separate layer):", self.extval_points_combo)
        self.extval_field_combo = QComboBox()
        extval_form.addRow("Validation attribute field:", self.extval_field_combo)
        self.extval_same_field_chk = QCheckBox("Same field name as the calibration attribute above")
        self.extval_same_field_chk.setChecked(True)
        self.extval_same_field_chk.toggled.connect(
            lambda checked: self.extval_field_combo.setEnabled(not checked))
        self.extval_field_combo.setEnabled(False)
        extval_form.addRow("", self.extval_same_field_chk)
        extval_layout.addLayout(extval_form)
        extval_note = QLabel(
            "The 'Georeferenced points' layer in Input data above is used entirely for "
            "calibration (fit on all of it, no internal split). This separate layer is used "
            "entirely for validation -- exactly the fixed-external-validation-set design common "
            "in repeated-calibration benchmarking studies (calibrate on many different subsets, "
            "always validate against the same independent held-out points). Both point sets, and "
            "the observed vs predicted values for the validation points, are saved to the run's "
            "results folder."
        )
        extval_note.setWordWrap(True)
        extval_note.setStyleSheet("color: gray; font-size: 10px;")
        extval_layout.addWidget(extval_note)
        self.val_stack.addWidget(extval_w)

        layout.addWidget(val_box)

        # --- Output group ---
        out_box = QGroupBox("Output")
        out_form = QFormLayout(out_box)
        out_row = QHBoxLayout()
        self.out_path_edit = QLineEdit()
        out_browse = QPushButton("Browse…")
        out_browse.clicked.connect(self._browse_output_path)
        out_row.addWidget(self.out_path_edit)
        out_row.addWidget(out_browse)
        out_form.addRow("Output raster (GeoTIFF):", out_row)
        self.load_output_chk = QCheckBox("Load output raster into QGIS after processing")
        self.load_output_chk.setChecked(True)
        out_form.addRow("", self.load_output_chk)
        out_note = QLabel(
            "Each run collects everything into one folder next to the raster, "
            "'<output name>_SAMKit_results':\n"
            "  00_RUN_SUMMARY.txt:  what was run, the accuracy, and where each file went\n"
            "  01_accuracy/:  metrics, per-fold accuracy, and an observed-vs-predicted plot\n"
            "  02_observed_vs_predicted/:  the observed and predicted value of the attribute "
            "of interest at every point, with residuals\n"
            "  03_input_samples/:  the extracted training table and a target/covariate summary\n"
            "  04_calibration_validation_split/:  exactly which points were used for what "
            "(plus the calibration region boundary for spatial-block runs)\n"
            "  05_maps/:  publication-quality maps of the prediction (and kriging variance), "
            "if enabled below\n"
            "A cumulative performance log (samkit_model_performance_log.csv) is also appended "
            "next to the raster so runs can be compared over time, and the results open in a "
            "table window as soon as the run finishes."
        )
        out_note.setWordWrap(True)
        out_note.setStyleSheet("color: gray; font-size: 10px;")
        out_form.addRow("", out_note)
        layout.addWidget(out_box)

        # --- Publication maps group ---
        self._build_map_export_controls(layout)

        run_row = QHBoxLayout()
        self.run_btn = QPushButton("Run spatial predictive model")
        self.run_btn.clicked.connect(self.run)
        run_row.addWidget(self.run_btn)
        layout.addLayout(run_row)

        self.progress = QProgressBar()
        layout.addWidget(self.progress)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(4000)
        layout.addWidget(self.log)

        self._on_points_layer_changed(self.points_combo.currentLayer())

    # ------------------------------------------------------------------
    # Publication-quality maps
    # ------------------------------------------------------------------
    def _build_map_export_controls(self, layout):
        from qgis.core import QgsSettings
        settings = QgsSettings()

        box = QGroupBox("Publication maps (saved automatically after every run)")
        form = QFormLayout(box)

        self.maps_chk = QCheckBox("Save publication-quality maps of the prediction "
                                  "(and of the kriging variance, for kriging models)")
        self.maps_chk.setChecked(settings.value("samkit/maps_enabled", True, type=bool))
        form.addRow("", self.maps_chk)
        self.maps_chk.toggled.connect(
            lambda on: QgsSettings().setValue("samkit/maps_enabled", bool(on)))

        self.map_title_edit = QLineEdit()
        self.map_title_edit.setPlaceholderText("automatic: Predicted <attribute>")
        form.addRow("Map title (optional):", self.map_title_edit)

        self.map_units_edit = QLineEdit()
        self.map_units_edit.setPlaceholderText("e.g.  %,  g/kg,  mg/kg,  dS/m  (shown in the legend)")
        form.addRow("Units of the attribute:", self.map_units_edit)

        self.map_author_edit = QLineEdit(settings.value("samkit/map_author", "", type=str))
        self.map_author_edit.setPlaceholderText("name / organisation for the map credit "
                                                "(remembered)")
        form.addRow("Author / credit:", self.map_author_edit)

        style_row = QHBoxLayout()
        self.map_ramp_combo = QComboBox()
        self.map_ramp_combo.addItems(list(self._map_export().COLOR_RAMPS.keys()))
        self.map_ramp_combo.setToolTip(
            "All ramps are perceptually uniform and colour-vision-deficiency safe. "
            "Viridis is the default; Cividis is optimised for red-green colour blindness; "
            "Greyscale is for black-and-white print.")
        style_row.addWidget(self.map_ramp_combo)
        self.map_reverse_chk = QCheckBox("reverse")
        style_row.addWidget(self.map_reverse_chk)
        self.map_class_combo = QComboBox()
        self.map_class_combo.addItems(["Continuous colour bar", "Classed (rounded breaks)"])
        style_row.addWidget(self.map_class_combo)
        style_row.addStretch()
        form.addRow("Colour ramp / legend:", style_row)

        out_row = QHBoxLayout()
        self.map_page_combo = QComboBox()
        self.map_page_combo.addItems(list(self._map_export().PAGE_SIZES_MM.keys()))
        out_row.addWidget(self.map_page_combo)
        self.map_dpi_spin = QSpinBox()
        self.map_dpi_spin.setRange(96, 1200)
        self.map_dpi_spin.setSingleStep(50)
        self.map_dpi_spin.setValue(300)
        self.map_dpi_spin.setSuffix(" dpi")
        out_row.addWidget(self.map_dpi_spin)
        self.map_png_chk = QCheckBox("PNG")
        self.map_png_chk.setChecked(True)
        self.map_pdf_chk = QCheckBox("PDF (vector)")
        self.map_pdf_chk.setChecked(True)
        self.map_svg_chk = QCheckBox("SVG")
        for w in (self.map_png_chk, self.map_pdf_chk, self.map_svg_chk):
            out_row.addWidget(w)
        out_row.addStretch()
        form.addRow("Page / resolution / formats:", out_row)

        me = self._map_export()
        self.map_grid_combo = QComboBox()
        self._grid_keys = list(me.GRID_LABELS.keys())
        self.map_grid_combo.addItems([me.GRID_LABELS[k] for k in self._grid_keys])
        saved = settings.value("samkit/map_grid_labels", me.DEFAULT_GRID_LABELS, type=str)
        self.map_grid_combo.setCurrentIndex(
            self._grid_keys.index(saved) if saved in self._grid_keys else 0)
        self.map_grid_combo.setToolTip(
            "Latitude/longitude: a WGS 84 graticule labelled e.g. 39\u00b002\u203231\u2033N, "
            "85\u00b031\u203255\u2033W, drawn over the map whatever its CRS.\n"
            "Projected: the map's own grid in metres, e.g. 4,323,000 m N.")
        self.map_grid_combo.currentIndexChanged.connect(
            lambda i: QgsSettings().setValue("samkit/map_grid_labels", self._grid_keys[i]))
        form.addRow("Coordinate grid labels:", self.map_grid_combo)

        self.map_points_chk = QCheckBox("Show the sample locations (calibration / validation "
                                        "points are symbolised separately)")
        self.map_points_chk.setChecked(True)
        form.addRow("", self.map_points_chk)
        self.map_layouts_chk = QCheckBox("Also add the map layouts to this project's Layout "
                                         "Manager for hand-editing (needs 'Load output raster "
                                         "into QGIS')")
        self.map_layouts_chk.setChecked(True)
        form.addRow("", self.map_layouts_chk)

        note = QLabel(
            "Each map is a full cartographic layout: title, map body in a neatline at a "
            "rounded scale, labelled coordinate grid in the raster's CRS, legend with units, "
            "sample locations, scale bar and representative fraction, north arrow (true "
            "north), locator inset, and a metadata block (CRS/EPSG, projection, cell size, "
            "model, validation accuracy, covariates, author, date). PNG is written at the "
            "chosen dpi (300 dpi is the usual journal minimum; 600 for line art); PDF keeps "
            "text and linework as vectors. Files go to '05_maps' in the run's results folder. "
            "The loaded prediction layer gets the same styling in QGIS.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray; font-size: 10px;")
        form.addRow("", note)

        self.maps_chk.toggled.connect(lambda on: [w.setEnabled(on) for w in (
            self.map_title_edit, self.map_units_edit, self.map_author_edit,
            self.map_ramp_combo, self.map_reverse_chk, self.map_class_combo,
            self.map_page_combo, self.map_dpi_spin, self.map_png_chk, self.map_pdf_chk,
            self.map_svg_chk, self.map_points_chk, self.map_layouts_chk,
            self.map_grid_combo)])
        layout.addWidget(box)

    @staticmethod
    def _map_export():
        from . import map_export
        return map_export

    def _map_formats(self):
        fmts = []
        if self.map_png_chk.isChecked():
            fmts.append("png")
        if self.map_pdf_chk.isChecked():
            fmts.append("pdf")
        if self.map_svg_chk.isChecked():
            fmts.append("svg")
        return fmts

    def _export_run_maps(self, paths, out_path, variance_path, field_name, model_name,
                         validation_label, metrics, n_samples, feature_names, points,
                         pred_layer, var_layer, register_layouts):
        """Write the publication maps for this run; never lets a map failure fail the run."""
        from qgis.core import QgsSettings
        formats = self._map_formats()
        if not formats:
            self._log("(Publication maps: no output format ticked -- skipped.)")
            return []
        settings = QgsSettings()
        settings.setValue("samkit/map_author", self.map_author_edit.text().strip())
        settings.setValue("samkit/maps_enabled", True)
        me = self._map_export()
        maps_dir = paths["maps"]
        self._log("")
        self._log("Rendering publication maps…")
        QApplication.processEvents()
        files = me.export_run_maps(
            out_path, maps_dir,
            variable=field_name,
            units=self.map_units_edit.text().strip(),
            model_name=model_name,
            validation_label=validation_label,
            metrics=metrics,
            n_samples=n_samples,
            feature_names=feature_names,
            points=points if self.map_points_chk.isChecked() else None,
            variance_path=variance_path,
            ramp=self.map_ramp_combo.currentText(),
            reverse_ramp=self.map_reverse_chk.isChecked(),
            classification="classed" if self.map_class_combo.currentIndex() == 1 else "continuous",
            page=self.map_page_combo.currentText(),
            dpi=self.map_dpi_spin.value(),
            formats=formats,
            author=self.map_author_edit.text().strip(),
            title=self.map_title_edit.text().strip() or None,
            pred_layer=pred_layer,
            var_layer=var_layer,
            register_layouts=register_layouts,
            log=self._log,
            grid_labels=self._grid_keys[self.map_grid_combo.currentIndex()],
        )
        self._last_map_points_layer = getattr(files, "points_layer", None)
        return files

    def _add_map_point_layer_to_project(self, pred_layer):
        """Put the maps' sample-location layer (a GeoPackage in 05_maps) into the
        project, just above the prediction, so the registered layouts keep
        working after the project is saved and reopened."""
        lyr = getattr(self, "_last_map_points_layer", None)
        if lyr is None or not lyr.isValid():
            return
        try:
            # Do NOT rename the layer here: the layouts already reference it,
            # and its name is the legend label. Add it as it is.
            QgsProject.instance().addMapLayer(lyr)
        except Exception as e:
            self._log(f"  (could not add the sample-location layer to the project: {e})")

    def _build_variogram_controls(self, prefix, form_layout, label_suffix=""):
        """
        Builds and registers (as self.<prefix>_variogram_combo / _auto_chk /
        _nugget / _sill / _range) the variogram-model selector plus an
        optional manual nugget/sill/range override, shared by the OK, RK, and
        UK parameter pages. Auto-fit (matching pykrige/gstat's default
        behaviour) is the default; unchecking it enables manual entry.
        """
        combo = QComboBox()
        combo.addItems(["linear", "power", "gaussian", "spherical", "exponential"])
        combo.setCurrentText("spherical")
        form_layout.addRow(f"Variogram model{label_suffix}:", combo)
        setattr(self, f"{prefix}_variogram_combo", combo)
        fit_combo = QComboBox()
        fit_combo.addItems(["gstat-compatible (gstat sample variogram + fit.method 7 WLS)",
                            "PyKrige auto-fit (pre-1.0 SAMKit)"])
        form_layout.addRow("Variogram fitting:", fit_combo)
        setattr(self, f"{prefix}_fit_combo", fit_combo)

        auto_chk = QCheckBox("Auto-fit nugget/sill/range from data (recommended)")
        auto_chk.setChecked(True)
        form_layout.addRow("", auto_chk)
        setattr(self, f"{prefix}_auto_chk", auto_chk)

        nugget = QDoubleSpinBox()
        nugget.setRange(0, 1e6)
        nugget.setDecimals(4)
        nugget.setValue(0.0)
        sill = QDoubleSpinBox()
        sill.setRange(0, 1e9)
        sill.setDecimals(4)
        sill.setValue(1.0)
        range_box = QDoubleSpinBox()
        range_box.setRange(0, 1e9)
        range_box.setDecimals(2)
        range_box.setValue(100.0)
        form_layout.addRow("  nugget:", nugget)
        form_layout.addRow("  sill (nugget + partial sill):", sill)
        form_layout.addRow("  range:", range_box)
        setattr(self, f"{prefix}_nugget", nugget)
        setattr(self, f"{prefix}_sill", sill)
        setattr(self, f"{prefix}_range", range_box)

        def _toggle(auto_checked):
            for w in (nugget, sill, range_box):
                w.setEnabled(not auto_checked)
        auto_chk.toggled.connect(_toggle)
        _toggle(True)

        note = QLabel(
            "Manual values apply to spherical/exponential/gaussian models only (ignored, with "
            "a log warning, for linear/power -- those auto-fit instead). If the chosen model "
            "fits a degenerate (near-zero sill) variogram, SAMKit automatically falls back to a "
            "linear variogram model."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: gray; font-size: 9px;")
        form_layout.addRow("", note)

    def _rf_preset(self):
        return ("r_randomforest", "ranger", "samkit_legacy")[self.rf_preset_combo.currentIndex()]

    def _vfit(self, prefix):
        return "pykrige" if getattr(self, f"{prefix}_fit_combo").currentIndex() == 1 else "gstat"

    def _rk_residuals(self):
        return "inbag" if self.rk_resid_combo.currentIndex() == 1 else "oob"

    def _warn_if_geographic(self, ref_ri, model_name):
        """Kriging distances are only meaningful in a projected (metric) CRS."""
        try:
            from osgeo import osr
            srs = osr.SpatialReference(); srs.ImportFromWkt(ref_ri.projection_wkt)
            if srs.IsGeographic():
                self._log("WARNING: the covariate rasters are in a geographic CRS (degrees). "
                          "Terrain covariates computed from a DEM in degrees are invalid "
                          "(slope near 90 degrees everywhere, MRVBF/MRRTF 0 or nodata, "
                          "distorted curvature/TPI/TWI). Project the DEM to metres in tab 1 "
                          "and recompute the covariates in tab 2.")
            if srs.IsGeographic() and "Kriging" in model_name:
                self._log("WARNING: the reference raster CRS is geographic (degrees). Kriging "
                          "distances and variogram ranges will be in degrees and anisotropic "
                          "(1 deg longitude != 1 deg latitude). Reproject the DEM/covariates to a "
                          "projected CRS (e.g. UTM via the DEM tab) before kriging.")
                QMessageBox.warning(self, "Spatial Predictive Modeling",
                                    "The covariate grid is in a geographic CRS (degrees).\n"
                                    "Kriging distances would be computed in degrees. Reproject to a "
                                    "projected CRS (e.g. UTM) first for valid kriging results.")
        except Exception:
            pass

    def _get_variogram_parameters(self, prefix, variogram_model, log):
        """Returns a manual {'nugget','sill','range'} dict if the auto-fit checkbox for
        `prefix` is unchecked and the model supports it, otherwise None (auto-fit)."""
        auto_chk = getattr(self, f"{prefix}_auto_chk")
        if auto_chk.isChecked():
            return None
        if variogram_model not in ("spherical", "exponential", "gaussian"):
            log(f"Manual variogram parameters are only supported for spherical/exponential/"
                f"gaussian models; auto-fitting '{variogram_model}' instead.")
            return None
        return {
            "nugget": getattr(self, f"{prefix}_nugget").value(),
            "sill": getattr(self, f"{prefix}_sill").value(),
            "range": getattr(self, f"{prefix}_range").value(),
        }

    def _on_model_changed(self, idx):
        self.params_stack.setCurrentIndex(idx)
        # Ordinary Kriging does not need covariate rasters for prediction itself,
        # but we still use one raster to define the output grid.
        self.raster_list.setEnabled(True)
        if idx == 3:
            self.raster_list.setToolTip(
                "For Ordinary Kriging, check at least one raster only to define the output "
                "grid (extent/resolution) - its pixel values themselves are not used as predictors.")
        else:
            self.raster_list.setToolTip("")

    def _on_val_method_changed(self, idx):
        self.val_stack.setCurrentIndex(idx)

    def _update_eval_split_label(self, *args):
        on = self.eval_cal_pct_chk.isChecked()
        self.eval_cal_pct.setEnabled(on)
        k = self.eval_folds.value()
        if on:
            cal = self.eval_cal_pct.value()
            self.eval_cal_split_lbl.setText(
                f"→ {cal}% calibration / {100 - cal}% validation, {k} random repetitions")
        else:
            self.eval_cal_split_lbl.setText(
                f"(standard k-fold: {100 - 100 / k:.1f}% calibration / {100 / k:.1f}% validation)")

    def _on_draw_calibration_polygon(self):
        canvas = self.iface.mapCanvas()
        if canvas is None:
            QMessageBox.warning(self, "Draw calibration polygon", "No map canvas available.")
            return
        # Don't remember our own tool as the "previous" one if the button is hit twice.
        current = canvas.mapTool()
        if not isinstance(current, map_tools.PolygonMapTool):
            self._prev_map_tool = current
        self._poly_map_tool = map_tools.PolygonMapTool(canvas)
        self._poly_map_tool.polygonFinished.connect(self._on_polygon_drawn)
        self._poly_map_tool.drawingCanceled.connect(self._on_polygon_draw_canceled)
        canvas.setMapTool(self._poly_map_tool)
        canvas.setFocus()
        self.draw_status_label.setText("drawing… click vertices on the map; double-click, "
                                       "right-click or Enter to finish (Esc cancels)")
        self.draw_status_label.setStyleSheet("color: #a04000;")

    def _restore_prev_map_tool(self):
        canvas = self.iface.mapCanvas()
        if canvas is not None and self._prev_map_tool is not None:
            canvas.setMapTool(self._prev_map_tool)
        self._poly_map_tool = None

    def _on_polygon_draw_canceled(self):
        self.draw_status_label.setText("drawing cancelled")
        self.draw_status_label.setStyleSheet("color: gray;")
        self._restore_prev_map_tool()

    def _on_polygon_drawn(self, geom):
        canvas = self.iface.mapCanvas()
        try:
            canvas_crs = canvas.mapSettings().destinationCrs()
            uri = f"MultiPolygon?crs={canvas_crs.authid()}"
            layer_name = "SAMKit_calibration_region"
            mem_layer = QgsVectorLayer(uri, layer_name, "memory")
            provider = mem_layer.dataProvider()
            geom = QgsGeometry(geom)
            geom.convertToMultiType()
            feat = QgsFeature()
            feat.setGeometry(geom)
            provider.addFeature(feat)
            mem_layer.updateExtents()

            # Replace any previously-drawn calibration layer from this session.
            if self._drawn_calib_layer_id is not None:
                existing = QgsProject.instance().mapLayer(self._drawn_calib_layer_id)
                if existing is not None:
                    QgsProject.instance().removeMapLayer(existing)

            QgsProject.instance().addMapLayer(mem_layer)
            self._drawn_calib_layer_id = mem_layer.id()
            self.calib_polygon_combo.setLayer(mem_layer)

            n_vertices = sum(len(ring) - 1 for poly in geom.asMultiPolygon() for ring in poly)
            bb = geom.boundingBox()
            self.draw_status_label.setText(
                f"polygon drawn: {n_vertices} vertices, "
                f"x [{bb.xMinimum():.2f}, {bb.xMaximum():.2f}], "
                f"y [{bb.yMinimum():.2f}, {bb.yMaximum():.2f}]")
            self.draw_status_label.setStyleSheet("color: green;")
        except Exception as e:
            QMessageBox.critical(self, "Draw calibration polygon",
                                 f"Could not create the calibration layer: {e}")
            self.draw_status_label.setText("drawing failed -- see error")
            self.draw_status_label.setStyleSheet("color: red;")
        finally:
            self._restore_prev_map_tool()

    def _on_points_layer_changed(self, layer):
        self.field_combo.clear()
        if layer is None or not isinstance(layer, QgsVectorLayer):
            self._on_scenario_field_changed()
            return
        for field in layer.fields():
            if field.isNumeric():
                self.field_combo.addItem(field.name())

        self._set_default_scenario_dir(layer)
        meta = self._scenario_meta(layer)
        if meta:
            # Show the scenario's own percentage/seed so the boxes match what is used.
            try:
                self.scen_pct_spin.setValue(int(meta.get("percent", 100)))
                self.scen_seed_spin.setValue(int(meta.get("seed", self.scen_seed_spin.value())))
            except (TypeError, ValueError):
                pass
        self._on_scenario_field_changed()

    def _on_extval_layer_changed(self, layer):
        self.extval_field_combo.clear()
        if layer is None or not isinstance(layer, QgsVectorLayer):
            return
        for field in layer.fields():
            if field.isNumeric():
                self.extval_field_combo.addItem(field.name())

    # ------------------------------------------------------------------
    # Point-data scenarios (random subsets of the input points)
    # ------------------------------------------------------------------
    SCENARIO_GROUP_NAME = "SAMKit scenarios"
    SCENARIO_PROP = "samkit/scenario"          # JSON metadata stored on the layer

    @staticmethod
    def _layer_file_path(layer):
        """Path of the file behind a vector layer, or '' for memory/db layers."""
        try:
            path = layer.source().split("|")[0]
        except Exception:
            return ""
        return path if os.path.isfile(path) else ""

    @staticmethod
    def _safe_name(text):
        text = re.sub(r"[^\w\-]+", "_", str(text).strip())
        return text.strip("_") or "points"

    def _scenario_meta(self, layer):
        """Return the scenario metadata dict of a layer, or None if it isn't one."""
        if layer is None:
            return None
        raw = layer.customProperty(self.SCENARIO_PROP, "")
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    def _set_default_scenario_dir(self, layer):
        if self._scen_dir_user_set:
            return
        src = self._layer_file_path(layer) if layer is not None else ""
        if src:
            base = os.path.dirname(src)
            folder = base if os.path.basename(base) == "samkit_scenarios" \
                else os.path.join(base, "samkit_scenarios")
        else:
            folder = os.path.join(os.path.expanduser("~"), "SAMKit_scenarios")
        self.scen_dir_edit.setText(folder)
        self._refresh_scenario_list()

    def _on_scenario_dir_edited(self, _text):
        self._scen_dir_user_set = True

    def _browse_scenario_dir(self):
        start = self.scen_dir_edit.text().strip() or os.path.expanduser("~")
        folder = QFileDialog.getExistingDirectory(self, "Folder for point-data scenarios", start)
        if folder:
            self.scen_dir_edit.setText(folder)
            self._scen_dir_user_set = True
            self._refresh_scenario_list()

    def _scenario_base_layer(self, layer):
        """
        The layer a new scenario should be sampled from. If the current points
        layer is itself a scenario, go back to its original (full) layer so that
        e.g. '50 %' always means 50 % of ALL points, not 50 % of a previous subset.
        """
        meta = self._scenario_meta(layer)
        if not meta:
            return layer, False
        proj = QgsProject.instance()
        src = proj.mapLayer(meta.get("source_layer_id", ""))
        if isinstance(src, QgsVectorLayer):
            return src, False
        src_path = os.path.normcase(os.path.abspath(meta.get("source_path", "") or "~none~"))
        for lyr in proj.mapLayers().values():
            if isinstance(lyr, QgsVectorLayer) and self._layer_file_path(lyr) and \
                    os.path.normcase(os.path.abspath(self._layer_file_path(lyr))) == src_path:
                return lyr, False
        return layer, True   # original not loaded -> fall back to the scenario itself

    def _eligible_fids(self, layer, field_name):
        """Feature ids with a valid geometry and (if a field is given) a numeric value."""
        fids = []
        idx = layer.fields().indexFromName(field_name) if field_name else -1
        req = QgsFeatureRequest()
        if idx < 0:
            req.setNoAttributes()
        else:
            req.setSubsetOfAttributes([idx])
        for feat in layer.getFeatures(req):
            geom = feat.geometry()
            if geom is None or geom.isEmpty():
                continue
            if idx >= 0:
                try:
                    if not np.isfinite(float(feat.attribute(idx))):
                        continue
                except (TypeError, ValueError):
                    continue
            fids.append(feat.id())
        return fids

    def _on_scenario_field_changed(self, *_args):
        layer = self.points_combo.currentLayer()
        self._scen_eligible_count = 0
        if isinstance(layer, QgsVectorLayer):
            base, _ = self._scenario_base_layer(layer)
            try:
                self._scen_eligible_count = len(
                    self._eligible_fids(base, self.field_combo.currentText()))
            except Exception:
                self._scen_eligible_count = 0
        self._update_scenario_count_label()

    @staticmethod
    def _n_keep(n_total, pct):
        return 0 if n_total <= 0 else max(1, min(n_total, int(round(n_total * pct / 100.0))))

    def _update_scenario_count_label(self, *_args):
        n = self._scen_eligible_count
        if n <= 0:
            self.scen_count_label.setText("No usable points (select a point layer and field).")
            return
        k = self._n_keep(n, self.scen_pct_spin.value())
        layer = self.points_combo.currentLayer()
        base, _ = self._scenario_base_layer(layer) if layer is not None else (None, False)
        src = f" of '{base.name()}'" if base is not None else ""
        self.scen_count_label.setText(
            f"{k} of {n} usable points{src} will be kept ({n - k} left out) when you click "
            f"'Create scenario'. Until then the model uses the selected point layer as it is.")

    def _write_points_subset(self, layer, fids, path, layer_name):
        """Write the given features of `layer` to a new GeoPackage at `path`.

        The features are first copied into a temporary memory layer (SaveVectorOptions
        has no fid filter in QGIS 3.16-3.22), which is then written to disk. All fields
        and the layer CRS / geometry type are kept.
        """
        # Keep every original attribute and add the original feature id, so
        # each scenario point can be traced back to the source layer even when
        # that layer is a shapefile or CSV (whose ids are not stored as a field).
        from qgis.core import QgsField, QgsFields
        from qgis.PyQt.QtCore import QVariant
        id_name = "source_fid"
        while layer.fields().indexFromName(id_name) >= 0:
            id_name = "_" + id_name
        out_fields = QgsFields(layer.fields())
        out_fields.append(QgsField(id_name, QVariant.LongLong))
        tmp = QgsMemoryProviderUtils.createMemoryLayer(
            layer_name, out_fields, layer.wkbType(), layer.crs())
        fid_list = [int(f) for f in fids]
        try:
            req = QgsFeatureRequest().setFilterFids(fid_list)
        except TypeError:   # some PyQGIS builds map QgsFeatureIds to a Python set
            req = QgsFeatureRequest().setFilterFids(set(fid_list))
        src_feats = list(layer.getFeatures(req))
        if len(src_feats) != len(fids):
            raise RuntimeError(f"Expected {len(fids)} features, read {len(src_feats)}.")
        feats = []
        for sf in src_feats:
            nf = QgsFeature(out_fields)
            nf.setGeometry(sf.geometry())
            nf.setAttributes(list(sf.attributes()) + [int(sf.id())])
            feats.append(nf)
        if not tmp.dataProvider().addFeatures(feats)[0]:
            raise RuntimeError("Could not copy the selected points.")
        tmp.updateExtents()

        opts = QgsVectorFileWriter.SaveVectorOptions()
        opts.driverName = "GPKG"
        opts.fileEncoding = "UTF-8"
        opts.layerName = layer_name
        ctx = QgsProject.instance().transformContext()
        if hasattr(QgsVectorFileWriter, "writeAsVectorFormatV3"):
            res = QgsVectorFileWriter.writeAsVectorFormatV3(tmp, path, ctx, opts)
        else:
            res = QgsVectorFileWriter.writeAsVectorFormatV2(tmp, path, ctx, opts)
        err = res[0] if isinstance(res, tuple) else res
        msg = res[1] if isinstance(res, tuple) and len(res) > 1 else ""
        if err != QgsVectorFileWriter.NoError:
            raise RuntimeError(f"Could not write '{path}': {msg or err}")

    def _write_polygon_layer(self, layer, path, layer_name):
        """Write every feature of polygon `layer` (e.g. the calibration region) to a
        GeoPackage at `path`, keeping its CRS, so the boundary used for a run is kept."""
        opts = QgsVectorFileWriter.SaveVectorOptions()
        opts.driverName = "GPKG"
        opts.fileEncoding = "UTF-8"
        opts.layerName = layer_name
        ctx = QgsProject.instance().transformContext()
        if hasattr(QgsVectorFileWriter, "writeAsVectorFormatV3"):
            res = QgsVectorFileWriter.writeAsVectorFormatV3(layer, path, ctx, opts)
        else:
            res = QgsVectorFileWriter.writeAsVectorFormatV2(layer, path, ctx, opts)
        err = res[0] if isinstance(res, tuple) else res
        msg = res[1] if isinstance(res, tuple) and len(res) > 1 else ""
        if err != QgsVectorFileWriter.NoError:
            raise RuntimeError(f"Could not write '{path}': {msg or err}")

    def _remove_layers_using(self, paths):
        """Remove project layers that read from any of `paths` (so they can be overwritten)."""
        targets = {os.path.normcase(os.path.abspath(p)) for p in paths}
        proj = QgsProject.instance()
        for lid, lyr in list(proj.mapLayers().items()):
            p = self._layer_file_path(lyr) if isinstance(lyr, QgsVectorLayer) else ""
            if p and os.path.normcase(os.path.abspath(p)) in targets:
                if self.points_combo.currentLayer() is lyr:
                    self.points_combo.setLayer(None)
                proj.removeMapLayer(lid)

    def _add_scenario_layer_to_qgis(self, path, name, meta):
        """Load a scenario GeoPackage into the 'SAMKit scenarios' layer-tree group."""
        lyr = QgsVectorLayer(path, name, "ogr")
        if not lyr.isValid():
            raise RuntimeError(f"QGIS could not load '{path}'.")
        slim = {k: v for k, v in meta.items()
                if k not in ("kept_source_fids", "_folder")}
        lyr.setCustomProperty(self.SCENARIO_PROP, json.dumps(slim))
        proj = QgsProject.instance()
        root = proj.layerTreeRoot()
        group = root.findGroup(self.SCENARIO_GROUP_NAME)
        if group is None:
            group = root.insertGroup(0, self.SCENARIO_GROUP_NAME)
        proj.addMapLayer(lyr, False)
        group.insertLayer(0, lyr)
        return lyr

    def _use_scenario_layer(self, lyr):
        """Make `lyr` the input points layer, keeping the selected field if possible."""
        field = self.field_combo.currentText()
        self.points_combo.setLayer(lyr)
        if field:
            i = self.field_combo.findText(field)
            if i >= 0:
                self.field_combo.setCurrentIndex(i)

    def _on_create_scenario(self):
        title = "Point-data scenario"
        layer = self.points_combo.currentLayer()
        field_name = self.field_combo.currentText()
        if not isinstance(layer, QgsVectorLayer):
            QMessageBox.warning(self, title, "Please select a point vector layer first.")
            return
        if not field_name:
            QMessageBox.warning(self, title, "Please select the attribute of interest (field) first.")
            return
        folder = self.scen_dir_edit.text().strip()
        if not folder:
            QMessageBox.warning(self, title, "Please choose a folder to save scenarios in.")
            return

        base, fallback = self._scenario_base_layer(layer)
        if fallback:
            self._log("Note: the original points layer of the selected scenario is not loaded "
                      "in QGIS, so the new scenario is drawn from the scenario layer itself.")

        fids = self._eligible_fids(base, field_name)
        n_total = len(fids)
        if n_total == 0:
            QMessageBox.warning(self, title, f"'{base.name()}' has no points with a valid "
                                             f"geometry and a value in '{field_name}'.")
            return
        pct = int(self.scen_pct_spin.value())
        seed = int(self.scen_seed_spin.value())
        n_keep = self._n_keep(n_total, pct)
        if n_keep < 5:
            if QMessageBox.question(
                    self, title,
                    f"This scenario keeps only {n_keep} point(s); modelling needs at least 5. "
                    f"Create it anyway?") != QMessageBox.Yes:
                return

        rng = np.random.RandomState(seed)
        keep_idx = np.sort(rng.choice(n_total, size=n_keep, replace=False))
        keep_fids = [int(fids[i]) for i in keep_idx]
        keep_set = set(keep_fids)
        held_fids = [int(f) for f in fids if f not in keep_set]

        name = self.scen_name_edit.text().strip() or \
            f"{base.name()}_{pct}pct_seed{seed}"
        name = self._safe_name(name)
        gpkg = os.path.join(folder, name + ".gpkg")
        held_gpkg = os.path.join(folder, name + "_heldout.gpkg")
        meta_path = os.path.join(folder, name + ".json")
        save_held = self.scen_heldout_chk.isChecked() and held_fids

        existing = [p for p in (gpkg, held_gpkg, meta_path) if os.path.exists(p)]
        if existing:
            if QMessageBox.question(
                    self, title,
                    f"A scenario called '{name}' already exists in\n{folder}\n\nOverwrite it?"
            ) != QMessageBox.Yes:
                return
            self._remove_layers_using(existing)

        meta = {
            "samkit_scenario": 1,
            "name": name,
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "source_layer_name": base.name(),
            "source_layer_id": base.id(),
            "source_path": self._layer_file_path(base),
            "field": field_name,
            "percent": pct,
            "seed": seed,
            "n_source_points": n_total,
            "n_kept": len(keep_fids),
            "n_heldout": len(held_fids),
            "scenario_file": os.path.basename(gpkg),
            "heldout_file": os.path.basename(held_gpkg) if save_held else "",
            "kept_source_fids": keep_fids,
        }

        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        try:
            os.makedirs(folder, exist_ok=True)
            for p in existing:
                try:
                    os.remove(p)
                except OSError:
                    pass
            self._write_points_subset(base, keep_fids, gpkg, name)
            if save_held:
                self._write_points_subset(base, held_fids, held_gpkg, name + "_heldout")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            self._append_scenario_index(folder, meta)

            lyr = self._add_scenario_layer_to_qgis(gpkg, name, meta)
            if save_held:
                held_meta = dict(meta, role="heldout")
                self._add_scenario_layer_to_qgis(held_gpkg, name + "_heldout", held_meta)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            self._log(f"Could not create scenario: {e}")
            self._log(traceback.format_exc())
            QMessageBox.critical(self, title, f"Could not create scenario:\n{e}")
            return
        QApplication.restoreOverrideCursor()

        self._log(f"Scenario '{name}': kept {len(keep_fids)} of {n_total} points "
                  f"({pct} %, seed {seed}) from '{base.name()}'.")
        self._log(f"  saved: {gpkg}")
        if save_held:
            self._log(f"  held-out points ({len(held_fids)}) saved: {held_gpkg}")
        self._log(f"  description: {meta_path}")

        if self.scen_use_chk.isChecked():
            self._use_scenario_layer(lyr)
            self._log(f"  '{name}' is now the input points layer for modelling.")
        self._scen_dir_user_set = True     # keep saving into this folder
        self._refresh_scenario_list(select_name=name)

    @staticmethod
    def _append_scenario_index(folder, meta):
        """Keep a one-line-per-scenario overview CSV in the scenario folder."""
        path = os.path.join(folder, "samkit_scenarios_index.csv")
        header = ["created", "name", "source_layer", "field", "percent", "seed",
                  "n_source_points", "n_kept", "n_heldout", "scenario_file", "heldout_file"]
        write_header = not os.path.exists(path)
        with open(path, "a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(header)
            w.writerow([meta["created"], meta["name"], meta["source_layer_name"], meta["field"],
                        meta["percent"], meta["seed"], meta["n_source_points"], meta["n_kept"],
                        meta["n_heldout"], meta["scenario_file"], meta["heldout_file"]])

    def _refresh_scenario_list(self, select_name=None):
        if not hasattr(self, "scen_saved_combo"):
            return
        current = select_name or self.scen_saved_combo.currentData()
        if isinstance(current, dict):
            current = current.get("name")
        self.scen_saved_combo.clear()
        folder = self.scen_dir_edit.text().strip()
        if not folder or not os.path.isdir(folder):
            self.scen_saved_combo.addItem("(no saved scenarios in this folder)", None)
            return
        metas = []
        for fn in os.listdir(folder):
            if not fn.lower().endswith(".json"):
                continue
            try:
                with open(os.path.join(folder, fn), encoding="utf-8") as f:
                    m = json.load(f)
            except Exception:
                continue
            if isinstance(m, dict) and m.get("samkit_scenario") and \
                    os.path.isfile(os.path.join(folder, m.get("scenario_file", ""))):
                m["_folder"] = folder
                metas.append(m)
        if not metas:
            self.scen_saved_combo.addItem("(no saved scenarios in this folder)", None)
            return
        metas.sort(key=lambda m: m.get("created", ""), reverse=True)
        for m in metas:
            label = (f"{m.get('name')}  —  {m.get('percent')} % "
                     f"({m.get('n_kept')}/{m.get('n_source_points')} pts, seed {m.get('seed')})")
            self.scen_saved_combo.addItem(label, m)
        if current:
            for i in range(self.scen_saved_combo.count()):
                d = self.scen_saved_combo.itemData(i)
                if isinstance(d, dict) and d.get("name") == current:
                    self.scen_saved_combo.setCurrentIndex(i)
                    break

    def _on_load_saved_scenario(self):
        meta = self.scen_saved_combo.currentData()
        if not isinstance(meta, dict):
            QMessageBox.information(self, "Point-data scenario",
                                    "There is no saved scenario selected.")
            return
        folder = meta.get("_folder") or self.scen_dir_edit.text().strip()
        gpkg = os.path.join(folder, meta.get("scenario_file", ""))
        name = meta.get("name", os.path.splitext(os.path.basename(gpkg))[0])

        # Reuse the layer if this scenario is already loaded.
        target = os.path.normcase(os.path.abspath(gpkg))
        lyr = None
        for l in QgsProject.instance().mapLayers().values():
            p = self._layer_file_path(l) if isinstance(l, QgsVectorLayer) else ""
            if p and os.path.normcase(os.path.abspath(p)) == target:
                lyr = l
                break
        try:
            if lyr is None:
                lyr = self._add_scenario_layer_to_qgis(gpkg, name, meta)
                held = meta.get("heldout_file")
                if held and os.path.isfile(os.path.join(folder, held)):
                    self._add_scenario_layer_to_qgis(os.path.join(folder, held),
                                                     name + "_heldout",
                                                     dict(meta, role="heldout"))
                self._log(f"Loaded scenario '{name}' from {gpkg}")
            else:
                self._log(f"Scenario '{name}' is already loaded in QGIS.")
        except Exception as e:
            QMessageBox.critical(self, "Point-data scenario", f"Could not load scenario:\n{e}")
            return

        if self.scen_use_chk.isChecked():
            self._use_scenario_layer(lyr)
            self._log(f"  '{name}' is now the input points layer for modelling.")

    def _refresh_raster_list(self):
        self.raster_list.clear()
        for lyr in QgsProject.instance().mapLayers().values():
            if isinstance(lyr, QgsRasterLayer):
                item = QListWidgetItem(lyr.name())
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Unchecked)
                item.setData(Qt.UserRole, lyr.id())
                self.raster_list.addItem(item)

    def _browse_output_path(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save output raster", filter="GeoTIFF (*.tif)")
        if path:
            if not path.lower().endswith(".tif"):
                path += ".tif"
            self.out_path_edit.setText(path)

    def _log(self, msg):
        self.log.appendPlainText(str(msg))
        self.log.repaint()

    def _refresh_dependency_status(self):
        """Update the dependency panel.

        deps.check_all() reads installed metadata from disk only -- it never
        imports the packages. Importing a freshly installed compiled package
        into the running QGIS session can segfault, so detection and loading
        are deliberately kept separate.
        """
        status_map = deps.check_all()
        needs_restart = False
        for display_name, (chk, status_label) in self.dep_checks.items():
            installed, version, _ = status_map[display_name]
            module_name = deps.PACKAGES[display_name][1]
            if installed:
                usable = deps.import_is_live(module_name) or self._can_use_now(module_name)
                if usable:
                    status_label.setText(f"installed (v{version})")
                    status_label.setStyleSheet("color: green;")
                else:
                    status_label.setText(f"installed (v{version}) -- restart QGIS to use")
                    status_label.setStyleSheet("color: #b36b00;")
                    needs_restart = True
                chk.setChecked(False)
            else:
                status_label.setText("not installed")
                status_label.setStyleSheet("color: red;")
                chk.setChecked(True)

        if needs_restart or self._restart_required:
            self.dep_restart_label.setText(
                "Please restart QGIS to finish activating the newly installed "
                "package(s). Nothing else needs to be done."
            )
            self.dep_restart_label.setVisible(True)
        else:
            self.dep_restart_label.setVisible(False)

    def _can_use_now(self, module_name):
        """Whether a package can be used in this session without a restart.

        A package found on disk but not yet imported is only safe to use if
        nothing has been installed during this session; otherwise the on-disk
        build may have been compiled against a different numpy than the one
        already loaded into this process.
        """
        return not self._restart_required

    def _set_dependency_controls_busy(self, busy):
        self.install_dep_btn.setEnabled(not busy)
        self.refresh_dep_btn.setEnabled(not busy)
        self.cancel_dep_btn.setEnabled(busy)
        for chk, _ in self.dep_checks.values():
            chk.setEnabled(not busy)

    def _install_checked_dependencies(self):
        if self._install_worker is not None and self._install_worker.isRunning():
            QMessageBox.information(self, "Install dependencies",
                                     "An install is already running.")
            return

        to_install = []
        for display_name, (chk, _) in self.dep_checks.items():
            if chk.isChecked():
                pip_name = deps.PACKAGES[display_name][0]
                to_install.append(pip_name)
        if not to_install:
            QMessageBox.information(self, "Install dependencies",
                                     "No packages checked. Tick the packages you want "
                                     "to install first.")
            return

        self._log(f"Installing: {', '.join(to_install)} …")
        self._log("This runs in the background. QGIS stays usable; you can cancel it.")
        self._set_dependency_controls_busy(True)

        worker = _DependencyInstallWorker(to_install, parent=self)
        worker.message.connect(self._log)
        worker.finished_ok.connect(self._on_install_finished)
        self._install_worker = worker
        worker.start()

    def _cancel_dependency_install(self):
        if self._install_worker is not None and self._install_worker.isRunning():
            self._log("Cancelling install …")
            self._install_worker.cancel()
            self.cancel_dep_btn.setEnabled(False)

    def _on_install_finished(self, ok):
        self._set_dependency_controls_busy(False)
        worker = self._install_worker
        self._install_worker = None
        if worker is not None:
            worker.deleteLater()

        if ok:
            self._restart_required = True

        self._refresh_dependency_status()

        if ok:
            QMessageBox.information(
                self, "Install dependencies",
                "Install finished successfully.\n\n"
                "Please restart QGIS now to start using the new package(s).\n\n"
                "Python cannot safely load newly installed compiled packages into "
                "a session that is already running, so SAMKit will keep reporting "
                "them as unavailable until you restart. Your QGIS installation "
                "itself was not modified."
            )
        else:
            QMessageBox.warning(
                self, "Install dependencies",
                "Install did not complete successfully. See the log below for "
                "details, including the exact pip command you can run manually."
            )

    # ------------------------------------------------------------------
    # Data diagnostics
    # ------------------------------------------------------------------
    def _log_data_diagnostics(self, y, X, feature_names, field_name):
        """
        Log summary stats + per-covariate correlation with the target so poor
        model performance can be traced back to a data problem (misalignment,
        a near-constant covariate, a near-constant target, or genuinely weak
        covariates) rather than treated as a mysterious black box.
        """
        self._log("")
        self._log(f"--- Data diagnostics (n={len(y)} samples used for fitting) ---")
        self._log(f"Target '{field_name}': min={np.min(y):.4g}  max={np.max(y):.4g}  "
                  f"mean={np.mean(y):.4g}  std={np.std(y):.4g}")
        if np.std(y) < 1e-8:
            self._log("  WARNING: the target has (near) zero variance across your samples. "
                      "Double-check that you selected the correct attribute field -- an ID "
                      "field or a constant column would look like this.")

        max_abs_corr = 0.0
        for j, name in enumerate(feature_names):
            col = X[:, j]
            std = np.std(col)
            if std < 1e-8:
                self._log(f"  '{name}': CONSTANT at your sample locations (std=0). This "
                          f"covariate cannot help predict anything and usually means the "
                          f"points fall outside this raster's real coverage, or the raster "
                          f"and points are misaligned (check CRS/extent).")
                continue
            r = float(np.corrcoef(col, y)[0, 1]) if np.std(y) > 1e-8 else float("nan")
            max_abs_corr = max(max_abs_corr, abs(r) if not np.isnan(r) else 0.0)
            self._log(f"  '{name}': min={np.min(col):.4g}  max={np.max(col):.4g}  "
                      f"mean={np.mean(col):.4g}  std={std:.4g}  corr(target)={r:.3f}")

        if max_abs_corr < 0.15:
            self._log("  WARNING: none of your covariates show even a weak linear correlation "
                      "with the target (all |r| < 0.15). A poor R2 here is more likely a data "
                      "issue than a modeling issue. Common causes:")
            self._log("    - the point layer's CRS doesn't actually match where you think it "
                      "does relative to the rasters (reprojection happens automatically, but "
                      "an incorrect *source* CRS on the point layer will silently place points "
                      "in the wrong location)")
            self._log("    - the covariate rasters don't actually relate to this particular "
                      "soil/environmental property")
            self._log("    - the wrong attribute field was selected")
            self._log("    - the input DEM (if these covariates were derived from one) is noisy "
                      "or contains artifacts -- see the note in the Environmental Covariates tab")
            self._log("  Use the 'Compute correlation table & heatmap' button above, and open "
                      "the training sample CSV written next to your output raster, to inspect "
                      "the extracted values directly.")
        self._log("--- end diagnostics ---")
        self._log("")

    def _write_training_csv(self, path, xs, ys, y, X, feature_names, field_name):
        header = ["x", "y", field_name] + list(feature_names)
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for i in range(len(y)):
                row = [f"{xs[i]:.6f}", f"{ys[i]:.6f}", f"{y[i]:.6g}"] + [f"{v:.6g}" for v in X[i]]
                writer.writerow(row)

    def _checked_raster_layers(self):
        layers = []
        for i in range(self.raster_list.count()):
            item = self.raster_list.item(i)
            if item.checkState() == Qt.Checked:
                lyr_id = item.data(Qt.UserRole)
                lyr = QgsProject.instance().mapLayer(lyr_id)
                if lyr is not None:
                    layers.append(lyr)
        return layers

    # ------------------------------------------------------------------
    # Shared data extraction (used by both correlation analysis and run())
    # ------------------------------------------------------------------
    def _extract_points_and_sample(self, points_layer, field_name, raster_paths, ref_ri, log,
                                    min_points=5):
        """
        Loads point coordinates + target attribute from points_layer (reprojecting to
        the raster CRS if needed), samples the given (already-aligned) raster_paths at
        those locations, and drops nodata/out-of-bounds rows. Returns (X, y, xs, ys).
        """
        log(f"Reading points from '{points_layer.name()}', field '{field_name}'…")
        src_crs = points_layer.crs()
        dst_crs = QgsCoordinateReferenceSystem(ref_ri.projection_wkt)
        transform = None
        if src_crs.isValid() and dst_crs.isValid() and src_crs != dst_crs:
            transform = QgsCoordinateTransform(src_crs, dst_crs, QgsProject.instance())
            log(f"  reprojecting points from {src_crs.authid()} to raster CRS")

        xs, ys, zs = [], [], []
        field_idx = points_layer.fields().indexFromName(field_name)
        for feat in points_layer.getFeatures():
            geom = feat.geometry()
            if geom is None or geom.isEmpty():
                continue
            val = feat.attribute(field_idx)
            if val is None:
                continue
            pt = geom.asPoint() if geom.type() == QgsWkbTypes.PointGeometry else geom.centroid().asPoint()
            if transform is not None:
                pt = transform.transform(pt)
            try:
                zval = float(val)
            except (TypeError, ValueError):
                continue
            xs.append(pt.x())
            ys.append(pt.y())
            zs.append(zval)

        xs = np.array(xs)
        ys = np.array(ys)
        zs = np.array(zs)
        n_pts = len(zs)
        log(f"  {n_pts} valid sample points collected.")
        if n_pts < min_points:
            raise RuntimeError(f"Need at least {min_points} valid sample points with "
                                f"non-null attribute values (found {n_pts}).")

        log("Sampling covariate rasters at sample point locations…")
        X_full = utils.sample_stack_at_points(raster_paths, xs, ys)
        valid_rows = ~np.any(np.isnan(X_full), axis=1)
        n_dropped = n_pts - int(valid_rows.sum())
        if n_dropped > 0:
            log(f"  dropping {n_dropped} point(s) with nodata/out-of-bounds covariates")
            if n_pts > 0 and n_dropped / n_pts > 0.3:
                log("  WARNING: a large fraction of points were dropped. This often means "
                    "your points and covariate rasters don't actually overlap well -- "
                    "double-check the CRS of the point layer and rasters, and that the "
                    "rasters truly cover the sample area.")
        X = X_full[valid_rows]
        y = zs[valid_rows]
        xs_v, ys_v = xs[valid_rows], ys[valid_rows]

        reason = ""
        if n_pts > 0 and n_dropped / n_pts > 0.3:
            reason = self._diagnose_point_overlap(points_layer, src_crs, dst_crs,
                                                  transform is not None, xs, ys,
                                                  raster_paths, X_full, log)
        if len(y) < min_points:
            raise RuntimeError(f"Fewer than {min_points} usable points remain after removing "
                                f"nodata. Check that covariate rasters cover the sample point "
                                f"locations." + (f"\n\nLikely cause: {reason}" if reason else ""))
        return X, y, xs_v, ys_v

    @staticmethod
    def _diagnose_point_overlap(points_layer, src_crs, dst_crs, reprojected, xs, ys,
                                raster_paths, X_full, log):
        """
        Explain why most points found no covariate values: compare the point and
        raster extents/CRSs, count misses per covariate, and recognise the common
        'coordinates and declared CRS disagree' mistake. Returns a one-line
        reason for the error message ('' if nothing specific was found).
        """
        try:
            from osgeo import gdal
        except Exception:
            return ""
        def ext_of(path):
            ds = gdal.Open(path)
            gt = ds.GetGeoTransform()
            x0, y0 = gt[0], gt[3]
            x1, y1 = x0 + gt[1] * ds.RasterXSize, y0 + gt[5] * ds.RasterYSize
            return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)
        def fmt(e):
            return f"x {e[0]:.6g} .. {e[2]:.6g},  y {e[1]:.6g} .. {e[3]:.6g}"

        reason = ""
        log("")
        log("--- Why the points miss the rasters (diagnosis) ---")
        log(f"  Point layer '{points_layer.name()}': CRS {src_crs.authid() or src_crs.description() or 'NOT SET'}"
            + ("  (reprojected to the raster CRS)" if reprojected else "  (NOT reprojected: "
               "treated as already in the raster CRS)"))
        pe = (float(np.min(xs)), float(np.min(ys)), float(np.max(xs)), float(np.max(ys)))
        log(f"  Points, in raster CRS units: {fmt(pe)}")
        log(f"  Raster CRS: {dst_crs.authid() or dst_crs.description() or 'NOT SET / unreadable'}")
        any_overlap = False
        for j, path in enumerate(raster_paths):
            try:
                re_ = ext_of(path)
            except Exception:
                continue
            miss = int(np.isnan(X_full[:, j]).sum())
            inside = int(((xs >= re_[0]) & (xs <= re_[2]) & (ys >= re_[1]) & (ys <= re_[3])).sum())
            any_overlap |= inside > 0
            log(f"  {os.path.basename(path)}: extent {fmt(re_)}; {inside} point(s) inside, "
                f"{miss} without a value")
        # The usual culprit: lon/lat numbers in a layer that claims a projected CRS, or
        # metre coordinates in a layer that claims a geographic CRS.
        looks_degrees = (np.all(np.abs(xs) <= 180) and np.all(np.abs(ys) <= 90))
        if not reprojected:
            raw_x = np.array([f.geometry().centroid().asPoint().x()
                              for f in points_layer.getFeatures() if f.hasGeometry()][:500])
        else:
            raw_x = None
        if src_crs.isValid() and not src_crs.isGeographic() and raw_x is not None and \
                len(raw_x) and np.all(np.abs(raw_x) <= 180) and looks_degrees:
            reason = (f"the point coordinates look like longitude/latitude degrees, but the "
                      f"point layer's CRS is set to {src_crs.authid()} (a projected CRS in "
                      f"metres). Set the layer's CRS to EPSG:4326 (Layer Properties > Source "
                      f"> Assigned CRS) and run again.")
        elif src_crs.isValid() and src_crs.isGeographic() and not looks_degrees and not reprojected:
            reason = (f"the point coordinates look like projected metres, but the point "
                      f"layer's CRS is set to {src_crs.authid()} (degrees). Set the layer's "
                      f"CRS to the projected CRS the coordinates were collected in.")
        elif not dst_crs.isValid():
            reason = ("the covariate rasters have no readable CRS, so the points could not be "
                      "reprojected onto them. Assign the rasters' CRS and run again.")
        elif not src_crs.isValid():
            reason = ("the point layer has no CRS. Assign the CRS the coordinates were "
                      "collected in (Layer Properties > Source) and run again.")
        elif not any_overlap:
            reason = ("the points and the rasters do not overlap at all: they cover different "
                      "places (compare the extents above), or one of the layers has the wrong "
                      "CRS assigned.")
        elif dst_crs.isValid() and dst_crs.isGeographic():
            empty = [os.path.basename(p) for j, p in enumerate(raster_paths)
                     if np.isnan(X_full[:, j]).all()]
            reason = ("the covariate rasters are in degrees (" + (dst_crs.authid() or "geographic")
                      + "), and " + (", ".join(empty) if empty else "some covariates")
                      + " have no values at the points. Terrain covariates computed from a DEM "
                      "in degrees are invalid (SAGA's MRVBF/MRRTF come out as all nodata; slope "
                      "near 90 degrees). Project the DEM to metres in tab 1, recompute the "
                      "covariates in tab 2, and run again.")
        else:
            reason = ("the points fall inside the raster extents but on nodata cells -- check "
                      "the covariates listed above with many points 'without a value'.")
        log(f"  => {reason}")
        log("--- end diagnosis ---")
        return reason

    def _extract_training_data(self, points_layer, field_name, raster_layers, align, log):
        """
        Aligns (optionally) the checked covariate rasters, extracts point
        coordinates + target attribute, samples covariates at those points,
        and drops nodata/out-of-bounds rows. Returns a dict with X, y, xs, ys,
        feature_names, raster_paths (possibly aligned copies), ref_ri.
        Raises RuntimeError with a user-facing message on failure.
        """
        tmpdir = tempfile.mkdtemp(prefix="samkit_")
        raster_paths = [lyr.source().split("|")[0] for lyr in raster_layers]
        feature_names = [lyr.name() for lyr in raster_layers]
        ref_path = raster_paths[0]

        if align and len(raster_paths) > 1:
            log("Aligning covariate rasters to common grid (reference = "
                f"'{feature_names[0]}')…")
            aligned_paths = [ref_path]
            for p, name in zip(raster_paths[1:], feature_names[1:]):
                out_p = os.path.join(tmpdir, f"aligned_{os.path.basename(p)}")
                utils.align_raster_to_reference(p, ref_path, out_p)
                aligned_paths.append(out_p)
                log(f"  aligned '{name}' -> {out_p}")
            raster_paths = aligned_paths

        ref_ri = utils.read_raster(ref_path)
        X, y, xs_v, ys_v = self._extract_points_and_sample(points_layer, field_name,
                                                            raster_paths, ref_ri, log, min_points=5)

        return {
            "X": X, "y": y, "xs": xs_v, "ys": ys_v,
            "feature_names": feature_names, "raster_paths": raster_paths,
            "ref_ri": ref_ri, "tmpdir": tmpdir,
        }

    # ------------------------------------------------------------------
    # Covariate correlation analysis
    # ------------------------------------------------------------------
    def _on_compute_correlation(self):
        points_layer = self.points_combo.currentLayer()
        field_name = self.field_combo.currentText()
        raster_layers = self._checked_raster_layers()

        if points_layer is None:
            QMessageBox.warning(self, "Correlation analysis", "Please select a point vector layer.")
            return
        if not field_name:
            QMessageBox.warning(self, "Correlation analysis", "Please select a numeric attribute field.")
            return
        if not raster_layers:
            QMessageBox.warning(self, "Correlation analysis", "Please check at least one covariate raster.")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save correlation table as", "correlation_analysis.csv", filter="CSV (*.csv)")
        if not save_path:
            return
        if not save_path.lower().endswith(".csv"):
            save_path += ".csv"

        self.corr_btn.setEnabled(False)
        self.log.clear()
        try:
            data = self._extract_training_data(points_layer, field_name, raster_layers,
                                               self.align_chk.isChecked(), self._log)
            X, y = data["X"], data["y"]
            feature_names = data["feature_names"]

            labels = [field_name] + list(feature_names)
            combined = np.column_stack([y] + [X[:, j] for j in range(X.shape[1])])
            with np.errstate(invalid="ignore"):
                corr = np.corrcoef(combined, rowvar=False)

            self._write_correlation_csv(save_path, labels, corr)
            self._log(f"Wrote correlation table to: {save_path}")

            self._log("")
            self._log("--- Correlation with target (sorted by |r|) ---")
            target_corrs = [(labels[j], corr[0, j]) for j in range(1, len(labels))]
            target_corrs.sort(key=lambda t: abs(t[1]) if not np.isnan(t[1]) else -1, reverse=True)
            for name, r in target_corrs:
                flag = "  <-- weak" if (not np.isnan(r) and abs(r) < 0.15) else ""
                self._log(f"  {field_name} vs {name}: r = {r:.3f}{flag}")

            self._log("")
            self._log("--- Covariate-covariate correlations >= 0.8 (possible redundancy) ---")
            n_labels = len(labels)
            any_high = False
            for i in range(1, n_labels):
                for j in range(i + 1, n_labels):
                    r = corr[i, j]
                    if not np.isnan(r) and abs(r) >= 0.8:
                        any_high = True
                        self._log(f"  {labels[i]} vs {labels[j]}: r = {r:.3f}")
            if not any_high:
                self._log("  (none found)")

            png_path = os.path.splitext(save_path)[0] + "_heatmap.png"
            try:
                self._render_correlation_heatmap_png(png_path, labels, corr)
                self._log(f"Wrote correlation heatmap image to: {png_path}")
                self._show_image_preview(png_path, "Covariate correlation heatmap")
            except Exception as e:
                self._log(f"(Could not render heatmap image: {e}. The CSV table above is still valid.)")

            QMessageBox.information(self, "Correlation analysis",
                                     f"Correlation table written to:\n{save_path}\n\n"
                                     f"Use this to decide which covariates to keep before running "
                                     f"a model -- drop covariates with very weak target correlation, "
                                     f"and consider dropping one of any pair of covariates that are "
                                     f"highly correlated with each other (redundant information).")
        except dsm_models.MissingDependencyError as e:
            self._log("MISSING DEPENDENCY: " + str(e))
            QMessageBox.critical(self, "Missing dependency", str(e))
        except Exception as e:
            self._log("ERROR: " + str(e))
            self._log(traceback.format_exc())
            QMessageBox.critical(self, "Correlation analysis", f"Failed:\n{e}")
        finally:
            self.corr_btn.setEnabled(True)

    def _write_correlation_csv(self, path, labels, corr):
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([""] + labels)
            for i, lab in enumerate(labels):
                row = [lab] + [f"{v:.4f}" if not np.isnan(v) else "" for v in corr[i]]
                writer.writerow(row)

    @staticmethod
    def _corr_color(v):
        v = max(-1.0, min(1.0, float(v)))
        white = (247, 247, 247)
        if v < 0:
            t = -v
            other = (33, 102, 172)   # blue
        else:
            t = v
            other = (178, 24, 43)    # red
        r = int(white[0] + (other[0] - white[0]) * t)
        g = int(white[1] + (other[1] - white[1]) * t)
        b = int(white[2] + (other[2] - white[2]) * t)
        return QColor(r, g, b)

    def _render_correlation_heatmap_png(self, path, labels, corr):
        n = len(labels)
        cell = 64
        margin_left = 150
        margin_top = 150
        margin_right = 20
        margin_bottom = 20
        width = margin_left + cell * n + margin_right
        height = margin_top + cell * n + margin_bottom

        pixmap = QPixmap(width, height)
        pixmap.fill(QColor("white"))
        painter = QPainter(pixmap)
        try:
            cell_font = QFont("Sans Serif", 8)
            label_font = QFont("Sans Serif", 9)

            painter.setFont(cell_font)
            for i in range(n):
                for j in range(n):
                    v = corr[i, j]
                    rect = QRectF(margin_left + j * cell, margin_top + i * cell, cell, cell)
                    if np.isnan(v):
                        painter.fillRect(rect, QColor(220, 220, 220))
                        continue
                    painter.fillRect(rect, self._corr_color(v))
                    painter.setPen(QColor("black") if abs(v) < 0.6 else QColor("white"))
                    painter.drawText(rect, Qt.AlignCenter, f"{v:.2f}")

            painter.setFont(label_font)
            painter.setPen(QColor("black"))
            for i, lab in enumerate(labels):
                row_rect = QRectF(0, margin_top + i * cell, margin_left - 8, cell)
                painter.drawText(row_rect, Qt.AlignRight | Qt.AlignVCenter, lab)

                painter.save()
                painter.translate(margin_left + i * cell + cell / 2.0, margin_top - 8)
                painter.rotate(-45)
                painter.drawText(0, 0, lab)
                painter.restore()
        finally:
            painter.end()

        ok = pixmap.save(path, "PNG")
        if not ok:
            raise RuntimeError("QPixmap.save() returned False")

    def _show_image_preview(self, path, title):
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        vlayout = QVBoxLayout(dlg)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        label = QLabel()
        pix = QPixmap(path)
        label.setPixmap(pix)
        scroll.setWidget(label)
        vlayout.addWidget(scroll)

        btn_row = QHBoxLayout()
        save_btn = QPushButton("Save PNG As…")

        def _save_as():
            dest, _ = QFileDialog.getSaveFileName(
                dlg, "Save heatmap as PNG", os.path.basename(path), filter="PNG image (*.png)")
            if not dest:
                return
            if not dest.lower().endswith(".png"):
                dest += ".png"
            if pix.save(dest, "PNG"):
                QMessageBox.information(dlg, "Save heatmap", f"Saved to:\n{dest}")
            else:
                QMessageBox.warning(dlg, "Save heatmap", "Could not save the image to that location.")

        save_btn.clicked.connect(_save_as)
        btn_row.addWidget(save_btn)
        btn_row.addStretch()
        vlayout.addLayout(btn_row)

        dlg.resize(min(950, pix.width() + 40), min(780, pix.height() + 90))
        dlg.show()
        self._preview_dialog = dlg  # keep a reference so it isn't garbage-collected

    # ------------------------------------------------------------------
    # Spatial block split helper
    # ------------------------------------------------------------------
    def _get_spatial_calibration_mask(self, xs, ys, ref_ri, polygon_layer, calibration_inside, log):
        poly_crs = polygon_layer.crs()
        ref_crs = QgsCoordinateReferenceSystem(ref_ri.projection_wkt)
        transform = None
        if poly_crs.isValid() and ref_crs.isValid() and poly_crs != ref_crs:
            transform = QgsCoordinateTransform(poly_crs, ref_crs, QgsProject.instance())
            log(f"  reprojecting calibration polygon from {poly_crs.authid()} to raster CRS")

        geoms = []
        for feat in polygon_layer.getFeatures():
            g = feat.geometry()
            if g is None or g.isEmpty():
                continue
            if transform is not None:
                g = QgsGeometry(g)
                g.transform(transform)
            geoms.append(g)

        if not geoms:
            raise RuntimeError("The calibration polygon layer has no usable features. "
                                "Draw or digitize at least one polygon first.")

        n = len(xs)
        mask = np.zeros(n, dtype=bool)
        for i in range(n):
            pt_geom = QgsGeometry.fromPointXY(QgsPointXY(float(xs[i]), float(ys[i])))
            inside = any(g.contains(pt_geom) for g in geoms)
            mask[i] = inside

        if not calibration_inside:
            mask = ~mask

        n_calib = int(mask.sum())
        n_valid = n - n_calib
        log(f"  spatial split: {n_calib} calibration point(s), {n_valid} validation point(s)")
        if n_calib < 5:
            raise RuntimeError(f"Only {n_calib} points fall inside the calibration polygon -- "
                                f"need at least 5. Draw a larger polygon.")
        if n_valid < 3:
            log("  WARNING: very few validation points remain outside the calibration polygon; "
                "the reported accuracy metrics will be unstable.")
        return mask

    # ------------------------------------------------------------------
    # Fold / split file saving and metrics logging
    # ------------------------------------------------------------------
    def _save_fold_files(self, folder, xs, ys, y, X, feature_names, field_name, folds, log):
        os.makedirs(folder, exist_ok=True)
        per_fold_dir = os.path.join(folder, "per_fold")
        os.makedirs(per_fold_dir, exist_ok=True)
        n = len(y)
        fold_assignment = np.full(n, -1, dtype=int)
        for i, idx in enumerate(folds):
            fold_assignment[idx] = i

        master_path = os.path.join(folder, "fold_assignments.csv")
        if dsm_models.is_partition(folds, n):
            with open(master_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["point_index", "x", "y", field_name, "fold"])
                for i in range(n):
                    writer.writerow([i, f"{xs[i]:.6f}", f"{ys[i]:.6f}", f"{y[i]:.6g}",
                                      int(fold_assignment[i])])
        else:
            # Repeated random splits: a point can be validated in several folds.
            val_folds = [[] for _ in range(n)]
            for i, idx in enumerate(folds):
                for pi in idx:
                    val_folds[int(pi)].append(i + 1)
            with open(master_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["point_index", "x", "y", field_name,
                                 "n_times_validation", "validation_in_folds"])
                for i in range(n):
                    writer.writerow([i, f"{xs[i]:.6f}", f"{ys[i]:.6f}", f"{y[i]:.6g}",
                                      len(val_folds[i]), ";".join(map(str, val_folds[i]))])

        # One consolidated file with every fold's calibration (training) AND
        # validation rows stacked together, tagged by fold number and role --
        # the single-file view of "all 10 folds' training/validation samples".
        combined_path = os.path.join(folder, "all_folds_calibration_and_validation.csv")
        all_idx = np.arange(n)
        with open(combined_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["fold", "role", "point_index", "x", "y", field_name] + list(feature_names))
            for i, val_idx in enumerate(folds):
                cal_idx = np.setdiff1d(all_idx, val_idx)
                for role, idx_arr in (("calibration", cal_idx), ("validation", val_idx)):
                    for pi in idx_arr:
                        row = [i + 1, role, int(pi), f"{xs[pi]:.6f}", f"{ys[pi]:.6f}", f"{y[pi]:.6g}"] + \
                              [f"{v:.6g}" for v in X[pi]]
                        writer.writerow(row)
        log(f"Saved combined all-folds training/validation table to: {combined_path}")

        # Per-fold individual files too, for anyone who prefers one file per fold.
        for i, val_idx in enumerate(folds):
            cal_idx = np.setdiff1d(all_idx, val_idx)
            self._write_training_csv(os.path.join(per_fold_dir, f"fold_{i + 1}_calibration.csv"),
                                     xs[cal_idx], ys[cal_idx], y[cal_idx], X[cal_idx],
                                     feature_names, field_name)
            self._write_training_csv(os.path.join(per_fold_dir, f"fold_{i + 1}_validation.csv"),
                                     xs[val_idx], ys[val_idx], y[val_idx], X[val_idx],
                                     feature_names, field_name)
        log(f"Saved {len(folds)}-fold calibration/validation split files to: {folder}")
        return folder

    def _write_metrics_log(self, out_path, meta, metrics, log, accuracy_dir=None):
        log_path = os.path.join(os.path.dirname(out_path) or ".", "samkit_model_performance_log.csv")
        header = ["timestamp", "output_raster", "model", "target_field", "validation_method",
                  "n_folds_or_split", "n_samples_used", "n_covariates", "R2", "RMSE", "MAE",
                  "bias", "CCC", "R2_regression", "hyperparameters"]
        row = [
            datetime.datetime.now().isoformat(timespec="seconds"),
            os.path.basename(out_path),
            meta.get("model", ""),
            meta.get("field", ""),
            meta.get("validation_method", ""),
            str(meta.get("n_folds_or_split", "")),
            str(meta.get("n_samples", "")),
            str(meta.get("n_covariates", "")),
            f"{metrics['r2']:.4f}" if metrics else "NA",
            f"{metrics['rmse']:.4f}" if metrics else "NA",
            f"{metrics['mae']:.4f}" if metrics else "NA",
            f"{metrics['bias']:.4f}" if metrics else "NA",
            f"{metrics['ccc']:.4f}" if metrics else "NA",
            f"{metrics['r2_regression']:.4f}" if metrics else "NA",
            meta.get("hyperparams", ""),
        ]
        write_header = not os.path.exists(log_path)
        with open(log_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerow(row)

        single_path = (os.path.join(accuracy_dir, "accuracy_metrics.csv") if accuracy_dir
                       else os.path.splitext(out_path)[0] + "_metrics.csv")
        with open(single_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(row)

        log(f"Appended this run's accuracy metrics to: {log_path}")
        log(f"Wrote this run's metrics to: {single_path}")
        return log_path, single_path

    # ------------------------------------------------------------------
    # Organized run outputs
    # ------------------------------------------------------------------
    def _prepare_results_dir(self, out_path):
        """
        Every artifact from a run goes into one clearly-named, numbered folder
        next to the output raster, instead of being scattered as loose files
        and sibling folders. Returns a dict of the sub-folder paths.
        """
        root = os.path.splitext(out_path)[0] + "_SAMKit_results"
        paths = {
            "root": root,
            "accuracy": os.path.join(root, "01_accuracy"),
            "obs_pred": os.path.join(root, "02_observed_vs_predicted"),
            "samples": os.path.join(root, "03_input_samples"),
            "split": os.path.join(root, "04_calibration_validation_split"),
        }
        for p in paths.values():
            os.makedirs(p, exist_ok=True)
        # created only when maps are actually written
        paths["maps"] = os.path.join(root, "05_maps")
        return paths

    # ------------------------------------------------------------------
    # Observed vs predicted table
    # ------------------------------------------------------------------
    @staticmethod
    def _obs_pred_columns(field_name, feature_names):
        return (["point_index", "x", "y",
                 f"observed_{field_name}", f"predicted_{field_name}",
                 "residual_observed_minus_predicted", "abs_residual",
                 "role", "fold", "prediction_type"] + list(feature_names))

    @staticmethod
    def _obs_pred_row(i, x, y, obs, pred, role, fold, pred_type, covariates):
        resid = obs - pred if pred is not None and not np.isnan(pred) else float("nan")
        return {
            "point_index": int(i),
            "x": float(x), "y": float(y),
            "observed": float(obs),
            "predicted": float(pred) if pred is not None else float("nan"),
            "residual": float(resid),
            "abs_residual": float(abs(resid)) if not np.isnan(resid) else float("nan"),
            "role": role,
            "fold": fold,
            "prediction_type": pred_type,
            "covariates": [float(v) for v in covariates],
        }

    def _build_obs_pred_from_cv(self, cv_out, xs, ys, y, X, log):
        """
        k-fold path: each point's prediction is the one made while that point was
        held OUT of the training set, so every row is an honest out-of-sample
        observed/predicted pair.
        """
        if not cv_out or "predicted" not in cv_out:
            log("(No held-out predictions were produced -- validation was disabled or "
                "there were too few samples, so no observed-vs-predicted table is written.)")
            return []
        pred = np.asarray(cv_out["predicted"], dtype=float)
        assignment = np.asarray(cv_out.get("fold_assignment", np.full(len(y), -1)))
        if cv_out.get("repeated_split"):
            # One row per point per fold in which it was held out.
            rows = []
            for j, i in enumerate(np.asarray(cv_out["point_index"], dtype=int)):
                rows.append(self._obs_pred_row(
                    i, xs[i], ys[i], y[i], pred[j],
                    "validation (held out in this fold)", int(assignment[j]) + 1,
                    "repeated random-split held-out prediction", X[i]))
            return rows
        rows = []
        for i in range(len(y)):
            fold = int(assignment[i]) + 1 if int(assignment[i]) >= 0 else ""
            rows.append(self._obs_pred_row(
                i, xs[i], ys[i], y[i], pred[i],
                "validation (held out in its own fold)", fold,
                "out-of-fold cross-validation prediction", X[i]))
        return rows

    def _build_obs_pred_from_split(self, xs_fit, ys_fit, y_fit, X_fit, calib_pred,
                                    xs_val, ys_val, y_val, X_val, val_pred, model_name):
        """
        Spatial-block / external-validation path: validation rows are true
        held-out predictions; calibration rows are in-sample fitted values,
        labelled as such so the two are never confused. For kriging models the
        in-sample fitted value at a data point is near-exact by construction
        (kriging is an exact interpolator), so those calibration residuals are
        expected to be ~0 and should not be read as model accuracy.
        """
        is_kriging = "Kriging" in model_name
        calib_type = ("in-sample fitted value (kriging is an exact interpolator -- "
                      "not a measure of accuracy)" if is_kriging
                      else "in-sample fitted value (not a measure of accuracy)")
        rows = []
        idx = 0
        for i in range(len(y_val)):
            pv = val_pred[i] if val_pred is not None and len(val_pred) > i else float("nan")
            rows.append(self._obs_pred_row(idx, xs_val[i], ys_val[i], y_val[i], pv,
                                            "validation (held out)", "",
                                            "held-out prediction", X_val[i]))
            idx += 1
        for i in range(len(y_fit)):
            cv = calib_pred[i] if calib_pred is not None and len(calib_pred) > i else float("nan")
            rows.append(self._obs_pred_row(idx, xs_fit[i], ys_fit[i], y_fit[i], cv,
                                            "calibration (used to fit)", "",
                                            calib_type, X_fit[i]))
            idx += 1
        return rows

    def _write_obs_pred_csv(self, path, rows, field_name, feature_names):
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self._obs_pred_columns(field_name, feature_names))
            for r in rows:
                writer.writerow([
                    r["point_index"], f"{r['x']:.6f}", f"{r['y']:.6f}",
                    f"{r['observed']:.6g}",
                    "" if np.isnan(r["predicted"]) else f"{r['predicted']:.6g}",
                    "" if np.isnan(r["residual"]) else f"{r['residual']:.6g}",
                    "" if np.isnan(r["abs_residual"]) else f"{r['abs_residual']:.6g}",
                    r["role"], r["fold"], r["prediction_type"],
                ] + [f"{v:.6g}" for v in r["covariates"]])
        return path

    @staticmethod
    def _validation_rows(rows):
        """Only the honest held-out rows -- the ones accuracy should be judged on."""
        return [r for r in rows
                if r["role"].startswith("validation") and not np.isnan(r["predicted"])]

    def _log_obs_pred_preview(self, rows, field_name, max_rows=15):
        val_rows = self._validation_rows(rows)
        if not val_rows:
            return
        self._log("")
        self._log(f"--- Observed vs predicted '{field_name}' (held-out points) ---")
        header = f"{'#':>5}  {'observed':>14}  {'predicted':>14}  {'residual':>14}  {'fold':>5}"
        self._log(header)
        self._log("-" * len(header))
        for r in val_rows[:max_rows]:
            self._log(f"{r['point_index']:>5}  {r['observed']:>14.6g}  "
                      f"{r['predicted']:>14.6g}  {r['residual']:>14.6g}  "
                      f"{str(r['fold']):>5}")
        if len(val_rows) > max_rows:
            self._log(f"... and {len(val_rows) - max_rows} more row(s) -- the full table is "
                      f"in the results folder and in the results window.")
        self._log(f"({len(val_rows)} held-out observed/predicted pairs in total.)")

    @staticmethod
    def _covariate_importances(model_name, fitted, feature_names):
        """
        Ranked impurity-based (mean decrease in impurity) importances of the
        fitted Random Forest -- or of the Random Forest trend inside Regression
        Kriging. Returns [(rank, name, importance), ...] or [] when the model
        has no importances (SVM, OK, linear-trend RK). Read-only: the fitted
        model and its predictions are not touched.
        """
        est = None
        if model_name == "Random Forest":
            est = fitted
        elif model_name == "Regression Kriging" and isinstance(fitted, dict):
            est = getattr(fitted.get("trend_predict_fn"), "__self__", None)
        imp = getattr(est, "feature_importances_", None)
        if imp is None or len(imp) != len(feature_names):
            return []
        order = np.argsort(-np.asarray(imp, dtype=float))
        return [(r + 1, feature_names[j], float(imp[j])) for r, j in enumerate(order)]

    def _write_importances(self, path, ranked, model_name):
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["rank", "covariate", "importance_MDI", "importance_percent", "model"])
            for rank, name, v in ranked:
                writer.writerow([rank, name, f"{v:.6f}", f"{100.0 * v:.2f}", model_name])
        return path

    def _write_per_fold_metrics_csv(self, path, per_fold, field_name):
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["fold", "n_validation_points", "R2", "RMSE", "MAE",
                              "bias", "CCC", "R2_regression", "target_field"])
            for m in per_fold:
                writer.writerow([m["fold"], m["n"], f"{m['r2']:.4f}", f"{m['rmse']:.4f}",
                                  f"{m['mae']:.4f}", f"{m['bias']:.4f}", f"{m['ccc']:.4f}",
                                  "" if np.isnan(m["r2_regression"]) else f"{m['r2_regression']:.4f}",
                                  field_name])
        return path

    def _write_covariate_summary_csv(self, path, y, X, feature_names, field_name):
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["variable", "role", "n", "min", "max", "mean", "std",
                              "pearson_r_with_target"])
            writer.writerow([field_name, "target (property of interest)", len(y),
                              f"{np.min(y):.6g}", f"{np.max(y):.6g}", f"{np.mean(y):.6g}",
                              f"{np.std(y):.6g}", "1.0000"])
            for j, name in enumerate(feature_names):
                col = X[:, j]
                if np.std(col) > 1e-12 and np.std(y) > 1e-12:
                    r = f"{float(np.corrcoef(col, y)[0, 1]):.4f}"
                else:
                    r = ""
                writer.writerow([name, "covariate", len(col), f"{np.min(col):.6g}",
                                  f"{np.max(col):.6g}", f"{np.mean(col):.6g}",
                                  f"{np.std(col):.6g}", r])
        return path

    # ------------------------------------------------------------------
    # Observed vs predicted scatter plot (drawn with Qt -- no matplotlib needed)
    # ------------------------------------------------------------------
    @staticmethod
    def _nice_ticks(lo, hi, n=8):
        span = hi - lo
        if span <= 0:
            return [lo]
        raw = span / max(1, n - 1)
        mag = 10 ** np.floor(np.log10(raw))
        for mult in (1, 2, 2.5, 5, 10):
            step = mult * mag
            if raw <= step:
                break
        start = np.ceil(lo / step) * step
        ticks = []
        v = start
        while v <= hi + step * 1e-9:
            ticks.append(float(v))
            v += step
        return ticks

    def _render_scatter_png(self, path, rows, field_name, model_name, metrics,
                             validation_label):
        val_rows = self._validation_rows(rows)
        if len(val_rows) < 2:
            raise RuntimeError("not enough held-out observed/predicted pairs to plot")
        obs = np.array([r["observed"] for r in val_rows], dtype=float)
        pred = np.array([r["predicted"] for r in val_rows], dtype=float)

        W, H = 780, 720
        left, right, top, bottom = 100, 34, 78, 92
        pw, ph = W - left - right, H - top - bottom

        lo = float(min(obs.min(), pred.min()))
        hi = float(max(obs.max(), pred.max()))
        pad = (hi - lo) * 0.07 if hi > lo else max(abs(hi) * 0.1, 1.0)
        lo, hi = lo - pad, hi + pad

        def sx(v):
            return left + (v - lo) / (hi - lo) * pw

        def sy(v):
            return top + ph - (v - lo) / (hi - lo) * ph

        pixmap = QPixmap(W, H)
        pixmap.fill(QColor("white"))
        painter = QPainter(pixmap)
        try:
            painter.setRenderHint(QPainter.Antialiasing, True)

            painter.setFont(QFont("Sans Serif", 12, QFont.Bold))
            painter.setPen(QColor(30, 30, 30))
            painter.drawText(QRectF(0, 14, W, 22), Qt.AlignCenter,
                              f"Observed vs predicted: {field_name}")
            painter.setFont(QFont("Sans Serif", 9))
            painter.setPen(QColor(90, 90, 90))
            painter.drawText(QRectF(0, 40, W, 18), Qt.AlignCenter,
                              f"{model_name}  |  {validation_label}  |  n = {len(obs)} held-out points")

            painter.fillRect(QRectF(left, top, pw, ph), QColor(252, 252, 252))

            ticks = self._nice_ticks(lo, hi)
            painter.setFont(QFont("Sans Serif", 8))
            for t in ticks:
                x, yv = sx(t), sy(t)
                painter.setPen(QPen(QColor(230, 230, 230), 1))
                painter.drawLine(int(x), int(top), int(x), int(top + ph))
                painter.drawLine(int(left), int(yv), int(left + pw), int(yv))
                painter.setPen(QColor(70, 70, 70))
                painter.drawText(QRectF(x - 45, top + ph + 6, 90, 16),
                                  Qt.AlignCenter, f"{t:.4g}")
                painter.drawText(QRectF(left - 92, yv - 9, 84, 18),
                                  Qt.AlignRight | Qt.AlignVCenter, f"{t:.4g}")

            painter.setPen(QPen(QColor(120, 120, 120), 1))
            painter.drawRect(QRectF(left, top, pw, ph))

            # 1:1 line -- perfect agreement
            painter.setPen(QPen(QColor(150, 150, 150), 2, Qt.DashLine))
            painter.drawLine(int(sx(lo)), int(sy(lo)), int(sx(hi)), int(sy(hi)))

            # least-squares fit of predicted on observed
            if np.std(obs) > 1e-12:
                slope, intercept = np.polyfit(obs, pred, 1)
                painter.setPen(QPen(QColor(178, 24, 43), 2))
                painter.drawLine(int(sx(lo)), int(sy(slope * lo + intercept)),
                                  int(sx(hi)), int(sy(slope * hi + intercept)))
            else:
                slope, intercept = float("nan"), float("nan")

            painter.setPen(QPen(QColor(21, 76, 121), 1))
            painter.setBrush(QColor(31, 119, 180, 130))
            for o, p in zip(obs, pred):
                painter.drawEllipse(QRectF(sx(o) - 4.5, sy(p) - 4.5, 9, 9))
            painter.setBrush(Qt.NoBrush)

            painter.setFont(QFont("Sans Serif", 10))
            painter.setPen(QColor(40, 40, 40))
            painter.drawText(QRectF(0, H - 30, W, 20), Qt.AlignCenter,
                              f"Observed {field_name}")
            painter.save()
            painter.translate(24, top + ph / 2.0)
            painter.rotate(-90)
            painter.drawText(QRectF(-180, -10, 360, 20), Qt.AlignCenter,
                              f"Predicted {field_name}")
            painter.restore()

            if metrics:
                lines = [f"R² = {metrics['r2']:.3f}", f"RMSE = {metrics['rmse']:.4g}",
                          f"MAE = {metrics['mae']:.4g}", f"bias = {metrics['bias']:.4g}",
                          f"CCC = {metrics['ccc']:.3f}"]
                if not np.isnan(slope):
                    lines.append(f"fit: y = {slope:.3f}x + {intercept:.4g}")
                box_w, box_h = 208, 18 * len(lines) + 14
                bx, by = left + 12, top + 12
                painter.setPen(QPen(QColor(180, 180, 180), 1))
                painter.setBrush(QColor(255, 255, 255, 235))
                painter.drawRect(QRectF(bx, by, box_w, box_h))
                painter.setBrush(Qt.NoBrush)
                painter.setFont(QFont("Sans Serif", 9))
                painter.setPen(QColor(40, 40, 40))
                for i, line in enumerate(lines):
                    painter.drawText(QRectF(bx + 10, by + 7 + i * 18, box_w - 20, 18),
                                      Qt.AlignLeft | Qt.AlignVCenter, line)

            painter.setFont(QFont("Sans Serif", 8))
            painter.setPen(QColor(120, 120, 120))
            painter.drawText(QRectF(left, H - 14, pw, 14), Qt.AlignLeft,
                              "dashed grey = 1:1 line (perfect agreement)   |   "
                              "red = least-squares fit")
        finally:
            painter.end()

        if not pixmap.save(path, "PNG"):
            raise RuntimeError("QPixmap.save() returned False")
        return path

    # ------------------------------------------------------------------
    # Run summary
    # ------------------------------------------------------------------
    def _write_run_summary(self, path, meta, metrics, per_fold, rows, paths, out_path,
                            variance_path=None, map_files=None):
        val_rows = self._validation_rows(rows)
        lines = []
        lines.append("SAMKit -- Spatial Predictive Modeling run summary")
        lines.append("=" * 62)
        lines.append(f"Run at              : {datetime.datetime.now().isoformat(timespec='seconds')}")
        lines.append(f"Property of interest : {meta.get('field', '')}")
        lines.append(f"Model                : {meta.get('model', '')}")
        lines.append(f"Hyperparameters      : {meta.get('hyperparams', '')}")
        lines.append(f"Validation strategy  : {meta.get('validation_method', '')} "
                     f"({meta.get('n_folds_or_split', '')})")
        scen = meta.get("scenario")
        if scen:
            lines.append(f"Point-data scenario  : {scen.get('name', '')} -- "
                         f"{scen.get('percent', '')} % of '{scen.get('source_layer_name', '')}' "
                         f"({scen.get('n_kept', '')} of {scen.get('n_source_points', '')} points, "
                         f"random seed {scen.get('seed', '')})")
        else:
            lines.append("Point-data scenario  : none (all points of the input layer)")
        lines.append(f"Sample points used   : {meta.get('n_samples', '')} "
                     f"(final mapped model fitted with {meta.get('n_fit', meta.get('n_samples', ''))})")
        lines.append(f"Covariates           : {meta.get('n_covariates', '')} "
                     f"({', '.join(meta.get('feature_names', []))})")
        lines.append("")
        lines.append("Accuracy on held-out points")
        lines.append("-" * 62)
        if metrics:
            lines.append(f"  n held-out points : {len(val_rows)}")
            lines.append(f"  R2                : {metrics['r2']:.4f}")
            lines.append(f"  RMSE              : {metrics['rmse']:.4f}")
            lines.append(f"  MAE               : {metrics['mae']:.4f}")
            lines.append(f"  bias (pred - obs) : {metrics['bias']:.4f}")
            lines.append(f"  CCC (Lin)         : {metrics['ccc']:.4f}")
            if not np.isnan(metrics.get("r2_regression", float("nan"))):
                lines.append(f"  R2 (regression of predicted on observed) : "
                             f"{metrics['r2_regression']:.4f}")
        else:
            lines.append("  No validation metrics were computed for this run "
                         "(validation disabled, or too few usable samples).")
        if per_fold:
            lines.append("")
            lines.append("Accuracy per fold")
            lines.append("-" * 62)
            lines.append(f"  {'fold':>5}  {'n':>4}  {'R2':>8}  {'RMSE':>10}  {'MAE':>10}  {'CCC':>8}")
            for m in per_fold:
                lines.append(f"  {m['fold']:>5}  {m['n']:>4}  {m['r2']:>8.3f}  "
                             f"{m['rmse']:>10.4g}  {m['mae']:>10.4g}  {m['ccc']:>8.3f}")
            r2s = [m["r2"] for m in per_fold if not np.isnan(m["r2"])]
            if len(r2s) > 1:
                lines.append(f"  R2 across folds: mean={np.mean(r2s):.3f}  sd={np.std(r2s):.3f}  "
                             f"min={np.min(r2s):.3f}  max={np.max(r2s):.3f}")
                lines.append("  (A large spread here means the single pooled R2 above is not "
                             "very stable -- expected with small sample sizes.)")
        imps = meta.get("importances") or []
        if imps:
            lines.append("")
            lines.append("Covariate importance (Random Forest, mean decrease in impurity)")
            lines.append("-" * 62)
            for rank, name, v in imps:
                lines.append(f"  {rank:>2}. {name:<40s} {100.0 * v:6.2f} %")
        lines.append("")
        lines.append("Files written")
        lines.append("-" * 62)
        lines.append(f"  Prediction raster            : {out_path}")
        if variance_path:
            lines.append(f"  Kriging variance raster      : {variance_path}")
        lines.append(f"  Observed vs predicted table  : "
                     f"{os.path.join(paths['obs_pred'], 'observed_vs_predicted.csv')}")
        lines.append(f"  Accuracy metrics             : "
                     f"{os.path.join(paths['accuracy'], 'accuracy_metrics.csv')}")
        if per_fold:
            lines.append(f"  Accuracy per fold            : "
                         f"{os.path.join(paths['accuracy'], 'accuracy_by_fold.csv')}")
        if imps:
            lines.append(f"  Covariate importance         : "
                         f"{os.path.join(paths['accuracy'], 'covariate_importance.csv')}")
        lines.append(f"  Scatter plot                 : "
                     f"{os.path.join(paths['accuracy'], 'observed_vs_predicted_plot.png')}")
        lines.append(f"  Training samples             : "
                     f"{os.path.join(paths['samples'], 'training_samples.csv')}")
        lines.append(f"  Covariate summary            : "
                     f"{os.path.join(paths['samples'], 'covariate_summary.csv')}")
        lines.append(f"  Calibration/validation split : {paths['split']}")
        for mf in (map_files or []):
            lines.append(f"  Publication map              : {mf}")
        lines.append("")
        lines.append("Column notes for observed_vs_predicted.csv")
        lines.append("-" * 62)
        lines.append("  observed_<field>  : the measured value at that sample point")
        lines.append("  predicted_<field> : the model's prediction for that point")
        lines.append("  residual...       : observed minus predicted (positive = "
                     "model under-predicted)")
        lines.append("  role              : whether the point was held out (validation) or "
                     "used to fit (calibration)")
        lines.append("  prediction_type   : how that prediction was produced -- only "
                     "held-out rows measure accuracy")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return path

    # ------------------------------------------------------------------
    # Results window
    # ------------------------------------------------------------------
    def _show_results_dialog(self, rows, field_name, feature_names, metrics, per_fold,
                              meta, png_path, results_root, map_png=None):
        dlg = QDialog(self)
        dlg.setWindowTitle(f"SAMKit results: {meta.get('model', '')}, {field_name}")
        vlayout = QVBoxLayout(dlg)

        summary = QLabel(self._results_headline(metrics, rows, meta))
        summary.setWordWrap(True)
        summary.setStyleSheet("font-size: 11px;")
        vlayout.addWidget(summary)

        tabs = QTabWidget()

        # --- observed vs predicted table ---
        cols = ["#", f"observed {field_name}", f"predicted {field_name}",
                "residual", "abs residual", "role", "fold", "x", "y"]
        table = QTableWidget(len(rows), len(cols))
        table.setHorizontalHeaderLabels(cols)
        # Sorting MUST stay off while we populate the table. With sorting on,
        # QTableWidget re-sorts the rows on every setItem() call, so the fixed
        # row index i in the loop below no longer points at the row we think it
        # does -- items land in shifting rows and most cells end up blank, even
        # though `rows` (and therefore the saved CSV) is complete. Enable it
        # only after every item is in place.
        table.setSortingEnabled(False)
        for i, r in enumerate(rows):
            def _num(v):
                item = QTableWidgetItem()
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    item.setText("")
                else:
                    item.setData(Qt.DisplayRole, float(f"{float(v):.6g}"))
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                return item
            table.setItem(i, 0, _num(r["point_index"]))
            table.setItem(i, 1, _num(r["observed"]))
            table.setItem(i, 2, _num(r["predicted"]))
            table.setItem(i, 3, _num(r["residual"]))
            table.setItem(i, 4, _num(r["abs_residual"]))
            role_item = QTableWidgetItem(r["role"])
            if not r["role"].startswith("validation"):
                role_item.setForeground(QColor(140, 140, 140))
            table.setItem(i, 5, role_item)
            table.setItem(i, 6, QTableWidgetItem(str(r["fold"])))
            table.setItem(i, 7, _num(r["x"]))
            table.setItem(i, 8, _num(r["y"]))
        table.resizeColumnsToContents()
        table.setSortingEnabled(True)  # safe now that every cell is populated
        tabs.addTab(table, f"Observed vs predicted ({len(rows)})")

        # --- per-fold accuracy table ---
        if per_fold:
            fold_cols = ["fold", "n", "R2", "RMSE", "MAE", "bias", "CCC"]
            ftable = QTableWidget(len(per_fold), len(fold_cols))
            ftable.setHorizontalHeaderLabels(fold_cols)
            for i, m in enumerate(per_fold):
                vals = [m["fold"], m["n"], m["r2"], m["rmse"], m["mae"], m["bias"], m["ccc"]]
                for j, v in enumerate(vals):
                    item = QTableWidgetItem()
                    item.setData(Qt.DisplayRole, float(f"{float(v):.6g}"))
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    ftable.setItem(i, j, item)
            ftable.resizeColumnsToContents()
            tabs.addTab(ftable, "Accuracy by fold")

        # --- covariate importance ---
        imps = meta.get("importances") or []
        if imps:
            itable = QTableWidget(len(imps), 3)
            itable.setHorizontalHeaderLabels(["rank", "covariate", "importance (%)"])
            for i, (rank, name, v) in enumerate(imps):
                a = QTableWidgetItem(); a.setData(Qt.DisplayRole, int(rank))
                c = QTableWidgetItem(); c.setData(Qt.DisplayRole, round(100.0 * v, 2))
                c.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                itable.setItem(i, 0, a)
                itable.setItem(i, 1, QTableWidgetItem(name))
                itable.setItem(i, 2, c)
            itable.resizeColumnsToContents()
            tabs.addTab(itable, "Covariate importance")

        # --- scatter plot ---
        if png_path and os.path.exists(png_path):
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            img_label = QLabel()
            img_label.setPixmap(QPixmap(png_path))
            img_label.setAlignment(Qt.AlignCenter)
            scroll.setWidget(img_label)
            tabs.addTab(scroll, "Scatter plot")

        # --- publication map preview ---
        if map_png and os.path.exists(map_png):
            mscroll = QScrollArea()
            mscroll.setWidgetResizable(True)
            mlabel = QLabel()
            pm = QPixmap(map_png)
            if not pm.isNull():
                mlabel.setPixmap(pm.scaledToWidth(900, Qt.SmoothTransformation))
            mlabel.setAlignment(Qt.AlignCenter)
            mscroll.setWidget(mlabel)
            tabs.addTab(mscroll, "Map")

        vlayout.addWidget(tabs)

        btn_row = QHBoxLayout()
        save_btn = QPushButton("Save table as CSV…")

        def _save_as():
            dest, _ = QFileDialog.getSaveFileName(
                dlg, "Save observed vs predicted table",
                "observed_vs_predicted.csv", filter="CSV (*.csv)")
            if not dest:
                return
            if not dest.lower().endswith(".csv"):
                dest += ".csv"
            try:
                self._write_obs_pred_csv(dest, rows, field_name, feature_names)
                QMessageBox.information(dlg, "Save table", f"Saved to:\n{dest}")
            except Exception as e:
                QMessageBox.warning(dlg, "Save table", f"Could not save:\n{e}")

        save_btn.clicked.connect(_save_as)
        btn_row.addWidget(save_btn)

        open_btn = QPushButton("Open results folder")
        open_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(results_root)))
        btn_row.addWidget(open_btn)
        btn_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.close)
        btn_row.addWidget(close_btn)
        vlayout.addLayout(btn_row)

        dlg.resize(940, 720)
        dlg.show()
        self._results_dialog = dlg  # keep a reference so it isn't garbage-collected
        return dlg

    def _results_headline(self, metrics, rows, meta):
        n_val = len(self._validation_rows(rows))
        head = (f"<b>{meta.get('model', '')}</b> predicting <b>{meta.get('field', '')}</b> "
                f"&nbsp;|&nbsp; {meta.get('validation_method', '')} "
                f"({meta.get('n_folds_or_split', '')}) &nbsp;|&nbsp; "
                f"{n_val} held-out point(s)")
        scen = meta.get("scenario")
        if scen:
            head += (f"<br><span style='color:#b36b00'><b>Point-data scenario:</b> "
                     f"{scen.get('percent', '')} % of '{scen.get('source_layer_name', '')}' -- "
                     f"model fitted with {meta.get('n_fit', meta.get('n_samples', ''))} sample points "
                     f"(seed {scen.get('seed', '')}).</span>")
        else:
            what = ("calibration points inside/outside the region"
                    if meta.get("validation_method") == "spatial block"
                    else "all usable points of the selected layer")
            head += (f"<br>Map model fitted with <b>{meta.get('n_fit', meta.get('n_samples', ''))}"
                     f"</b> sample points ({what}).")
        if metrics:
            head += (f"<br>R² = <b>{metrics['r2']:.3f}</b> &nbsp; "
                     f"RMSE = <b>{metrics['rmse']:.4g}</b> &nbsp; "
                     f"MAE = <b>{metrics['mae']:.4g}</b> &nbsp; "
                     f"bias = <b>{metrics['bias']:.4g}</b> &nbsp; "
                     f"CCC = <b>{metrics['ccc']:.3f}</b>")
        else:
            head += "<br>No validation metrics were computed for this run."
        return head

    # ------------------------------------------------------------------
    # Core run logic
    # ------------------------------------------------------------------
    def run(self):
        points_layer = self.points_combo.currentLayer()
        field_name = self.field_combo.currentText()
        out_path = self.out_path_edit.text().strip()
        model_name = self.model_combo.currentText()
        raster_layers = self._checked_raster_layers()
        val_method_idx = self.val_method_combo.currentIndex()  # 0 = k-fold, 1 = spatial block

        if points_layer is None:
            QMessageBox.warning(self, "Spatial Predictive Modeling", "Please select a point vector layer.")
            return
        # The model always uses exactly the layer selected under "Georeferenced
        # points". A point-data scenario (subset) is used only when that layer
        # IS a scenario -- i.e. after "Create scenario..." or "Load in QGIS".
        # (Earlier builds also drew a random subset at run time whenever the
        # percentage box was below 100 %, which silently degraded maps.)
        scen = self._scenario_meta(points_layer)
        if scen:
            self._log(f"Using point-data scenario '{scen.get('name', '')}': "
                      f"{scen.get('percent', '')} % of '{scen.get('source_layer_name', '')}' "
                      f"({scen.get('n_kept', '')} points, seed {scen.get('seed', '')}).")
            if field_name and scen.get("field") and scen.get("field") != field_name:
                self._log(f"  Note: this scenario was drawn using field '{scen.get('field')}'; "
                          f"you are now modelling '{field_name}'.")
        if not field_name:
            QMessageBox.warning(self, "Spatial Predictive Modeling", "Please select a numeric attribute field.")
            return
        if not raster_layers:
            QMessageBox.warning(self, "Spatial Predictive Modeling", "Please check at least one covariate/reference raster.")
            return
        if not out_path:
            QMessageBox.warning(self, "Spatial Predictive Modeling", "Please choose an output raster path.")
            return
        if val_method_idx == 1 and self.calib_polygon_combo.currentLayer() is None:
            QMessageBox.warning(self, "Spatial Predictive Modeling",
                                 "Please select (or first digitize) a calibration polygon layer "
                                 "for the spatial block split, or switch back to k-fold cross-validation.")
            return
        if val_method_idx == 2:
            if self.extval_points_combo.currentLayer() is None:
                QMessageBox.warning(self, "Spatial Predictive Modeling",
                                     "Please select a separate validation point layer for the "
                                     "external validation strategy, or switch to another validation "
                                     "strategy.")
                return
            if not self.extval_same_field_chk.isChecked() and not self.extval_field_combo.currentText():
                QMessageBox.warning(self, "Spatial Predictive Modeling",
                                     "Please select the validation attribute field, or check "
                                     "'Same field name as the calibration attribute above'.")
                return

        if self._restart_required:
            # Packages were installed during this session. Importing them now
            # would load compiled extensions into a process that already has a
            # different numpy loaded, which can segfault and take QGIS down
            # with no error message. Refuse rather than risk it.
            QMessageBox.warning(
                self, "Restart required",
                "Python packages were installed during this QGIS session.\n\n"
                "Please restart QGIS before running a model. Loading newly "
                "installed compiled packages into an already-running session "
                "can crash QGIS."
            )
            return

        self.run_btn.setEnabled(False)
        self.progress.setValue(0)
        self.log.clear()

        try:
            paths = self._prepare_results_dir(out_path)
            self._log(f"All tables, plots and split files for this run go to:\n  {paths['root']}")
            self._log("")
            if scen:
                self._log(f"Using point-data scenario '{scen.get('name', '')}': "
                          f"{scen.get('n_kept', '')} of {scen.get('n_source_points', '')} points "
                          f"({scen.get('percent', '')} %, seed {scen.get('seed', '')}) "
                          f"from '{scen.get('source_layer_name', '')}'.")
                self._log("")

            data = self._extract_training_data(points_layer, field_name, raster_layers,
                                               self.align_chk.isChecked(), self._log)
            X, y, xs_v, ys_v = data["X"], data["y"], data["xs"], data["ys"]
            feature_names, raster_paths, ref_ri = data["feature_names"], data["raster_paths"], data["ref_ri"]
            self._warn_if_geographic(ref_ri, model_name)
            self.progress.setValue(25)

            self._log_data_diagnostics(y, X, feature_names, field_name)

            try:
                samples_csv_path = os.path.join(paths["samples"], "training_samples.csv")
                self._write_training_csv(samples_csv_path, xs_v, ys_v, y, X, feature_names, field_name)
                self._log(f"Wrote extracted training sample table to: {samples_csv_path}")
                cov_summary_path = os.path.join(paths["samples"], "covariate_summary.csv")
                self._write_covariate_summary_csv(cov_summary_path, y, X, feature_names, field_name)
                self._log(f"Wrote target/covariate summary table to: {cov_summary_path}")
            except Exception as e:
                self._log(f"(Could not write sample tables: {e})")

            self.progress.setValue(35)

            # --- Determine validation strategy ---
            folds = None
            spatial_calib_mask = None
            validation_method_label = ""
            n_folds_or_split_label = ""

            if val_method_idx == 0:
                n_folds = self.eval_folds.value() if self.eval_chk.isChecked() else 0
                use_cal_pct = self.eval_cal_pct_chk.isChecked()
                cal_pct = self.eval_cal_pct.value()
                if use_cal_pct:
                    validation_method_label = "repeated random-split cross-validation"
                    n_folds_or_split_label = (f"k={n_folds}, {cal_pct}% calib / "
                                              f"{100 - cal_pct}% valid" if n_folds else "none")
                else:
                    validation_method_label = "k-fold cross-validation"
                    n_folds_or_split_label = f"k={n_folds}" if n_folds else "none"
                if n_folds and n_folds > 1:
                    if use_cal_pct:
                        folds = dsm_models.repeated_split_indices(
                            len(y), n_folds, cal_pct / 100.0, seed=42)
                        self._log(f"Repeated random splits: {n_folds} folds, each with "
                                  f"{len(y) - len(folds[0])} calibration / {len(folds[0])} "
                                  f"validation points ({cal_pct}% / {100 - cal_pct}%).")
                    else:
                        folds = dsm_models.kfold_indices(len(y), n_folds, seed=42)
                    try:
                        self._save_fold_files(paths["split"], xs_v, ys_v, y, X, feature_names,
                                              field_name, folds, self._log)
                    except Exception as e:
                        self._log(f"(Could not save fold files: {e})")
            elif val_method_idx == 1:
                validation_method_label = "spatial block"
                calib_inside = self.calib_inside_combo.currentIndex() == 0
                polygon_layer = self.calib_polygon_combo.currentLayer()
                self._log(f"Splitting points using calibration polygon '{polygon_layer.name()}' "
                          f"({'inside' if calib_inside else 'outside'} = calibration)…")
                spatial_calib_mask = self._get_spatial_calibration_mask(
                    xs_v, ys_v, ref_ri, polygon_layer, calib_inside, self._log)
                n_folds_or_split_label = (f"{int(spatial_calib_mask.sum())} calib / "
                                          f"{int((~spatial_calib_mask).sum())} valid")

                split_folder = paths["split"]
                self._write_training_csv(os.path.join(split_folder, "calibration_points.csv"),
                                         xs_v[spatial_calib_mask], ys_v[spatial_calib_mask],
                                         y[spatial_calib_mask], X[spatial_calib_mask],
                                         feature_names, field_name)
                self._write_training_csv(os.path.join(split_folder, "validation_points.csv"),
                                         xs_v[~spatial_calib_mask], ys_v[~spatial_calib_mask],
                                         y[~spatial_calib_mask], X[~spatial_calib_mask],
                                         feature_names, field_name)
                self._log(f"Saved calibration/validation split files to: {split_folder}")
                region_path = os.path.join(split_folder, "calibration_region.gpkg")
                try:
                    self._remove_layers_using([region_path])
                    self._write_polygon_layer(polygon_layer, region_path, "calibration_region")
                    self._log(f"Saved the calibration region boundary to: {region_path}")
                except Exception as e:
                    self._log(f"(Could not save the calibration region boundary: {e})")

                # For fitting, use only the calibration subset so the final map matches
                # the reported spatial-block accuracy exactly.
                X_fit, y_fit, xs_fit, ys_fit = (X[spatial_calib_mask], y[spatial_calib_mask],
                                                xs_v[spatial_calib_mask], ys_v[spatial_calib_mask])
                X_val, y_val, xs_val, ys_val = (X[~spatial_calib_mask], y[~spatial_calib_mask],
                                                xs_v[~spatial_calib_mask], ys_v[~spatial_calib_mask])

            else:  # val_method_idx == 2: external validation layer
                validation_method_label = "external validation layer"
                extval_layer = self.extval_points_combo.currentLayer()
                extval_field = field_name if self.extval_same_field_chk.isChecked() \
                    else self.extval_field_combo.currentText()
                self._log(f"Extracting external validation points from '{extval_layer.name()}', "
                          f"field '{extval_field}'…")
                X_val, y_val, xs_val, ys_val = self._extract_points_and_sample(
                    extval_layer, extval_field, raster_paths, ref_ri, self._log, min_points=3)

                # Entire calibration layer is used for fitting -- no internal split.
                X_fit, y_fit, xs_fit, ys_fit = X, y, xs_v, ys_v
                n_folds_or_split_label = f"{len(y_fit)} calib (all) / {len(y_val)} external valid"

                split_folder = paths["split"]
                self._write_training_csv(os.path.join(split_folder, "calibration_points.csv"),
                                         xs_fit, ys_fit, y_fit, X_fit, feature_names, field_name)
                self._write_training_csv(os.path.join(split_folder, "external_validation_points.csv"),
                                         xs_val, ys_val, y_val, X_val, feature_names, extval_field)
                self._log(f"Saved calibration/external-validation point files to: {split_folder}")

            self.progress.setValue(40)

            hyperparams_str = ""
            variance_arr = None
            # Filled by the model layer with the actual held-out predictions
            # (not just their summary metrics) so an observed-vs-predicted
            # table and plot can be written for the property of interest.
            cv_out = {}
            calib_pred = None
            val_pred = None

            # --- Fit + predict, per model ---
            if val_method_idx == 0:
                # k-fold path: fit on ALL points, CV metrics computed internally
                if model_name == "Random Forest":
                    max_depth = None if self.rf_max_depth.value() == 0 else self.rf_max_depth.value()
                    hyperparams_str = (f"n_estimators={self.rf_n_estimators.value()}, "
                                       f"max_depth={max_depth}")
                    model, predict_fn, metrics = dsm_models.fit_random_forest(
                        X, y, n_estimators=self.rf_n_estimators.value(), max_depth=max_depth, preset=self._rf_preset(),
                        n_folds=n_folds, folds=folds, log=self._log, cv_out=cv_out)
                    self.progress.setValue(60)
                    self._log("Predicting across the raster stack…")
                    stack, _ = utils.read_raster_stack(raster_paths)
                    pred_arr = dsm_models.predict_over_stack(predict_fn, stack)

                elif model_name == "Support Vector Machine (SVM)":
                    hyperparams_str = (f"kernel={self.svm_kernel.currentText()}, C={self.svm_C.value()}, "
                                       f"epsilon={self.svm_epsilon.value()}, gamma={self.svm_gamma.currentText()}")
                    model, predict_fn, metrics = dsm_models.fit_svm(
                        X, y, feature_names, kernel=self.svm_kernel.currentText(),
                        C=self.svm_C.value(), epsilon=self.svm_epsilon.value(),
                        gamma=self.svm_gamma.currentText(), n_folds=n_folds, folds=folds,
                        log=self._log, cv_out=cv_out)
                    self.progress.setValue(60)
                    self._log("Predicting across the raster stack…")
                    stack, _ = utils.read_raster_stack(raster_paths)
                    pred_arr = dsm_models.predict_over_stack(predict_fn, stack)

                elif model_name == "Ordinary Kriging":
                    variogram_model = self.ok_variogram_combo.currentText()
                    vparams = self._get_variogram_parameters("ok", variogram_model, self._log)
                    hyperparams_str = f"variogram={variogram_model}" + (
                        f" (manual: {vparams})" if vparams else " (auto-fit)")
                    ok_model, metrics = dsm_models.fit_ordinary_kriging(
                        xs_v, ys_v, y, variogram_model=variogram_model,
                        variogram_parameters=vparams, variogram_fitting=self._vfit("ok"), n_folds=n_folds, folds=folds,
                        log=self._log, cv_out=cv_out)
                    self.progress.setValue(60)
                    self._log("Predicting on output grid via kriging (can be slow for large grids)…")
                    rows, cols = ref_ri.shape
                    col_centers = np.array([ref_ri.pixel_to_xy(0, c)[0] for c in range(cols)])
                    row_centers = np.array([ref_ri.pixel_to_xy(r, 0)[1] for r in range(rows)])
                    # Same kriging prediction as before; the variance (which pykrige
                    # computes anyway) is now kept, so OK writes its uncertainty
                    # raster under k-fold as well, as documented.
                    pred_arr, variance_arr = dsm_models.predict_kriging_grid(
                        ok_model, col_centers, row_centers, log=self._log, return_variance=True)

                elif model_name == "Regression Kriging":
                    trend = self.rk_trend_combo.currentText()
                    variogram_model = self.rk_variogram_combo.currentText()
                    vparams = self._get_variogram_parameters("rk", variogram_model, self._log)
                    hyperparams_str = f"trend={trend}, variogram={variogram_model}" + (
                        f" (manual: {vparams})" if vparams else " (auto-fit)")
                    rk = dsm_models.fit_regression_kriging(
                        X, y, xs_v, ys_v, feature_names, trend=trend,
                        variogram_model=variogram_model, variogram_parameters=vparams, variogram_fitting=self._vfit("rk"), residuals=self._rk_residuals(), rf_preset=self._rf_preset(),
                        n_estimators=self.rf_n_estimators.value(),
                        n_folds=n_folds, folds=folds, log=self._log, cv_out=cv_out)
                    metrics = rk["metrics"]
                    self.progress.setValue(55)
                    self._log("Predicting trend across the raster stack…")
                    stack, _ = utils.read_raster_stack(raster_paths)
                    trend_arr = dsm_models.predict_over_stack(rk["trend_predict_fn"], stack)
                    self.progress.setValue(70)
                    self._log("Kriging residuals across the output grid…")
                    rows, cols = ref_ri.shape
                    col_centers = np.array([ref_ri.pixel_to_xy(0, c)[0] for c in range(cols)])
                    row_centers = np.array([ref_ri.pixel_to_xy(r, 0)[1] for r in range(rows)])
                    resid_arr, variance_arr = dsm_models.predict_kriging_grid(
                        rk["ok_resid"], col_centers, row_centers, log=self._log, return_variance=True)
                    pred_arr = trend_arr + resid_arr
                    pred_arr[np.isnan(trend_arr)] = np.nan
                    self._log("(Note: exported variance is the residual-kriging variance only -- "
                              "it does not include the trend model's own uncertainty.)")

                else:
                    raise RuntimeError(f"Unknown model: {model_name}")

            else:
                # spatial block path: fit on calibration subset only, evaluate on validation subset
                if model_name == "Random Forest":
                    max_depth = None if self.rf_max_depth.value() == 0 else self.rf_max_depth.value()
                    hyperparams_str = (f"n_estimators={self.rf_n_estimators.value()}, "
                                       f"max_depth={max_depth}")
                    model, predict_fn, _ = dsm_models.fit_random_forest(
                        X_fit, y_fit, n_estimators=self.rf_n_estimators.value(), max_depth=max_depth, preset=self._rf_preset(),
                        n_folds=0, log=self._log)
                    calib_pred = predict_fn(X_fit)
                    val_pred = predict_fn(X_val) if len(y_val) else np.array([])
                    metrics = dsm_models.compute_metrics(y_val, val_pred) if len(y_val) else None
                    self.progress.setValue(60)
                    self._log("Predicting across the raster stack (calibration-only model)…")
                    stack, _ = utils.read_raster_stack(raster_paths)
                    pred_arr = dsm_models.predict_over_stack(predict_fn, stack)

                elif model_name == "Support Vector Machine (SVM)":
                    hyperparams_str = (f"kernel={self.svm_kernel.currentText()}, C={self.svm_C.value()}, "
                                       f"epsilon={self.svm_epsilon.value()}, gamma={self.svm_gamma.currentText()}")
                    model, predict_fn, _ = dsm_models.fit_svm(
                        X_fit, y_fit, feature_names, kernel=self.svm_kernel.currentText(),
                        C=self.svm_C.value(), epsilon=self.svm_epsilon.value(),
                        gamma=self.svm_gamma.currentText(), n_folds=0, log=self._log)
                    calib_pred = predict_fn(X_fit)
                    val_pred = predict_fn(X_val) if len(y_val) else np.array([])
                    metrics = dsm_models.compute_metrics(y_val, val_pred) if len(y_val) else None
                    self.progress.setValue(60)
                    self._log("Predicting across the raster stack (calibration-only model)…")
                    stack, _ = utils.read_raster_stack(raster_paths)
                    pred_arr = dsm_models.predict_over_stack(predict_fn, stack)

                elif model_name == "Ordinary Kriging":
                    variogram_model = self.ok_variogram_combo.currentText()
                    vparams = self._get_variogram_parameters("ok", variogram_model, self._log)
                    hyperparams_str = f"variogram={variogram_model}" + (
                        f" (manual: {vparams})" if vparams else " (auto-fit)")
                    ok_model, _ = dsm_models.fit_ordinary_kriging(
                        xs_fit, ys_fit, y_fit, variogram_model=variogram_model,
                        variogram_parameters=vparams, variogram_fitting=self._vfit("ok"), n_folds=0, log=self._log)
                    calib_pred = dsm_models.predict_kriging_points(ok_model, xs_fit, ys_fit)
                    if len(y_val):
                        val_pred = dsm_models.predict_kriging_points(ok_model, xs_val, ys_val)
                        metrics = dsm_models.compute_metrics(y_val, val_pred)
                    else:
                        val_pred = np.array([])
                        metrics = None
                    self.progress.setValue(60)
                    self._log("Predicting on output grid via kriging (can be slow for large grids)…")
                    rows, cols = ref_ri.shape
                    col_centers = np.array([ref_ri.pixel_to_xy(0, c)[0] for c in range(cols)])
                    row_centers = np.array([ref_ri.pixel_to_xy(r, 0)[1] for r in range(rows)])
                    pred_arr, variance_arr = dsm_models.predict_kriging_grid(
                        ok_model, col_centers, row_centers, log=self._log, return_variance=True)

                elif model_name == "Regression Kriging":
                    trend = self.rk_trend_combo.currentText()
                    variogram_model = self.rk_variogram_combo.currentText()
                    vparams = self._get_variogram_parameters("rk", variogram_model, self._log)
                    hyperparams_str = f"trend={trend}, variogram={variogram_model}" + (
                        f" (manual: {vparams})" if vparams else " (auto-fit)")
                    rk = dsm_models.fit_regression_kriging(
                        X_fit, y_fit, xs_fit, ys_fit, feature_names, trend=trend,
                        variogram_model=variogram_model, variogram_parameters=vparams, variogram_fitting=self._vfit("rk"), residuals=self._rk_residuals(), rf_preset=self._rf_preset(),
                        n_estimators=self.rf_n_estimators.value(), n_folds=0, log=self._log)
                    calib_pred = (rk["trend_predict_fn"](X_fit)
                                  + dsm_models.predict_kriging_points(rk["ok_resid"], xs_fit, ys_fit))
                    if len(y_val):
                        trend_val_pred = rk["trend_predict_fn"](X_val)
                        resid_val_pred = dsm_models.predict_kriging_points(rk["ok_resid"], xs_val, ys_val)
                        val_pred = trend_val_pred + resid_val_pred
                        metrics = dsm_models.compute_metrics(y_val, val_pred)
                    else:
                        val_pred = np.array([])
                        metrics = None
                    self.progress.setValue(55)
                    self._log("Predicting trend across the raster stack (calibration-only model)…")
                    stack, _ = utils.read_raster_stack(raster_paths)
                    trend_arr = dsm_models.predict_over_stack(rk["trend_predict_fn"], stack)
                    self.progress.setValue(70)
                    self._log("Kriging residuals across the output grid…")
                    rows, cols = ref_ri.shape
                    col_centers = np.array([ref_ri.pixel_to_xy(0, c)[0] for c in range(cols)])
                    row_centers = np.array([ref_ri.pixel_to_xy(r, 0)[1] for r in range(rows)])
                    resid_arr, variance_arr = dsm_models.predict_kriging_grid(
                        rk["ok_resid"], col_centers, row_centers, log=self._log, return_variance=True)
                    pred_arr = trend_arr + resid_arr
                    pred_arr[np.isnan(trend_arr)] = np.nan
                    self._log("(Note: exported variance is the residual-kriging variance only -- "
                              "it does not include the trend model's own uncertainty.)")

                else:
                    raise RuntimeError(f"Unknown model: {model_name}")

                if metrics:
                    self._log(f"[{validation_method_label.title()} validation] "
                              f"n_validation={len(y_val)}: "
                              f"R2={metrics['r2']:.3f}  RMSE={metrics['rmse']:.3f}  MAE={metrics['mae']:.3f}  "
                              f"bias={metrics['bias']:.3f}  CCC={metrics['ccc']:.3f}")

            # --- Covariate importance ranking (RF, or the RF trend of RK) ---
            importances = []
            try:
                fitted = model if model_name == "Random Forest" else (
                    rk if model_name == "Regression Kriging" else None)
                importances = self._covariate_importances(model_name, fitted, feature_names)
            except NameError:
                importances = []
            if importances:
                label = ("Random Forest" if model_name == "Random Forest"
                         else "Random Forest trend of the regression kriging")
                self._log("")
                self._log(f"--- Covariate importance ({label}; mean decrease in impurity) ---")
                for rank, name, v in importances:
                    self._log(f"  {rank:>2}. {name:<40s} {100.0 * v:6.2f} %")
                try:
                    imp_csv = self._write_importances(
                        os.path.join(paths["accuracy"], "covariate_importance.csv"),
                        importances, model_name)
                    self._log(f"Wrote covariate importance ranking to: {imp_csv}")
                except Exception as e:
                    self._log(f"(Could not write the covariate importance table: {e})")

            # --- Observed vs predicted values for the property of interest ---
            per_fold = []
            if val_method_idx == 0:
                obs_pred_rows = self._build_obs_pred_from_cv(cv_out, xs_v, ys_v, y, X, self._log)
                per_fold = cv_out.get("per_fold") or []
            else:
                obs_pred_rows = self._build_obs_pred_from_split(
                    xs_fit, ys_fit, y_fit, X_fit, calib_pred,
                    xs_val, ys_val, y_val, X_val, val_pred, model_name)
            self._log_obs_pred_preview(obs_pred_rows, field_name)

            self.progress.setValue(90)

            # --- Write output raster ---
            utils.write_raster(out_path, pred_arr, ref_ri)
            self._log(f"Wrote output raster: {out_path}")

            if variance_arr is not None:
                variance_path = os.path.splitext(out_path)[0] + "_kriging_variance.tif"
                try:
                    utils.write_raster(variance_path, variance_arr, ref_ri)
                    self._log(f"Wrote kriging variance raster to: {variance_path}")
                except Exception as e:
                    self._log(f"(Could not write kriging variance raster: {e})")

            # --- Save accuracy metrics + observed/predicted outputs ---
            meta = {
                "model": model_name,
                "field": field_name,
                "validation_method": validation_method_label,
                "n_folds_or_split": n_folds_or_split_label,
                "n_samples": len(y),
                # points the final (mapped) model was actually fitted on
                "n_fit": int(spatial_calib_mask.sum()) if val_method_idx == 1 else len(y),
                "importances": importances,
                "n_covariates": len(feature_names),
                "feature_names": feature_names,
                "hyperparams": hyperparams_str,
                "scenario": self._scenario_meta(points_layer),
            }
            try:
                self._write_metrics_log(out_path, meta, metrics, self._log,
                                        accuracy_dir=paths["accuracy"])
            except Exception as e:
                self._log(f"(Could not write metrics log: {e})")

            obs_pred_csv = None
            scatter_png = None
            if obs_pred_rows:
                try:
                    obs_pred_csv = self._write_obs_pred_csv(
                        os.path.join(paths["obs_pred"], "observed_vs_predicted.csv"),
                        obs_pred_rows, field_name, feature_names)
                    self._log(f"Wrote observed vs predicted table to: {obs_pred_csv}")
                except Exception as e:
                    self._log(f"(Could not write observed vs predicted table: {e})")

                if per_fold:
                    try:
                        fold_csv = self._write_per_fold_metrics_csv(
                            os.path.join(paths["accuracy"], "accuracy_by_fold.csv"),
                            per_fold, field_name)
                        self._log(f"Wrote per-fold accuracy table to: {fold_csv}")
                        r2s = [m["r2"] for m in per_fold if not np.isnan(m["r2"])]
                        if len(r2s) > 1:
                            self._log(f"  R2 across folds: mean={np.mean(r2s):.3f}  "
                                      f"sd={np.std(r2s):.3f}  min={np.min(r2s):.3f}  "
                                      f"max={np.max(r2s):.3f}")
                    except Exception as e:
                        self._log(f"(Could not write per-fold accuracy table: {e})")

                try:
                    scatter_png = self._render_scatter_png(
                        os.path.join(paths["accuracy"], "observed_vs_predicted_plot.png"),
                        obs_pred_rows, field_name, model_name, metrics,
                        f"{validation_method_label} ({n_folds_or_split_label})")
                    self._log(f"Wrote observed vs predicted scatter plot to: {scatter_png}")
                except Exception as e:
                    self._log(f"(Could not render the scatter plot: {e}. "
                              f"The CSV table above is still valid.)")
                    scatter_png = None

            if variance_arr is None:
                variance_path = None

            # --- Output layers (created once, styled by the map export, then loaded) ---
            load_layers = self.load_output_chk.isChecked()
            rl = QgsRasterLayer(out_path, os.path.basename(out_path))
            vl = (QgsRasterLayer(variance_path, os.path.basename(variance_path))
                  if variance_path else None)

            # --- Publication-quality maps ---
            map_files = []
            if self.maps_chk.isChecked():
                self.progress.setValue(93)
                if val_method_idx == 0:
                    map_points = [(x, yy, "sample") for x, yy in zip(xs_v, ys_v)]
                else:
                    map_points = ([(x, yy, "calibration") for x, yy in zip(xs_fit, ys_fit)]
                                  + [(x, yy, "validation") for x, yy in zip(xs_val, ys_val)])
                register = load_layers and self.map_layouts_chk.isChecked()
                try:
                    map_files = self._export_run_maps(
                        paths, out_path, variance_path, field_name, model_name,
                        f"{validation_method_label} ({n_folds_or_split_label})",
                        metrics, len(y), feature_names, map_points,
                        rl if rl.isValid() else None,
                        vl if vl is not None and vl.isValid() else None,
                        register)
                    if register and map_files:
                        self._log("Added the map layout(s) to Project > Layout Manager "
                                  "for further editing.")
                except Exception as e:
                    self._log(f"(Could not render the publication maps: {e}. "
                              f"The prediction raster and tables are unaffected.)")
                    self._log(traceback.format_exc())

            try:
                summary_path = self._write_run_summary(
                    os.path.join(paths["root"], "00_RUN_SUMMARY.txt"), meta, metrics,
                    per_fold, obs_pred_rows, paths, out_path,
                    variance_path, map_files=map_files)
                self._log(f"Wrote run summary to: {summary_path}")
            except Exception as e:
                self._log(f"(Could not write run summary: {e})")

            self.progress.setValue(100)

            if load_layers:
                if rl.isValid():
                    QgsProject.instance().addMapLayer(rl)
                else:
                    self._log("  (warning: could not load output raster into QGIS)")
                if vl is not None:
                    if vl.isValid():
                        QgsProject.instance().addMapLayer(vl)
                    else:
                        self._log("  (warning: could not load kriging variance raster into QGIS)")
                # Layouts registered in the Layout Manager draw the sample points
                # from their GeoPackage layer; it has to be in the project too.
                if map_files and self.map_layouts_chk.isChecked():
                    self._add_map_point_layer_to_project(rl)

            if obs_pred_rows:
                map_png = next((f for f in map_files if f.lower().endswith(".png")), None)
                self._show_results_dialog(obs_pred_rows, field_name, feature_names, metrics,
                                          per_fold, meta, scatter_png, paths["root"],
                                          map_png=map_png)
            else:
                msg = f"Finished. Output written to:\n{out_path}"
                if variance_arr is not None:
                    msg += f"\nKriging variance written to:\n{variance_path}"
                msg += f"\n\nAll tables for this run are in:\n{paths['root']}"
                if map_files:
                    msg += f"\n\nPublication maps written to:\n{paths['maps']}"
                if metrics:
                    msg += (f"\n\nValidation metrics ({validation_method_label}):\n"
                            f"R2 = {metrics['r2']:.3f}\n"
                            f"RMSE = {metrics['rmse']:.3f}\nMAE = {metrics['mae']:.3f}\n"
                            f"bias = {metrics['bias']:.3f}\nCCC = {metrics['ccc']:.3f}")
                QMessageBox.information(self, "Spatial Predictive Modeling", msg)

        except dsm_models.MissingDependencyError as e:
            self._log("MISSING DEPENDENCY: " + str(e))
            QMessageBox.critical(self, "Missing dependency", str(e))
        except Exception as e:
            self._log("ERROR: " + str(e))
            self._log(traceback.format_exc())
            QMessageBox.critical(self, "Spatial Predictive Modeling", f"Processing failed:\n{e}")
        finally:
            self.run_btn.setEnabled(True)
