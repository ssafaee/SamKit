# -*- coding: utf-8 -*-
from qgis.PyQt.QtWidgets import QDialog, QVBoxLayout, QTabWidget, QScrollArea, QApplication
from qgis.PyQt.QtCore import Qt

from .dem_tab import DemTab
from .terrain_tab import TerrainAttributesTab
from .dsm_tab import DSMTab


class MainDialog(QDialog):

    def __init__(self, iface, parent=None):
        super().__init__(parent)
        self.iface = iface
        self.setWindowTitle("SAMKit (Spatial Analysis and Mapping Toolkit)")
        self.setSizeGripEnabled(True)

        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.dem_tab = DemTab(iface)
        self.terrain_tab = TerrainAttributesTab(iface)
        self.dsm_tab = DSMTab(iface)
        self.tabs.addTab(self._wrap_in_scroll_area(self.dem_tab), "1. DEM (download / project)")
        self.tabs.addTab(self._wrap_in_scroll_area(self.terrain_tab), "2. Environmental Covariates")
        self.tabs.addTab(self._wrap_in_scroll_area(self.dsm_tab), "3. Spatial Predictive Modeling")
        layout.addWidget(self.tabs)

        # A DEM prepared in tab 1 is handed straight to tab 2, so the normal
        # workflow doesn't require re-picking the layer by hand.
        self.dem_tab.demReady.connect(self._on_dem_ready)

        self._fit_to_available_screen()

    def _on_dem_ready(self, layer):
        try:
            self.terrain_tab.dem_combo.setLayer(layer)
            self.tabs.setCurrentIndex(1)
        except Exception:
            pass  # never let a convenience hand-off break the dialog

    @staticmethod
    def _wrap_in_scroll_area(widget):
        """Wrap a tab's content in a scroll area so the dialog never has to be
        taller than the screen -- if content doesn't fit, it scrolls instead
        of forcing the window (or QGIS itself) to grow oversized."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(widget)
        return scroll

    def _fit_to_available_screen(self):
        """Size the dialog to comfortably fit on screen, capped well below
        typical screen/QGIS window dimensions, and never larger than the
        available screen geometry."""
        desired_width, desired_height = 820, 760
        screen = QApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            width = min(desired_width, int(avail.width() * 0.90))
            height = min(desired_height, int(avail.height() * 0.85))
        else:
            width, height = desired_width, desired_height
        self.resize(width, height)
        # Allow shrinking further than the initial size if the user's screen
        # or QGIS main window is small; content will scroll as needed.
        self.setMinimumSize(480, 360)
