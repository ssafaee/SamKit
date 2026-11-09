# -*- coding: utf-8 -*-
import os

from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction

from .main_dialog import MainDialog


class SAMKitPlugin:
    """SAMKit -- Spatial Analysis and Mapping Toolkit."""

    def __init__(self, iface):
        self.iface = iface
        self.actions = []
        self.menu = "&SAMKit"
        self.dlg = None

    def initGui(self):
        icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
        self.action = QAction(QIcon(icon_path), "SAMKit (Spatial Analysis and Mapping Toolkit)",
                               self.iface.mainWindow())
        self.action.triggered.connect(self.run)
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToMenu(self.menu, self.action)
        self.actions.append(self.action)

    def unload(self):
        for action in self.actions:
            self.iface.removePluginMenu(self.menu, action)
            self.iface.removeToolBarIcon(action)
        if self.dlg is not None:
            self.dlg.close()
            self.dlg = None
        # Release layers held for map layouts before QGIS tears down, so they
        # are not destroyed after the application (crash on QGIS exit).
        try:
            from . import map_export
            map_export.release_resources()
        except Exception:
            pass

    def run(self):
        if self.dlg is None:
            self.dlg = MainDialog(self.iface, parent=self.iface.mainWindow())
        self.dlg.show()
        self.dlg.raise_()
        self.dlg.activateWindow()
