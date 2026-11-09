# -*- coding: utf-8 -*-
"""
map_tools.py
Small, self-contained interactive drawing map tools:

* RectangleMapTool -- click-drag-release a rectangle (used by the DEM tab to
  define the area of interest).
* PolygonMapTool   -- click vertices to draw a free-form polygon (used by the
  Spatial Predictive Modeling tab to draw the spatial-block calibration
  region directly on the QGIS canvas, as an alternative to picking an
  already-digitized polygon layer).

Both follow the standard QGIS QgsMapTool + QgsRubberBand pattern.
"""

from qgis.PyQt.QtCore import pyqtSignal, Qt
from qgis.PyQt.QtGui import QColor
from qgis.gui import QgsMapTool, QgsRubberBand
from qgis.core import QgsWkbTypes, QgsRectangle, QgsGeometry, QgsPointXY


class RectangleMapTool(QgsMapTool):
    """Click, drag, and release on the canvas to draw a rectangle.
    Emits rectangleFinished(QgsRectangle) in the canvas's own CRS when done.
    Right-click or Esc cancels the current drag without emitting anything."""

    rectangleFinished = pyqtSignal(object)

    def __init__(self, canvas):
        super().__init__(canvas)
        self.canvas = canvas
        self.rubber_band = QgsRubberBand(canvas, QgsWkbTypes.PolygonGeometry)
        self.rubber_band.setColor(QColor(255, 0, 0, 60))
        self.rubber_band.setStrokeColor(QColor(255, 0, 0, 200))
        self.rubber_band.setWidth(2)
        self.start_point = None
        self.dragging = False
        self.setCursor(Qt.CrossCursor)

    def _set_rubber_band(self, p1, p2):
        rect = QgsRectangle(p1, p2)
        geom = QgsGeometry.fromRect(rect)
        self.rubber_band.setToGeometry(geom, None)

    def canvasPressEvent(self, event):
        if event.button() == Qt.RightButton:
            self._cancel()
            return
        self.start_point = self.toMapCoordinates(event.pos())
        self.dragging = True
        self.rubber_band.reset(QgsWkbTypes.PolygonGeometry)

    def canvasMoveEvent(self, event):
        if not self.dragging or self.start_point is None:
            return
        current_point = self.toMapCoordinates(event.pos())
        self._set_rubber_band(self.start_point, current_point)

    def canvasReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self._cancel()
            return
        if not self.dragging or self.start_point is None:
            return
        end_point = self.toMapCoordinates(event.pos())
        self.dragging = False
        rect = QgsRectangle(self.start_point, end_point)
        self.rubber_band.reset(QgsWkbTypes.PolygonGeometry)
        self.start_point = None
        if rect.width() <= 0 or rect.height() <= 0:
            return  # degenerate (a click with no drag) -- ignore
        self.rectangleFinished.emit(rect)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self._cancel()

    def _cancel(self):
        self.dragging = False
        self.start_point = None
        self.rubber_band.reset(QgsWkbTypes.PolygonGeometry)

    def deactivate(self):
        self.rubber_band.reset(QgsWkbTypes.PolygonGeometry)
        super().deactivate()


class PolygonMapTool(QgsMapTool):
    """Draw a free-form polygon on the canvas, vertex by vertex.

    Controls
      * Left-click            add a vertex
      * Double-click / Enter  finish (needs at least 3 vertices)
      * Right-click           finish (or cancel if fewer than 3 vertices)
      * Backspace / Delete    remove the last vertex
      * Esc                   cancel

    Emits polygonFinished(QgsGeometry) in the canvas's own CRS when done, or
    drawingCanceled() when cancelled. Self-intersecting outlines are repaired
    with makeValid() before emitting.
    """

    polygonFinished = pyqtSignal(object)
    drawingCanceled = pyqtSignal()

    def __init__(self, canvas):
        super().__init__(canvas)
        self.canvas = canvas
        self.rubber_band = QgsRubberBand(canvas, QgsWkbTypes.PolygonGeometry)
        self.rubber_band.setColor(QColor(255, 0, 0, 60))
        self.rubber_band.setStrokeColor(QColor(255, 0, 0, 200))
        self.rubber_band.setWidth(2)
        # vertex markers
        self.vertex_band = QgsRubberBand(canvas, QgsWkbTypes.PointGeometry)
        self.vertex_band.setColor(QColor(255, 0, 0, 220))
        self.vertex_band.setIconSize(7)
        icon = getattr(QgsRubberBand, "ICON_CIRCLE", None)
        if icon is None and hasattr(QgsRubberBand, "IconType"):
            icon = QgsRubberBand.IconType.ICON_CIRCLE
        if icon is not None:
            self.vertex_band.setIcon(icon)
        self.points = []
        self._finished = False
        self.setCursor(Qt.CrossCursor)

    # -- drawing helpers -------------------------------------------------
    def _redraw(self, cursor_point=None):
        self.rubber_band.reset(QgsWkbTypes.PolygonGeometry)
        self.vertex_band.reset(QgsWkbTypes.PointGeometry)
        pts = list(self.points)
        if cursor_point is not None:
            pts.append(cursor_point)
        for i, p in enumerate(pts):
            self.rubber_band.addPoint(p, i == len(pts) - 1)
        for i, p in enumerate(self.points):
            self.vertex_band.addPoint(p, i == len(self.points) - 1)

    def _same_as_last(self, pt):
        """True if pt is within ~3 screen pixels of the last vertex."""
        if not self.points:
            return False
        tol = self.canvas.mapUnitsPerPixel() * 3
        last = self.points[-1]
        return abs(last.x() - pt.x()) <= tol and abs(last.y() - pt.y()) <= tol

    # -- mouse -----------------------------------------------------------
    def canvasReleaseEvent(self, event):
        if self._finished:
            return
        if event.button() == Qt.RightButton:
            if len(self.points) >= 3:
                self._finish()
            else:
                self._cancel()
            return
        if event.button() != Qt.LeftButton:
            return
        pt = QgsPointXY(self.toMapCoordinates(event.pos()))
        if not self._same_as_last(pt):
            self.points.append(pt)
        self._redraw(pt)

    def canvasDoubleClickEvent(self, event):
        if self._finished or event.button() != Qt.LeftButton:
            return
        # The first click of the double-click already added this vertex.
        if len(self.points) >= 3:
            self._finish()

    def canvasMoveEvent(self, event):
        if self._finished or not self.points:
            return
        self._redraw(QgsPointXY(self.toMapCoordinates(event.pos())))

    # -- keyboard --------------------------------------------------------
    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Escape:
            self._cancel()
        elif key in (Qt.Key_Backspace, Qt.Key_Delete):
            if self.points:
                self.points.pop()
                self._redraw()
            event.accept()
        elif key in (Qt.Key_Return, Qt.Key_Enter):
            if len(self.points) >= 3:
                self._finish()
            event.accept()

    # -- finish / cancel -------------------------------------------------
    def _finish(self):
        ring = list(self.points)
        ring.append(ring[0])  # close the ring explicitly
        geom = QgsGeometry.fromPolygonXY([ring])
        if not geom.isGeosValid():
            fixed = geom.makeValid()
            if fixed is not None and not fixed.isEmpty():
                geom = fixed
            # makeValid can return a GeometryCollection mixing polygons and
            # stray lines/points -- keep only the polygon parts.
            if geom.type() != QgsWkbTypes.PolygonGeometry:
                parts = [g for g in geom.asGeometryCollection()
                         if g.type() == QgsWkbTypes.PolygonGeometry]
                geom = QgsGeometry.unaryUnion(parts) if parts else QgsGeometry()
        self._finished = True
        self._reset_bands()
        self.points = []
        if geom is None or geom.isEmpty() or geom.area() <= 0:
            self.drawingCanceled.emit()
            return
        self.polygonFinished.emit(geom)

    def _cancel(self):
        self._finished = True
        self.points = []
        self._reset_bands()
        self.drawingCanceled.emit()

    def _reset_bands(self):
        self.rubber_band.reset(QgsWkbTypes.PolygonGeometry)
        self.vertex_band.reset(QgsWkbTypes.PointGeometry)

    def deactivate(self):
        self._reset_bands()
        super().deactivate()
