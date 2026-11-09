# -*- coding: utf-8 -*-
"""
SAMKit -- Spatial Analysis and Mapping Toolkit
QGIS plugin entry point.

Author: Samira Safaee, Ph.D. (samira7v@gmail.com)
Repository: https://github.com/ssafaee/SamKit
Licence: GNU GPL v2 or later (see LICENSE); third-party provenance in NOTICE.
"""

__author__ = "Samira Safaee, Ph.D."
__email__ = "samira7v@gmail.com"


def _activate_bundled_libs():
    """Put SAMKit's private 'libs' folder on sys.path.

    Optional dependencies (scikit-learn, pykrige) are installed there rather
    than into QGIS's own site-packages, so QGIS is never modified. The folder
    is APPENDED, not prepended, so QGIS's own numpy/scipy always win if a
    copy ever ends up in libs/.
    """
    import os
    import sys
    libs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libs")
    if os.path.isdir(libs_dir) and libs_dir not in sys.path:
        sys.path.append(libs_dir)


_activate_bundled_libs()


def classFactory(iface):
    from .samkit_plugin import SAMKitPlugin
    return SAMKitPlugin(iface)
