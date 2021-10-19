#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CALFEM VTK

Used to run VTK Visualization tool without a model

@author: Andreas Åmand
"""

import sys
import vis_vtk as cfvv
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = cfvv.MainWindow()
    ex.show()
    sys.exit(app.exec_())