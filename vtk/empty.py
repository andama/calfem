#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 14:08:07 2021

@author: Andreas Åmand
"""

import sys
import vis_vtk as cfvv
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

if __name__ == "__main__":
    #beam = beam
    app = QApplication(sys.argv)
    #ex = MainWindow().__init__(beam)
    #MainWindow.vtk_actor(beam.coord)
    ex = cfvv.MainWindow()
    ex.show()
    sys.exit(app.exec_())