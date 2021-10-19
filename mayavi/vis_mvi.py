# -*- coding: utf-8 -*-
"""
CALFEM VTK

Contains all the functions for 3D visualization in CALFEM using VTK

@author: Andreas Åmand
"""
"""
import numpy as np
import vtk
import PyQt5
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
#from PyQt5 import uic
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.Qt import QApplication, QUrl, QDesktopServices
import webbrowser
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
"""
import numpy as np
import PyQt5
from mayavi import mlab

#mlab.test_contour3d()

class mvi:
    def show():
        return mlab.show()

class beam3d:
    def geometry(coord):
        nnodes = np.size(coord, axis = 0)
        nodes = []
        print(nnodes)
        for i in range(nnodes):
            nodes.append(mlab.points3d(coord[i,0],coord[i,1],coord[i,2]))
        print(nodes)
        return nodes
        #mlab.show()


def test_triangular_mesh():
    """An example of a cone, ie a non-regular mesh defined by its
        triangles.
    """
    n = 8
    t = np.linspace(-np.pi, np.pi, n)
    z = np.exp(1j * t)
    x = z.real.copy()
    y = z.imag.copy()
    z = np.zeros_like(x)

    triangles = [(0, i, i + 1) for i in range(1, n)]
    x = np.r_[0, x]
    y = np.r_[0, y]
    z = np.r_[1, z]
    t = np.r_[0, t]

    return mlab.triangular_mesh(x, y, z, triangles, scalars=t)


#test_triangular_mesh()
#mlab.show()