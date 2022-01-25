# -*- coding: utf-8 -*-
"""
Testing

@author: Andreas Åmand
"""

import os
import sys

os.system('clear')
sys.path.append("../")

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import calfem.core as cfc
import calfem.geometry as cfg
import numpy as np
#import vis_vedo as cfvv
import vis_vedo_no_qt as cfvv
from PyQt5 import Qt

# Balk
#coord = np.array([[0, 0, 0],[3, 0, 0]])
#dof = np.array([[1, 2, 3, 4, 5, 6],[7, 8, 9, 10, 11, 12]])
#edof = np.array([[1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12]])
#cfvv.draw_mesh(edof,coord,dof,5,color='black')
"""
# Solid
coord = np.array([[0, 0, 0],[1, 0, 0],[1, 1, 0],[0, 1, 0],[0, 0, 1],[1, 0, 1],[1, 1, 1],[0, 1, 1]])
dof = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9],[10, 11, 12],[13, 14, 15],[16, 17, 18],[19, 20, 21],[22, 23, 24]])
edof = np.array([[1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]])
a = np.zeros(8*3)
#print(a)
ex,ey,ez = cfc.coordxtr(edof,coord,dof)
cfvv.test(edof,ex,ey,ez,a,render_nodes=False)
"""


#import calfem.mesh as cfm
#import calfem.vis as cfv
#import calfem.core as cfc

#import vedo_utils as vdu
#from vedo import *


# ---- Define geometry ------------------------------------------------------

g = cfg.geometry()

g.point([0, 0, 0],      0)
g.point([1, 0, 0],      1)
g.point([0, 1, 0],      2)
g.point([0, 1, 1],      3, el_size=0.1)
g.point([0.5, -0.3, 0], 4)
g.point([-0.3, 0.5, 0], 5)
g.point([0.75, 0.75, 0],6)

g.spline([0,4,1])
g.spline([1,6,2])
g.spline([2,5,0])
g.spline([0,3])
g.spline([3,2])
g.spline([3,1])

g.ruledSurface([0,1,2])
g.ruledSurface([0,5,3])
g.ruledSurface([1,5,4])
g.ruledSurface([2,3,4])

g.volume([0,1,2,3])

points = g.getPointCoords()

cfvv.draw_geometry(points)


cfvv.show_and_wait()

