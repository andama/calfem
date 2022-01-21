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
import numpy as np
#import vis_vedo as cfvv
import vis_vedo_no_qt as cfvv
from PyQt5 import Qt

# Balk
#coord = np.array([[0, 0, 0],[3, 0, 0]])
#dof = np.array([[1, 2, 3, 4, 5, 6],[7, 8, 9, 10, 11, 12]])
#edof = np.array([[1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12]])
#cfvv.draw_mesh(edof,coord,dof,5,color='black')

# Solid
coord = np.array([[0, 0, 0],[1, 0, 0],[1, 1, 0],[0, 1, 0],[0, 0, 1],[1, 0, 1],[1, 1, 1],[0, 1, 1]])
dof = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9],[10, 11, 12],[13, 14, 15],[16, 17, 18],[19, 20, 21],[22, 23, 24]])
edof = np.array([[1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]])
a = np.zeros(8*3)
#print(a)
ex,ey,ez = cfc.coordxtr(edof,coord,dof)
cfvv.test(edof,ex,ey,ez,7,a,render_nodes=False)

cfvv.show_and_wait()