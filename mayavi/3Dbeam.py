#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D beam example using VTK to visualize

@author: Andreas Åmand
"""

#import sys
#import vtk
#from PyQt5.QtWidgets import *
#from PyQt5.QtCore import *
#from PyQt5 import uic
#from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import calfem.core as cfc
import numpy as np
#from numpy import *
#sys.path.append('../')
import vis_mvi as cfvm
#import core_beam_extensions as cfcb
import PyQt5
from mayavi import mlab

#import model


#def model():
K = np.zeros([18,18])
f = np.zeros([18,1])
f[7,0] = -3000
coord = np.array([
    [3, 1, 0],
    [1.5, 1, 2],
    [0, 1, 4]
])
dof = np.array([
    [ 1, 2, 3, 4, 5, 6],
    [ 7, 8, 9,10,11,12],
    [13,14,15,16,17,18]
])
edof = np.array([
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
])
ex,ey,ez = cfc.coordxtr(edof,coord,dof)
#elements, nodes = cfvv.beam.set_geometry(edof,coord,dof)
#nodes = cfvm.beam3d.geometry(coord)

eo = np.array([-3, 4, 0])

E = 210000000
v = 0.3
G = E/(2*(1+v))
A = 11250*0.000001
Iy = 63.1*0.000001
Iz = 182.6*0.000001
Kv = 0.856*0.000001

ep = [E, G, A, Iy, Iz, Kv]

for i in range(2):
    Ke = cfc.beam3e(ex[i], ey[i], ez[i], eo, ep)
    K = cfc.assem(edof[i],K,Ke)

bcPrescr = np.array([1, 2, 3, 4, 7, 9, 13, 14, 15, 16])
a, r = cfc.solveq(K, f, bcPrescr)

ed = cfc.extractEldisp(edof,a)

es = np.zeros((4,6))
for i in range(2):
    #ed[i] = cfc.extractEldisp(edof[i],a)
    es[2*i:2*i+2,0:6] = cfc.beam3s(ex[i],ey[i],ez[i],eo,ep,ed[i])

#    return coord, dof, edof

#t=np.linspace(0,2*np.pi,50)
#u=np.cos(t)*np.pi

#x,y,z = np.sin(u), np.cos(u), np.sin(t)

#mlab.points3d(x,y,z)

if __name__ == "__main__":
    cfvm.test_triangular_mesh()
    cfvm.mvi.show()
    #mlab.show()
    #app = QApplication(sys.argv)
    #ex = cfvv.MainWindow(elements,nodes)
    #ex.show()
    #sys.exit(app.exec_())
    #mlab.points3d(x,y,z)
    #mlab.show()


