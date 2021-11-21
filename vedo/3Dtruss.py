#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D truss example using VTK to visualize

@author: Andreas Åmand
"""
import os
import sys
#import vtk
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
#from PyQt5 import uic
#from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import calfem.core as cfc
import numpy as np
#sys.path.append('../')
import vis_vedo as cfvv
from PyQt5 import Qt
#import core_beam_extensions as cfcb
#from ui_form import UI_Form

os.system('clear')

#from vtkmodules.vtkCommonDataModel import (
#    vtkCellArray,
#    vtkLine,
#    vtkPolyData
#)

        
"""
K = np.zeros([114,114])
f = np.zeros([114,1])
f[8,0] = -3000
f[14,0] = -3000
f[20,0] = -3000
f[26,0] = -3000
f[32,0] = -3000
f[50,0] = -3000
f[56,0] = -3000
f[62,0] = -3000
f[68,0] = -3000
f[74,0] = -3000
print(K)

edof_beams = np.array([
    # vänster sida längst ner
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 
    [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
    [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
    [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36],
    [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42],
    # höger sida längst ner
    [43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
    [55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66],
    [61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72],
    [67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78],
    [73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84],
    # botten mellan vänster & höger
    [7, 8, 9, 10, 11, 12, 49, 50, 51, 52, 53, 54], 
    [13, 14, 15, 16, 17, 18, 55, 56, 57, 58, 59, 60],
    [19, 20, 21, 22, 23, 24, 61, 62, 63, 64, 65, 66],
    [25, 26, 27, 28, 29, 30, 67, 68, 69, 70, 71, 72],
    [31, 32, 33, 34, 35, 36, 73, 74, 75, 76, 77, 78]
])

edof_bars = np.array([
    # vänster sida, vertikala/sneda
    [1, 2, 3, 85, 86, 87],
    [7, 8, 9, 85, 86, 87],
    [13, 14, 15, 85, 86, 87],
    [13, 14, 15, 88, 89, 90],
    [19, 20, 21, 88, 89, 90],
    [19, 20, 21, 91, 92, 93],
    [19, 20, 21, 94, 95, 96],
    [25, 26, 27, 94, 95, 96],
    [25, 26, 27, 97, 98, 99],
    [31, 32, 33, 97, 98, 99],
    [37, 38, 39, 97, 98, 99],
    # vänster sida, överst
    [85, 86, 87, 88, 89, 90],
    [88, 89, 90, 91, 92, 93],
    [91, 92, 93, 94, 95, 96],
    [94, 95, 96, 97, 98, 99],
    # höger sida, vertikala/sneda
    [43, 44, 45, 100, 101, 102],
    [49, 50, 51, 100, 101, 102],
    [55, 56, 57, 100, 101, 102],
    [55, 56, 57, 103, 104, 105],
    [61, 62, 63, 103, 104, 105],
    [61, 62, 63, 106, 107, 108],
    [61, 62, 63, 109, 110, 111],
    [67, 68, 69, 109, 110, 111],
    [67, 68, 69, 112, 113, 114],
    [73, 74, 75, 112, 113, 114],
    [79, 80, 81, 112, 113, 114],
    # höger sida, överst
    [100, 101, 102, 103, 104, 105],
    [103, 104, 105, 103, 104, 105],
    [106, 107, 108, 109, 110, 111],
    [109, 110, 111, 112, 113, 114],
    # överst mellan vänster & höger
    [85, 86, 87, 100, 101, 102],
    [88, 89, 90, 103, 104, 105],
    [91, 92, 93, 106, 107, 108],
    [94, 95, 96, 109, 110, 111],
    [97, 98, 99, 112, 113, 114]
])

E = 210000000
v = 0.3
G = E/(2*(1+v))
A_beam = 11250*0.000001
A_bar = np.pi*0.05*0.05
Iy = 63.1*0.000001
Iz = 182.6*0.000001
Kv = 0.856*0.000001



#ex_beams = np.zeros([17,2])
#ey_beams = np.zeros([17,2])
#ez_beams = np.zeros([17,2])

#ex_bars = np.zeros([35,2])
#ey_bars = np.zeros([35,2])
#ez_bars = np.zeros([35,2])


ex_beams = np.array([
    # vänster sida längst ner
    [0, 3],
    [3, 6],
    [6, 9],
    [9, 12],
    [12, 15],
    [15, 18],
    # höger sida längst ner
    [0, 3],
    [3, 6],
    [6, 9],
    [9, 12],
    [12, 15],
    [15, 18],
    # botten mellan vänster & höger
    [3, 3],
    [6, 6],
    [9, 9],
    [12, 12],
    [15, 15]
])
ey_beams = np.zeros([17,2])
ez_beams = np.array([
    # vänster sida längst ner
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    # höger sida längst ner
    [3, 3],
    [3, 3],
    [3, 3],
    [3, 3],
    [3, 3],
    [3, 3],
    # botten mellan vänster & höger
    [0, 3],
    [0, 3],
    [0, 3],
    [0, 3],
    [0, 3]
])

eo_beams = np.array([
    # vänster sida längst ner
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
    # höger sida längst ner
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
    # botten mellan vänster & höger
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1]
])

ex_bars = np.array([
    # vänster sida, vertikala/sneda
    [0, 3],
    [3, 3],
    [6, 3],
    [6, 6],
    [9, 6],
    [9, 9],
    [9, 12],
    [12, 12],
    [12, 15],
    [15, 15],
    [18, 15],
    # vänster sida, överst
    [3, 6],
    [6, 9],
    [9, 12],
    [12, 15],
    # höger sida, vertikala/sneda
    [0, 3],
    [3, 3],
    [6, 3],
    [6, 6],
    [9, 6],
    [9, 9],
    [9, 12],
    [12, 12],
    [12, 15],
    [15, 15],
    [18, 15],
    # höger sida, överst
    [3, 6],
    [6, 9],
    [9, 12],
    [12, 15],
    # överst mellan vänster & höger
    [3, 3],
    [6, 6],
    [9, 9],
    [12, 12],
    [15, 15]
])

ey_bars = np.array([
    # vänster sida, vertikala/sneda
    [0, 3],
    [0, 3],
    [0, 3],
    [0, 3],
    [0, 3],
    [0, 3],
    [0, 3],
    [0, 3],
    [0, 3],
    [0, 3],
    [0, 3],
    # vänster sida, överst
    [3, 3],
    [3, 3],
    [3, 3],
    [3, 3],
    # höger sida, vertikala/sneda
    [0, 3],
    [0, 3],
    [0, 3],
    [0, 3],
    [0, 3],
    [0, 3],
    [0, 3],
    [0, 3],
    [0, 3],
    [0, 3],
    [0, 3],
    # höger sida, överst
    [3, 3],
    [3, 3],
    [3, 3],
    [3, 3],
    # överst mellan vänster & höger
    [3, 3],
    [3, 3],
    [3, 3],
    [3, 3],
    [3, 3]
])

ez_bars = np.array([
    # vänster sida, vertikala/sneda
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    # vänster sida, överst
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    # höger sida, vertikala/sneda
    [3, 3],
    [3, 3],
    [3, 3],
    [3, 3],
    [3, 3],
    [3, 3],
    [3, 3],
    [3, 3],
    [3, 3],
    [3, 3],
    [3, 3],
    # höger sida, överst
    [3, 3],
    [3, 3],
    [3, 3],
    [3, 3],
    # överst mellan vänster & höger
    [0, 3],
    [0, 3],
    [0, 3],
    [0, 3],
    [0, 3]
])

ep_beams = [E, G, A_beam, Iy, Iz, Kv]
ep_bars = [E, A_bar]

#for i in range(8):
#    Ke = cfc.flw2qe(Ex[i],Ey[i],ep,D)
#    K = cfc.assem(Edof[i],K,Ke)
#Ke = np.zeros([12,12])
for i in range(17):
    Ke = cfc.beam3e(ex_beams[i], ey_beams[i], ez_beams[i], eo_beams[i], ep_beams)
    K = cfc.assem(edof_beams[i],K,Ke)

#Ke = np.zeros([6,6])
for i in range(35):
    Ke = cfc.bar3e(ex_bars[i], ey_bars[i], ez_bars[i], ep_bars)
    K = cfc.assem(edof_bars[i],K,Ke)

# ----- Solve equation system -----

bcPrescr = np.array([1,2,3,4,5,6,37,38,39,40,41,42,43,44,45,46,47,48,79,80,81,82,83,84])
#bcVal = np.array([0,0,0,0,0,0,0.5e-3,1e-3,1e-3])
a = cfc.solveq(K,f,bcPrescr)

print(a)

# ----- Compute element flux vector -----

#Ed = cfc.extractEldisp(Edof,a)
#Es = np.zeros((8,2))
#for i in range(8):
#    Es[i],Et = cfc.flw2qs(Ex[i],Ey[i],ep,D,Ed[i])
"""

"""
coord = np.array([
    [0, 0, 0],
    [3, 0, 0],
    [6, 0, 0],
    [9, 0, 0],
    #[3, 3, 0],
    #[6, 3, 0],
    #[9, 3, 0],
    #[0, 0, 3],
    #[3, 0, 3],
    #[6, 0, 3],
    #[9, 0, 3],
    #[3, 3, 3],
    #[6, 3, 3],
    #[9, 3, 3]
])

dof = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12],
    #[13, 14, 15],
    #[16, 17, 18],
    #[19, 20, 21],
    #[22, 23, 24],
    #[25, 26, 27],
    #[28, 29, 30],
    #[31, 32, 33],
    #[34, 35, 36],
    #[37, 38, 39],
    #[40, 41, 42]
])

edof = np.array([
    [1, 2, 3, 4, 5, 6],
    [4, 5, 6, 7, 8, 9],
    [7, 8, 9, 10, 11, 12],
    #[1, 2, 3, 13, 14, 15],
    #[4, 5, 6, 13, 14, 15],
    #[7, 8, 9, 13, 14, 15],
    #[7, 8, 9, 16, 17, 18],
    #[10, 11, 12, 16, 17, 18],
    #[10, 11, 12, 19, 20, 21],
    #[13, 14, 15, 16, 17, 18],
    #[16, 17, 18, 19, 20, 21],
    #[22, 23, 24, 25, 26, 27],
    #[25, 26, 27, 28, 29, 30],
    #[28, 29, 30, 31, 32, 33],
    #[22, 23, 24, 34, 35, 36],
    #[25, 26, 27, 34, 35, 36],
    #[28, 29, 30, 34, 35, 36],
    #[28, 29, 30, 37, 38, 39],
    #[31, 32, 33, 37, 38, 39],
    #[31, 32, 33, 40, 41, 42],
    #[34, 35, 36, 37, 38, 39],
    #[37, 38, 39, 40, 41, 42],
    #[1, 2, 3, 22, 23, 24],
    #[4, 5, 6, 25, 26, 27],
    #[7, 8, 9, 29, 30, 31],
    #[10, 11, 12, 31, 32, 33],
    #[13, 14, 15, 34, 35, 36],
    #[16, 17, 18, 37, 38, 39],
    #[19, 20, 21, 40, 41, 42]
])
"""
coord = np.array([
    [0, 0, 0],
    [3, 0, 0],
    [6, 0, 0],
    [9, 0, 0],
    [3, 3, 0],
    [6, 3, 0],
    [9, 3, 0],
    [0, 0, 3],
    [3, 0, 3],
    [6, 0, 3],
    [9, 0, 3],
    [3, 3, 3],
    [6, 3, 3],
    [9, 3, 3]
])

dof = np.array([
    [1, 2, 3, 4, 5, 6],
    [7, 8, 9, 10, 11, 12],
    [13, 14, 15, 16, 17, 18],
    [19, 20, 21, 22, 23, 24],
    [25, 26, 27, 28, 29, 30],
    [31, 32, 33, 34, 35, 36],
    [37, 38, 39, 40, 41, 42], ###
    [43, 44, 45, 46, 47, 48],
    [49, 50, 51, 52, 53, 54],
    [55, 56, 57, 58, 59, 60],
    [61, 62, 63, 64, 65, 66],
    [67, 68, 69, 70, 71, 72],
    [73, 74, 75, 76, 77, 78],
    [79, 80, 81, 82, 83, 84]
])

edof = np.array([
    [1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12],
    [7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18],
    [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
    [1,  2,  3,  4,  5,  6,  25, 26, 27, 28, 29, 30],
    [7,  8,  9,  10, 11, 12, 25, 26, 27, 28, 29, 30],
    [13, 14, 15, 16, 17, 18, 25, 26, 27, 28, 29, 30],
    [13, 14, 15, 16, 17, 18, 31, 32, 33, 34, 35, 36],
    [19, 20, 21, 22, 23, 24, 31, 32, 33, 34, 35, 36],
    [19, 20, 21, 22, 23, 24, 37, 38, 39, 40, 41, 42],
    [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36],
    [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42], ###
    [43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
    [55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66],
    [43, 44, 45, 46, 47, 48, 67, 68, 69, 70, 71, 72],
    [49, 50, 51, 52, 53, 54, 67, 68, 69, 70, 71, 72],
    [55, 56, 57, 58, 59, 60, 67, 68, 69, 70, 71, 72],
    [55, 56, 57, 58, 59, 60, 73, 74, 75, 76, 77, 78],
    [61, 62, 63, 64, 65, 66, 73, 74, 75, 76, 77, 78],
    [61, 62, 63, 64, 65, 66, 79, 80, 81, 82, 83, 84],
    [67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78],
    [73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84],
    [1,  2,  3,  4,  5,  6,  43, 44, 45, 46, 47, 48],
    [7,  8,  9,  10, 11, 12, 49, 50, 51, 52, 53, 54],
    [13, 14, 15, 16, 17, 18, 55, 56, 57, 58, 59, 60],
    [19, 20, 21, 22, 23, 24, 61, 62, 63, 64, 65, 66],
    [25, 26, 27, 28, 29, 30, 67, 68, 69, 70, 71, 72],
    [31, 32, 33, 34, 35, 36, 73, 74, 75, 76, 77, 78],
    [37, 38, 39, 40, 41, 42, 79, 80, 81, 82, 83, 84]
])

nnode = np.size(coord, axis = 0)
ndof = np.size(dof, axis = 0)*np.size(dof, axis = 1)
nel = np.size(edof, axis = 0)

ex,ey,ez = cfc.coordxtr(edof,coord,dof)
#print(nnode)
#print(ndof)
#print(nel)
#elements, nodes = cfvv.beam.set_geometry(edof,coord,dof)
nodes, elements = cfvv.beam3d.geometry(edof,coord,dof)

eo = np.array([0, 0, 1])

E = 210000000
v = 0.3
G = E/(2*(1+v))

#HEA300
A = 11250*0.000001 #m^2
Iy = 63.1*0.000001 #m^4
hy = 0.29*0.5 #m
Iz = 182.6*0.000001 #m^4
hz = 0.3*0.5 #m
Kv = 0.856*0.000001 #m^4

ep = [E, G, A, Iy, Iz, Kv]

K = np.zeros([ndof,ndof])

for i in range(nel):
    Ke = cfc.beam3e(ex[i], ey[i], ez[i], eo, ep)
    K = cfc.assem(edof[i],K,Ke)


f = np.zeros([ndof,1])
#f[20,0] = -3000
#f[38,0] = -3000
#f[62,0] = -3000
#f[80,0] = -3000

f[19,0] = -3000
f[37,0] = -3000
f[20,0] = 3000 # Punktlast i z-led
f[38,0] = -3000 # Punktlast i z-led
f[61,0] = -3000
f[79,0] = -3000

bcPrescr = np.array([1, 2, 3, 4, 5, 6, 19, 22, 23, 24, 37, 40, 41, 42, 43, 44, 45, 46, 47, 48, 61, 64, 65, 66, 79, 82, 83, 84])
#bcPrescr = np.array([0, 1, 2, 3, 4, 5, 18, 21, 22, 23, 36, 39, 40, 41, 42, 43, 44, 45, 46, 47, 60, 63, 64, 65, 78, 81, 82, 83])
a, r = cfc.solveq(K, f, bcPrescr)

#def_nodes, def_elements = cfvv.beam3d.def_geometry(edof,coord,dof,a,5)

ed = cfc.extractEldisp(edof,a)

#n=13 # 13 points in a 3m long beam = 250mm long segments
n=2

es = np.zeros((nel*n,6))
edi = np.zeros((nel*n,4))
eci = np.zeros((nel*n,1))

#es[0:n,:], edi[0:n,:], eci[0:n,:] = cfc.beam3s(ex[0],ey[0],ez[0],eo,ep,ed[0],[0,0,0,0],n)

for i in range(29):
    #ed[i] = cfc.extractEldisp(edof[i],a)
    es[n*i:n*i+n,:], edi[n*i:n*i+n,:], eci[n*i:n*i+n,:] = cfc.beam3s(ex[i],ey[i],ez[i],eo,ep,ed[i],[0,0,0,0],n)

#print(es)
#print(edi)
#print(eci)

def_nodes, def_elements = cfvv.beam3d.def_geometry(edof,coord,dof,a,5)

#normal_stresses = cfvv.beam3d.el_values(edof,es,edi,eci,E,v,A,Iy,Iz,hy,hz)

cfvv.beam3d.el_values(edof,es,edi,eci,E,v,A,Iy,Iz,hy,hz)

if __name__ == "__main__":
    app = Qt.QApplication(sys.argv)
    window = cfvv.MainWindow(nodes,elements,def_nodes,def_elements)
    #window = cfvv.MainWindow(nodes,elements,def_nodes,def_elements)
    #app.aboutToQuit.connect(cfvv.MainWindow.onClose) # <-- connect the onClose event
    app.exec_()


"""
if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = cfvv.MainWindow(elements,nodes)
    ex.show()
    sys.exit(app.exec_())
"""