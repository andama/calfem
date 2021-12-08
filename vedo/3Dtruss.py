#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D truss example using VTK to visualize

@author: Andreas Åmand
"""

import os

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import calfem.core as cfc
import numpy as np
import vis_vedo as cfvv
from PyQt5 import Qt

os.system('clear')


        
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
    [37, 38, 39, 40, 41, 42],
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
    [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42], 
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

# Send data of undeformed geometry
#beamdraw_geometry = cfvv.beam3d()
cfvv.beam3d.draw_geometry(edof,coord,dof,0.02,0.2)

eo = np.array([0, 0, 1])

E = 210000000
v = 0.3
G = E/(2*(1+v))

#HEA300-beams
A = 11250*0.000001      # m^2
A_web = 2227*0.000001   # m^2
Iy = 63.1*0.000001      # m^4
hy = 0.29*0.5           # m
Iz = 182.6*0.000001     # m^4
hz = 0.3*0.5            # m
Kv = 0.856*0.000001     # m^4

ep = [E, G, A, Iy, Iz, Kv]

#eq = np.zeros([nel,4])
#eq[23] = [0,-2000,0,0]
#eq[24] = [0,-2000,0,0]
#eq[25] = [0,-2000,0,0]

K = np.zeros([ndof,ndof])
f = np.zeros([ndof,1])

for i in range(nel):
    #Ke,fe = cfc.beam3e(ex[i], ey[i], ez[i], eo, ep, eq[i])
    Ke = cfc.beam3e(ex[i], ey[i], ez[i], eo, ep)
    
    K = cfc.assem(edof[i],K,Ke)
    #f = cfc.assem(edof[i],f,fe)




f[7,0] = -3000
f[13,0] = -3000
f[19,0] = -3000
f[49,0] = -3000
f[55,0] = -3000
f[61,0] = -3000

#f[19,0] = -3000
#f[37,0] = -3000
#f[20,0] = 3000 # Punktlast i z-led
#f[38,0] = 500 # Punktlast i z-led
#f[61,0] = -3000
#f[79,0] = -3000

bcPrescr = np.array([1, 2, 3, 4, 5, 6, 19, 22, 23, 24, 37, 40, 41, 42, 43, 44, 45, 46, 47, 48, 61, 64, 65, 66, 79, 82, 83, 84])
a,r = cfc.solveq(K, f, bcPrescr)

ed = cfc.extractEldisp(edof,a)

# Number of points along the beam
nseg=13 # 13 points in a 3m long beam = 250mm long segments


es = np.zeros((nel*nseg,6))
edi = np.zeros((nel*nseg,4))
eci = np.zeros((nel*nseg,1))

for i in range(29):
    #es[nseg*i:nseg*i+nseg,:], edi[nseg*i:nseg*i+nseg,:], eci[nseg*i:nseg*i+nseg,:] = cfc.beam3s(ex[i],ey[i],ez[i],eo,ep,ed[i],eq[i],nseg)
    es[nseg*i:nseg*i+nseg,:], edi[nseg*i:nseg*i+nseg,:], eci[nseg*i:nseg*i+nseg,:] = cfc.beam3s(ex[i],ey[i],ez[i],eo,ep,ed[i],[0,0,0,0],nseg)
    #es[nseg*i:nseg*i+nseg,:] = cfc.beam3s(ex[i],ey[i],ez[i],eo,ep,ed[i],[0,0,0,0],nseg)

N = es[:,0]
Vy = es[:,1]
Vz = es[:,2]
T = es[:,3]
My = es[:,4]
Mz = es[:,5]

normal_stresses = np.zeros((nel*nseg,1))
shear_stresses_y = np.zeros((nel*nseg,1))
shear_stresses_z = np.zeros((nel*nseg,1))

# Stress calculation based on element forces
for i in range(nel*nseg):
    # Calculate least favorable normal stress using Navier's formula
    if N[i] < 0:
        normal_stresses[i] = N[i]/A - np.absolute(My[i]/Iy*hz) - np.absolute(Mz[i]/Iz*hy)
        #normal_stresses[i] = N[i]/A - My[i]/Iy*hz - Mz[i]/Iz*hy
    else:
        normal_stresses[i] = N[i]/A + np.absolute(My[i]/Iy*hz) + np.absolute(Mz[i]/Iz*hy)
        #normal_stresses[i] = N[i]/A + My[i]/Iy*hz + Mz[i]/Iz*hy

    # Calculate shear stress in y-direction (Assuming only web taking shear stresses)
    shear_stresses_y[i] = Vy[i]/A_web

    # Calculate shear stress in y-direction (Assuming only flanges taking shear stresses)
    shear_stresses_z[i] = Vz[i]/(A-A_web)

    
# Below the data for the undeformed mesh is sent, along with element values.
# Normal stresses are sent by default, but comment it out and uncomment 
# shear_stresses_y/shear_stresses_z to visualize them

# Send data of deformed geometry & normal stresses as element values
#cfvv.beam3d.draw_displaced_geometry(edof,coord,dof,a,normal_stresses,'Max normal stress',def_scale=5,nseg=nseg)

# Send data of deformed geometry & normal stresses as element values
#cfvv.beam3d.draw_displaced_geometry(edof,coord,dof,a,shear_stresses_y,'Shear stress y',def_scale=5,nseg=nseg)

# Send data of deformed geometry & normal stresses as element values
#cfvv.beam3d.draw_displaced_geometry(edof,coord,dof,a,shear_stresses_z,'Shear stress z',def_scale=5,nseg=nseg)


#cfvv.beam3d(edof,coord,dof,a,normal_stresses,'Max normal stress',nseg=nseg)

#Start Calfem-vedo visualization
cfvv.show_and_wait()