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
coord = np.array([
    #Botten
    [0,     0,     0], #DOF 1-3
    [0.2,   0,     0], #DOF 4-6
    [0.4,   0,     0], #DOF ?-?
    [0,     0,     0.2], #DOF 1-3
    [0.2,   0,     0.2],  #DOF 1-3
    [0.4,   0,     0.2],
    [0,     0,     0.4],
    [0.2,   0,     0.4],
    [0.4,   0,     0.4],
    #Mitten
    [0,     0.2,   0],
    [0.2,   0.2,   0],
    [0.4,   0.2,   0],
    [0,     0.2,   0.2],
    [0.2,   0.2,   0.2],
    [0.4,   0.2,   0.2],
    [0,     0.2,   0.4],
    [0.2,   0.2,   0.4],
    [0.4,   0.2,   0.4],
    #Överst
    [0,     0.4,   0],
    [0.2,   0.4,   0],
    [0.4,   0.4,   0],
    [0,     0.4,   0.2],
    [0.2,   0.4,   0.2],
    [0.4,   0.4,   0.2],
    [0,     0.4,   0.4],
    [0.2,   0.4,   0.4],
    [0.4,   0.4,   0.4]
])

dof = np.array([
    #Botten
    [1,  2,  3],
    [4,  5,  6],
    [7,  8,  9],
    [10, 11, 12],
    [13, 14, 15],
    [16, 17, 18],
    [19, 20, 21],
    [22, 23, 24],
    [25, 26, 27],
    #Mitten
    [28, 29, 30],
    [31, 32, 33],
    [34, 35, 36],
    [37, 38, 39],
    [40, 41, 42],
    [43, 44, 45],
    [46, 47, 48],
    [49, 50, 51],
    [52, 53, 54],
    #Överst
    [55, 56, 57],
    [58, 59, 60],
    [61, 62, 63],
    [64, 65, 66],
    [67, 68, 69],
    [70, 71, 72],
    [73, 74, 75],
    [76, 77, 78],
    [79, 80, 81]
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
"""

coord = np.array([
    [0,     0,     0],
    [0.2,   0,     0],
    [0.2,   0.2,     0],
    [0,   0.2,     0],
    [0,   0,     0.2],
    [0.2,   0,     0.2],
    [0.2,   0.2,     0.2],
    [0,   0.2,     0.2]
])

dof = np.array([
    [1,  2,  3],
    [4,  5,  6],
    [7,  8,  9],
    [10, 11, 12],
    [13, 14, 15],
    [16, 17, 18],
    [19, 20, 21],
    [22, 23, 24]
])

edof = np.array([
    [1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
])


nnode = np.size(coord, axis = 0)
ndof = np.size(dof, axis = 0)*np.size(dof, axis = 1)
nel = np.size(edof, axis = 0)

ex,ey,ez = cfc.coordxtr(edof,coord,dof)

# Send data of undeformed geometry
#cfvv.solid3d.draw_geometry(edof,coord,dof,0.02,1)
cfvv.draw_geometry(edof,coord,dof,0.02,1,export=False)

ep = [2]

E = 210000000
v = 0.3
D = cfc.hooke(4,E,v)

K = np.zeros([ndof,ndof])
#Ke = cfc.flw3i8e(ex, ey, ez, ep, D)
#K = cfc.assem(edof,K,Ke)
#f = np.zeros([ndof,1])
#f[7,0] = -3000

#bcPrescr = np.array([1,2,3,4,5,6,13,14,15,16,17,18])
#a,r = cfc.solveq(K, f, bcPrescr)

#cfvv.beam3d.draw_displaced_geometry(edof,coord,dof,a,def_scale=5)

"""
# Send data of undeformed geometry
cfvv.beam3d.geometry(edof,coord,dof,0.02,0.2)

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
cfvv.beam3d.def_geometry(edof,coord,dof,a,normal_stresses,'Max normal stress',def_scale=5,nseg=nseg)

# Send data of deformed geometry & normal stresses as element values
#cfvv.beam3d.def_geometry(edof,coord,dof,a,shear_stresses_y,'Shear stress y',def_scale=5,nseg=nseg)

# Send data of deformed geometry & normal stresses as element values
#cfvv.beam3d.def_geometry(edof,coord,dof,a,shear_stresses_z,'Shear stress z',def_scale=5,nseg=nseg)

"""
#Start Calfem-vedo visualization
cfvv.show_and_wait()