# -*- coding: utf-8 -*-
"""
3D example using Vedo, bar & beam elements

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

### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###

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

edof_beams = np.array([
    [1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12],
    [7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18],
    [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
    [1,  2,  3,  4,  5,  6,  25, 26, 27, 28, 29, 30],
    [7,  8,  9,  10, 11, 12, 25, 26, 27, 28, 29, 30],
    [13, 14, 15, 16, 17, 18, 31, 32, 33, 34, 35, 36],
    [19, 20, 21, 22, 23, 24, 37, 38, 39, 40, 41, 42],
    [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36],
    [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42],

    [43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
    [55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66],
    [43, 44, 45, 46, 47, 48, 67, 68, 69, 70, 71, 72],
    [49, 50, 51, 52, 53, 54, 67, 68, 69, 70, 71, 72],
    [55, 56, 57, 58, 59, 60, 73, 74, 75, 76, 77, 78],
    [61, 62, 63, 64, 65, 66, 79, 80, 81, 82, 83, 84],
    [67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78],
    [73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84],

    [1,  2,  3,  4,  5,  6,  43, 44, 45, 46, 47, 48],
    [7,  8,  9,  10, 11, 12, 49, 50, 51, 52, 53, 54],
    [13, 14, 15, 16, 17, 18, 55, 56, 57, 58, 59, 60],
    [19, 20, 21, 22, 23, 24, 61, 62, 63, 64, 65, 66]
])

nnode = np.size(coord, axis = 0)
ndof = np.size(dof, axis = 0)*np.size(dof, axis = 1)
nel_beams = np.size(edof_beams, axis = 0)

ex_beams,ey_beams,ez_beams = cfc.coordxtr(edof_beams,coord,dof)

eo = np.array([
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],

    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],

    [-1, 0, 0],
    [-1, 0, 0],
    [-1, 0, 0],
    [-1, 0, 0]
    ])

E = 210000000               # Pa
v = 0.3
G = E/(2*(1+v))             # Pa

#HEA300-beams
A_beams = 11250*0.000001    # m^2
A_web_beams = 2227*0.000001 # m^2
Iy = 63.1*0.000001          # m^4
hy = 0.29*0.5               # m
Iz = 182.6*0.000001         # m^4
hz = 0.3*0.5                # m
Kv = 0.856*0.000001         # m^4

ep_beams = [E, G, A_beams, Iy, Iz, Kv]
"""
eq = np.zeros([nel_beams,4])
eq[18] = [0,-2000,0,0]
eq[19] = [0,-2000,0,0]
eq[20] = [0,-2000,0,0]
eq[21] = [0,-2000,0,0]
"""
K = np.zeros([ndof,ndof])
f = np.zeros([ndof,1])

eq = [0,-10000,0,0]

for i in range(nel_beams):
    print(i)
    """
    if i < 18:
        Ke = cfc.beam3e(ex_beams[i], ey_beams[i], ez_beams[i], eo, ep_beams)
        K = cfc.assem(edof_beams[i],K,Ke)
    else:
        Ke, fe = cfc.beam3e(ex_beams[i], ey_beams[i], ez_beams[i], eo, ep_beams, eq[i,:])
        K, f = cfc.assem(edof_beams[i],K,Ke,f,fe)
    #Ke = cfc.beam3e(ex_beams[i], ey_beams[i], ez_beams[i], eo, ep_beams)
    """
    if i < 18:
        Ke = cfc.beam3e(ex_beams[i], ey_beams[i], ez_beams[i], eo[i], ep_beams)
        K = cfc.assem(edof_beams[i],K,Ke)
    else:
        Ke, fe = cfc.beam3e(ex_beams[i], ey_beams[i], ez_beams[i], eo[i], ep_beams,eq)
        #print(fe)
        K, f = cfc.assem(edof_beams[i],K,Ke,f,fe)
# FRÅGA JONAS OM NEDAN
#f = -np.absolute(f)
#print(f)
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###

edof_bars = np.array([
    [13, 14, 15, 25, 26, 27],
    [19, 20, 21, 31, 32, 33],
    [55, 56, 57, 67, 68, 69],
    [61, 62, 63, 73, 74, 75],
    [25, 26, 27, 67, 68, 69],
    [31, 32, 33, 73, 74, 75],
    [37, 38, 39, 79, 80, 81]
])

nel_bars = np.size(edof_bars, axis = 0)

#ex_bars,ey_bars,ez_bars = cfc.coordxtr(edof_bars,coord,dof)

ex_bars = np.array([
    [6,3],
    [9,6],
    [6,3],
    [9,6],
    [3,3],
    [6,6],
    [9,9],
])

ey_bars = np.array([
    [0,3],
    [0,3],
    [0,3],
    [0,3],
    [3,3],
    [3,3],
    [3,3],
])

ez_bars = np.array([
    [0,0],
    [0,0],
    [3,3],
    [3,3],
    [0,3],
    [0,3],
    [0,3],
])

#Solid 100mm circular bars
A_bars = 0.05*np.pi*np.pi   # m^2

ep_bars = [E, A_bars]

for i in range(nel_bars):
    Ke = cfc.bar3e(ex_bars[i], ey_bars[i], ez_bars[i], ep_bars)
    K = cfc.assem(edof_bars[i],K,Ke)

### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###




# Used for point loading
#f[7,0] = -3000
#f[13,0] = -3000
#f[19,0] = -3000
#f[49,0] = -3000
#f[55,0] = -3000
#f[61,0] = -3000

#f[19,0] = -3000
#f[37,0] = -3000
#f[20,0] = 3000 # Punktlast i z-led
#f[38,0] = 500 # Punktlast i z-led
#f[61,0] = -3000
#f[79,0] = -3000

#bcPrescr = np.array([1, 2, 3, 4, 5, 6, 19, 22, 23, 24, 37, 40, 41, 42, 43, 44, 45, 46, 47, 48, 61, 64, 65, 66, 79, 82, 83, 84])
bcPrescr = np.array([1, 2, 3, 19, 37, 43, 44, 45, 61, 79])
a,r = cfc.solveq(K, f, bcPrescr)
print(a)

ed_beams = cfc.extractEldisp(edof_beams,a)

# Number of points along the beam
#nseg=2  # 2 points in a 3m long beam = 3000mm long segments
nseg=13 # 13 points in a 3m long beam = 250mm long segments


es_beams = np.zeros((nel_beams*nseg,6))
edi_beams = np.zeros((nel_beams*nseg,4))
eci_beams = np.zeros((nel_beams*nseg,1))

for i in range(nel_beams):
    #es_beams[nseg*i:nseg*i+nseg,:], edi_beams[nseg*i:nseg*i+nseg,:], eci_beams[nseg*i:nseg*i+nseg,:] = cfc.beam3s(ex_beams[i],ey_beams[i],ez_beams[i],eo,ep_beams,ed_beams[i],eq[i,:],nseg)
    #es[nseg*i:nseg*i+nseg,:], edi[nseg*i:nseg*i+nseg,:], eci[nseg*i:nseg*i+nseg,:] = cfc.beam3s(ex[i],ey[i],ez[i],eo,ep,ed[i],eq[i],nseg)
    
    if i < 18:
        es_beams[nseg*i:nseg*i+nseg,:], edi_beams[nseg*i:nseg*i+nseg,:], eci_beams[nseg*i:nseg*i+nseg,:] = cfc.beam3s(ex_beams[i],ey_beams[i],ez_beams[i],eo[i],ep_beams,ed_beams[i],[0,0,0,0],nseg)
    else:
        es_beams[nseg*i:nseg*i+nseg,:], edi_beams[nseg*i:nseg*i+nseg,:], eci_beams[nseg*i:nseg*i+nseg,:] = cfc.beam3s(ex_beams[i],ey_beams[i],ez_beams[i],eo[i],ep_beams,ed_beams[i],eq,nseg)
    #es[nseg*i:nseg*i+nseg,:] = cfc.beam3s(ex[i],ey[i],ez[i],eo,ep,ed[i],[0,0,0,0],nseg)

N_beams = es_beams[:,0]
Vy = es_beams[:,1]
Vz = es_beams[:,2]
T = es_beams[:,3]
My = es_beams[:,4]
Mz = es_beams[:,5]

ed_bars = cfc.extractEldisp(edof_bars,a)

es_bars = np.zeros((nel_bars,1))

for i in range(nel_bars):
    #es[nseg*i:nseg*i+nseg,:], edi[nseg*i:nseg*i+nseg,:], eci[nseg*i:nseg*i+nseg,:] = cfc.beam3s(ex[i],ey[i],ez[i],eo,ep,ed[i],eq[i],nseg)
    es_bars[i,:] = cfc.bar3s(ex_bars[i],ey_bars[i],ez_bars[i],ep_bars,ed_bars[i])
    #es[nseg*i:nseg*i+nseg,:] = cfc.beam3s(ex[i],ey[i],ez[i],eo,ep,ed[i],[0,0,0,0],nseg)

N_bars = es_bars

normal_stresses_beams = np.zeros(nel_beams*nseg)
shear_stresses_y = np.zeros(nel_beams*nseg)
shear_stresses_z = np.zeros(nel_beams*nseg)

#print(N)


# Stress calculation based on element forces
for i in range(nel_beams*nseg):
    # Calculate least favorable normal stress using Navier's formula
    #normal_stresses_beams[i] = N_beams[i]/A_beams + np.absolute(My[i]/Iy*hz) + np.absolute(Mz[i]/Iz*hy)
    
    if N_beams[i] < 0:
        normal_stresses_beams[i] = N_beams[i]/A_beams - np.absolute(My[i]/Iy*hz) - np.absolute(Mz[i]/Iz*hy)
        #normal_stresses[i] = N[i]/A - My[i]/Iy*hz - Mz[i]/Iz*hy
    else:
        normal_stresses_beams[i] = N_beams[i]/A_beams + np.absolute(My[i]/Iy*hz) + np.absolute(Mz[i]/Iz*hy)
        #normal_stresses[i] = N[i]/A + My[i]/Iy*hz + Mz[i]/Iz*hy
    
    # Calculate shear stress in y-direction (Assuming only web taking shear stresses)
    shear_stresses_y[i] = Vy[i]/A_web_beams

    # Calculate shear stress in y-direction (Assuming only flanges taking shear stresses)
    shear_stresses_z[i] = Vz[i]/(A_beams-A_web_beams)
#print(normal_stresses)

normal_stresses_bars = np.zeros(nel_bars)

for i in range(nel_bars):
    # Calculate least favorable normal stress
    normal_stresses_bars[i] = N_bars[i]/A_bars
    
# Below the data for the undeformed mesh is sent, along with element values.
# Normal stresses are sent by default, but comment it out and uncomment 
# shear_stresses_y/shear_stresses_z to visualize them

# Send data of deformed geometry & normal stresses as element values
#cfvv.beam3d.draw_displaced_geometry(edof,coord,dof,a,normal_stresses,'Max normal stress',def_scale=5,nseg=nseg)
cfvv.draw_mesh(edof_beams,coord,dof,5,alpha=0.5,nseg=nseg,color='green')


#cfvv.draw_displaced_geometry(edof,coord,dof,5,el_values=normal_stresses,label='Max normal stress',alpha=0.3,nseg=nseg)
cfvv.draw_displaced_mesh(edof_beams,coord,dof,5,a,normal_stresses_beams,nseg=nseg,def_scale=1,colormap='coolwarm')
cfvv.add_scalar_bar('Max normal stress beams')
#cfvv.add_legend(def_beam_elements)
cfvv.add_text('Beam',color='green',pos='top-left')
# Send data of deformed geometry & normal stresses as element values
#cfvv.draw_displaced_geometry(edof,coord,dof,a,shear_stresses_y,1,label='Shear stress y',def_scale=5,nseg=nseg)

# Send data of deformed geometry & normal stresses as element values
#cfvv.draw_displaced_geometry(edof,coord,dof,a,shear_stresses_z,1,label='Shear stress z',def_scale=5,nseg=nseg)


#cfvv.beam3d(edof,coord,dof,a,normal_stresses,'Max normal stress',nseg=nseg)

### IMPORTANT: Keep 6 dofs here, otherwise visualization breaks
### In reality, 3D-bars only have 6 dofs
edof_bars = np.array([
    [13, 14, 15, 16, 17, 18, 25, 26, 27, 28, 29, 30],#[13, 14, 15, 25, 26, 27],
    [19, 20, 21, 22, 23, 24, 31, 32, 33, 34, 35, 36],#[19, 20, 21, 31, 32, 33],
    [55, 56, 57, 58, 59, 60, 67, 68, 69, 70, 71, 72],#[55, 56, 57, 67, 68, 69],
    [61, 62, 63, 64, 65, 66, 73, 74, 75, 76, 77, 78],#[55, 56, 57, 73, 74, 75],
    [25, 26, 27, 28, 29, 30, 67, 68, 69, 70, 71, 72],#[25, 26, 27, 67, 68, 69],
    [31, 32, 33, 34, 35, 36, 73, 74, 75, 76, 77, 78],#[31, 32, 33, 73, 74, 75],
    [37, 38, 39, 40, 41, 42, 79, 80, 81, 82, 83, 84]#[37, 38, 39, 79, 80, 81]
])

cfvv.draw_mesh(edof_bars,coord,dof,2,alpha=0.5,color='yellow')
#print('bar disp')
cfvv.draw_displaced_mesh(edof_bars,coord,dof,2,a,normal_stresses_bars,def_scale=1,colormap='coolwarm')
cfvv.add_scalar_bar('Max normal stress bars',pos=[0.75,0.1])
#cfvv.add_legend(def_bar_elements)
cfvv.add_text('Bar',color='yellow',pos=[0,0.95])#,render=True)

#Start Calfem-vedo visualization
cfvv.show_and_wait()
