# -*- coding: utf-8 -*-
"""
Example using Vedo, spring elements

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

coord = np.array([
    [0,0,0],
    [0.5,0,0],
    [1,0,0],
    [1.5,0,0]
])

dof = np.array([
    [1],
    [2],
    [3],
    [4]
])

edof = np.array([
    [1, 2],
    [2, 3],
    [3, 4]
])

#nnode = np.size(coord, axis = 0)
ndof = np.size(dof, axis = 0)*np.size(dof, axis = 1)
nel = np.size(edof, axis = 0)

#ex,ey,ez = cfc.coordxtr(edof,coord,dof)

k = 1000

ep = [3*k, k, 8*k]

K = np.zeros([ndof,ndof])
f = np.zeros([ndof,1])

for i in range(nel):
    Ke = cfc.spring1e(ep[i])
    K = cfc.assem(edof[i],K,Ke)

f[3,0] = 500 #Newton

bcPrescr = np.array([1])

a,r = cfc.solveq(K, f, bcPrescr)

cfvv.draw_mesh(edof,coord,dof,1)
cfvv.draw_displaced_mesh(edof,coord,dof,1,a,offset=[0,0.2,0],render_nodes=True)
cfvv.add_text_3D('k=3 kN/m',[0.15,-0.1,0],size=0.03)
cfvv.add_text_3D('k=1 kN/m',[0.65,-0.1,0],size=0.03)
cfvv.add_text_3D('k=8 kN/m',[1.15,-0.1,0],size=0.03)
cfvv.add_text_3D('F_x =500 N',[1.55,-0.02,0],size=0.03)
#cfvv.add_rulers()
#cfvv.draw_displaced_geometry(edof,coord,dof,5,a,scale=0.5,alpha=0.5)

#Start Calfem-vedo visualization
cfvv.show_and_wait()
