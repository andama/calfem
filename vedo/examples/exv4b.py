# -*- coding: utf-8 -*-
"""
3D example using Vedo, solid elements

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
from scipy.io import loadmat



solid_data = loadmat('3Dsolid.mat')

edof = solid_data['edof']
edof = np.delete(edof,0,1)
coord = solid_data['coord']
dof = solid_data['dof']
a = solid_data['a']
ed = solid_data['ed']
es = solid_data['es']
et = solid_data['et']
eci = solid_data['eci']
ns = solid_data['ns']
L = solid_data['L']
X = solid_data['X']


ndof = np.size(dof, axis = 0)*np.size(dof, axis = 1)
ncoord = np.size(coord, axis = 0)
nel = np.size(edof, axis = 0)

mode_a = np.zeros((nel, 1))
y = np.zeros(8)
for i in range(nel):
	coords = cfvv.tools.get_coord_from_edof(edof[i,:],dof,4)
	X[coords,0]
	for j in range(8):
		x = cfvv.tools.get_a_from_coord(coords[j],3,X[:,0])
		y[j] = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)

	mode_a[i,:] = np.average(y)


Freq=np.sqrt(L[0]/(2*np.pi))







### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###


# Second plot, first mode from eigenvalue analysis
#cfvv.add_text('Eigenvalue analysis: first mode',window=1)
scalefact = 100 #deformation scale factor
#mode_mesh = cfvv.draw_displaced_geometry(edof,coord,dof,4,X[:,0],mode_a,def_scale=scalefact,scale=0.002,render_nodes=False,merge=True)
cfvv.add_text(f'Frequency: {Freq[0]} Hz',pos='top-right')
cfvv.add_text(f'Deformation scalefactor: {scalefact}',pos='top-left')
#cfvv.add_scalar_bar(mode_mesh,'Tot. el. displacement',window=1)

cfvv.animate(edof,coord,dof,4,mode_a,10,def_scale=scalefact)






#Start Calfem-vedo visualization
cfvv.render()
#cfvv.show_and_wait()

