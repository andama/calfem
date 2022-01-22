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
#from scipy.io import loadmat

edof,coord,dof,a,es,ns,lamb,eigen = cfvv.import_mat('exv4',['edof','coord','dof','a','es','ns','lambda','eigen'])

ex,ey,ez = cfc.coordxtr(edof,coord,dof)

#print(data)

#solid_data = loadmat('exv4.mat')

#edof = solid_data['edof']
#edof = np.delete(edof,0,1)
#coord = solid_data['coord']
#dof = solid_data['dof']
#a = solid_data['a']
#ed = solid_data['ed']
#es = solid_data['es']
#et = solid_data['et']
#eci = solid_data['eci']
#ns = solid_data['ns']
#L = solid_data['L']
#X = solid_data['X']





#print(L)

#print(es)

#print(edof)

ndof = np.size(dof, axis = 0)*np.size(dof, axis = 1)
ncoord = np.size(coord, axis = 0)
print('Number of CALFEM coordinates: ',ncoord)
nel = np.size(edof, axis = 0)



#inter = np.zeros((ncoord, 1))

#for i in range(ncoord):
	#inter[i,:] = X[3*i,0]*X[3*i,0] + X[3*i,1]*X[3*i,1] + X[3*i,2]*X[3*i,2]

mode_a = np.zeros((nel, 1))
tot_deform = np.zeros(8)
for i in range(nel):
	coords = cfvv.get_coord_from_edof(edof[i,:],dof,4)
	#print(coords)
	eigen[coords,0]
	for j in range(8):
		deform = cfvv.get_a_from_coord(coords[j],3,eigen[:,0])
		tot_deform[j] = np.sqrt(deform[0]**2 + deform[1]**2 + deform[2]**2)

	mode_a[i,:] = np.average(tot_deform)
	#print(x1)
	"""
	mode_a[i,:] = np.sqrt(
		X[coords[0],0]*X[coords[0],0] + 
		X[coords[1],0]*X[coords[1],0] + 
		X[coords[2],0]*X[coords[2],0] + 
		X[coords[3],0]*X[coords[3],0] + 
		X[coords[4],0]*X[coords[4],0] + 
		X[coords[5],0]*X[coords[5],0] + 
		X[coords[6],0]*X[coords[6],0] + 
		X[coords[7],0]*X[coords[7],0]
		)
	"""
#print(np.size(mode_a, axis = 0))
#print(np.size(mode_a, axis = 1))
#print(np.size(X, axis = 0))
#print(np.size(X, axis = 1))
#mode_a[:,0] = X[:,0]
#print(mode_a)

Freq=np.sqrt(lamb[0]/(2*np.pi))
#print(L)
#print(np.size(Freq, axis = 1))

#print(np.size(X[0,:], axis = 0))
#print(np.size(X[0,:], axis = 1))
#print(X[:,0])

#print(np.size(a, axis = 0))
#print(np.size(a, axis = 1))
#print(a)




stresses = np.zeros([nel,6])

#print(nel)

#print(np.average(es[0,:,0]))
#print(np.average(es[:,0,8]))

for i in range(0, nel):
	#print(i)
	stresses[i,0] = np.average(es[:,0,i])
	stresses[i,1] = np.average(es[:,1,i])
	stresses[i,2] = np.average(es[:,2,i])
	stresses[i,3] = np.average(es[:,3,i])
	stresses[i,4] = np.average(es[:,4,i])
	stresses[i,5] = np.average(es[:,5,i])

#print(stresses)

von_mises_elements = np.zeros([nel,1])

for i in range(0, nel):
	#print(i)
	von_mises_elements[i] = np.sqrt( 0.5 * ( np.square(stresses[i,0]-stresses[i,1]) + np.square(stresses[i,1]-stresses[i,2]) + np.square(stresses[i,2]-stresses[i,0]) ) + 3 * (np.square(stresses[i,3]) + np.square(stresses[i,4]) + np.square(stresses[i,5])) )

#print(von_mises)

von_mises_nodes = np.zeros([nel,8])



for i in range(0, nel):
	#print(i)
	von_mises_nodes[i,0] = np.sqrt( 0.5 * ( np.square(ns[0,0,i]-ns[0,1,i]) + np.square(ns[0,1,i]-ns[0,2,i]) + np.square(ns[0,2,i]-ns[0,0,i]) ) + 3 * (np.square(ns[0,3,i]) + np.square(ns[0,4,i]) + np.square(ns[0,5,i])) )
	von_mises_nodes[i,1] = np.sqrt( 0.5 * ( np.square(ns[1,0,i]-ns[1,1,i]) + np.square(ns[1,1,i]-ns[1,2,i]) + np.square(ns[1,2,i]-ns[1,0,i]) ) + 3 * (np.square(ns[1,3,i]) + np.square(ns[1,4,i]) + np.square(ns[1,5,i])) )
	von_mises_nodes[i,2] = np.sqrt( 0.5 * ( np.square(ns[2,0,i]-ns[2,1,i]) + np.square(ns[2,1,i]-ns[2,2,i]) + np.square(ns[2,2,i]-ns[2,0,i]) ) + 3 * (np.square(ns[2,3,i]) + np.square(ns[2,4,i]) + np.square(ns[2,5,i])) )
	von_mises_nodes[i,3] = np.sqrt( 0.5 * ( np.square(ns[3,0,i]-ns[3,1,i]) + np.square(ns[3,1,i]-ns[3,2,i]) + np.square(ns[3,2,i]-ns[3,0,i]) ) + 3 * (np.square(ns[3,3,i]) + np.square(ns[3,4,i]) + np.square(ns[3,5,i])) )
	von_mises_nodes[i,4] = np.sqrt( 0.5 * ( np.square(ns[4,0,i]-ns[4,1,i]) + np.square(ns[4,1,i]-ns[4,2,i]) + np.square(ns[4,2,i]-ns[4,0,i]) ) + 3 * (np.square(ns[4,3,i]) + np.square(ns[4,4,i]) + np.square(ns[4,5,i])) )
	von_mises_nodes[i,5] = np.sqrt( 0.5 * ( np.square(ns[5,0,i]-ns[5,1,i]) + np.square(ns[5,1,i]-ns[5,2,i]) + np.square(ns[5,2,i]-ns[5,0,i]) ) + 3 * (np.square(ns[5,3,i]) + np.square(ns[5,4,i]) + np.square(ns[5,5,i])) )
	von_mises_nodes[i,6] = np.sqrt( 0.5 * ( np.square(ns[6,0,i]-ns[6,1,i]) + np.square(ns[6,1,i]-ns[6,2,i]) + np.square(ns[6,2,i]-ns[6,0,i]) ) + 3 * (np.square(ns[6,3,i]) + np.square(ns[6,4,i]) + np.square(ns[6,5,i])) )
	von_mises_nodes[i,7] = np.sqrt( 0.5 * ( np.square(ns[7,0,i]-ns[7,1,i]) + np.square(ns[7,1,i]-ns[7,2,i]) + np.square(ns[7,2,i]-ns[7,0,i]) ) + 3 * (np.square(ns[7,3,i]) + np.square(ns[7,4,i]) + np.square(ns[7,5,i])) )





#print(von_mises_elements)
#print(von_mises_nodes)




### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###

# First plot, undeformed mesh
cfvv.figure(1)

cfvv.draw_mesh(edof,coord,dof,4,scale=0.005)
cfvv.add_text('Undeformed mesh')
#cfvv.add_rulers()


# Second plot, first mode from eigenvalue analysis
cfvv.figure(2)

scalefact = 100 #deformation scale factor
#scalefact = 1 #deformation scale factor
#cfvv.test(edof,ex,ey,ez,eigen[:,0],mode_a*1000,def_scale=scalefact)
cfvv.draw_displaced_mesh(edof,coord,dof,4,eigen[:,0],mode_a*1000,def_scale=scalefact)
cfvv.add_text('Eigenvalue analysis: first mode',pos='top-left')
cfvv.add_text(f'Frequency: {round(Freq[0],2)} Hz')
cfvv.add_text(f'Deformation scalefactor: {scalefact}',pos='top-right')
#cfvv.add_scalar_bar('Tot. el. displacement [mm]')
#cfvv.add_projection(plane='xz',offset=-1,rulers=True)



# Third plot, deformed mesh with element stresses
cfvv.figure(3)

scalefact = 3 #deformation scale factor
#mesh1 = cfvv.test(edof,ex,ey,ez,a,von_mises_elements/1000000,def_scale=scalefact)
mesh1 = cfvv.draw_displaced_mesh(edof,coord,dof,4,a,von_mises_elements/1000000,def_scale=scalefact)
cfvv.add_text('Static analysis: self-weight & ecc. vertical load',pos='top-left')
cfvv.add_text(f'Deformation scalefactor: {scalefact}',pos='top-right')
#cfvv.add_scalar_bar('von Mises [MPa]')


# Fourth plot, deformed mesh with nodal stresses
cfvv.figure(4)

# Return the mesh for export
#mesh2 = cfvv.test(edof,ex,ey,ez,a,von_mises_nodes/1000000,def_scale=scalefact,merge=True)
mesh2 = cfvv.draw_displaced_mesh(edof,coord,dof,4,a,von_mises_nodes/1000000,def_scale=scalefact,merge=True)
#cfvv.add_scalar_bar('von Mises [MPa]')
cfvv.add_text('Static analysis: self-weight & ecc. vertical load',pos='top-left')
cfvv.add_text(f'Deformation scalefactor: {scalefact}',pos='top-right')

# Export the mesh to 'exv4a.vtk'
cfvv.export_vtk('exv4a', mesh2)


#Start Calfem-vedo visualization
cfvv.show_and_wait()
