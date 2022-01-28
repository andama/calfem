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
import vedo_utils as cfvu
#from scipy.io import loadmat

#edof,coord,dof,a,es,ns,L,X = cfvv.import_mat('exv4',['edof','coord','dof','a','es','ns','L','X'])
edof,coord,dof,a,vM_el,vM_n,lamb,eig = cfvv.import_mat('exv4',['edof','coord','dof','a','vM_el','vM_n','lambda','eig'])
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


ndof = np.size(dof, axis = 0)*np.size(dof, axis = 1)
ncoord = np.size(coord, axis = 0)
nel = np.size(edof, axis = 0)

mode_a = np.zeros((nel, 1))
y = np.zeros(8)
for i in range(nel):
	coords = cfvu.get_coord_from_edof(edof[i,:],dof,4)
	#eig[coords,0]
	for j in range(8):
		x = cfvu.get_a_from_coord(coords[j],3,eig[:,0])
		y[j] = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)

	mode_a[i,:] = np.average(y)


Freq=np.sqrt(lamb[0]/(2*np.pi))







### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###


# Second plot, first mode from eigenvalue analysis
#cfvv.add_text('Eigenvalue analysis: first mode',window=1)
scalefact = 100 #deformation scale factor
#mode_mesh = cfvv.draw_displaced_geometry(edof,coord,dof,4,X[:,0],mode_a,def_scale=scalefact,scale=0.002,render_nodes=False,merge=True)
cfvv.add_text(f'Frequency: {Freq[0]} Hz',pos='top-right')
cfvv.add_text(f'Deformation scalefactor: {scalefact}',pos='top-left')
#cfvv.add_scalar_bar(mode_mesh,'Tot. el. displacement',window=1)

cfvv.animate(edof,coord,dof,4,eig[:,0],10,mode_a*1000,def_scale=scalefact,export=True,file='anim/exv4b')
#cfvv.animate(edof,coord,dof,4,eig[:,0],10,def_scale=scalefact,export=True)


'''
        #print(a)

        #ncoord = np.size(coord, axis = 0)
        #nnode = np.size(coord, axis = 0)

        ex,ey,ez = cfc.coordxtr(edof,coord,dof)

        ed = cfc.extractEldisp(edof,a)
        if element_type == 3:
            if val != 'nodal_values_by_el':
                coord2, topo, node_dofs, a_node, node_scalars = vdu.convert_to_node_topo(edof,ex,ey,ez,ed,ignore_first=False,dofs_per_node=1)
            else:
                coord2, topo, node_dofs, a_node, node_scalars = vdu.convert_to_node_topo(edof,ex,ey,ez,ed,values,ignore_first=False,dofs_per_node=1)
        elif element_type == 4:
            if val != 'nodal_values_by_el':
                coord2, topo, node_dofs, a_node, node_scalars = vdu.convert_to_node_topo(edof,ex,ey,ez,ed,ignore_first=False)
            else:
                coord2, topo, node_dofs, a_node, node_scalars = vdu.convert_to_node_topo(edof,ex,ey,ez,ed,values,ignore_first=False)
        #a_node = vdu.convert_a(coord,coord2,a,3)

        #def_coord = np.zeros([nnode,3])

        def_coord = coord2 + a_node*def_scale

        #print(a_node)
        
        #for i in range(np.size(def_coord, axis = 0)):
            #def_coord[i] = coord2[i] + a_node[i]
        #    def_coord[i,0] = coord2[i,0] + a[i*3]
        #    def_coord[i,1] = coord2[i,1] + a[i*3+1]
        #    def_coord[i,2] = coord2[i,2] + a[i*3+2]

        


        #print(def_coord)
        
        """
        for i in range(nnode):
        #a_dx, a_dy, a_dz = get_a_from_coord(i,3,a,def_scale)
        #x = coord[i][0]+a_dx
        #y = coord[i][1]+a_dy
        #z = coord[i][2]+a_dz
        #def_coord[i] = [x,y,z]
            def_coord[i,0] = a[i*3]
            def_coord[i,1] = a[i*3+1]
            def_coord[i,2] = a[i*3+2]
        """

        #print(a)
        #print(np.size(a, axis = 0))
        #print(np.size(a, axis = 1))
        """
        for i in range(0, ncoord):
            #if a.any() == None:
            #    x = coord[i,0]
            #    y = coord[i,1]
            #    z = coord[i,2]
            #else:
            #a_dx, a_dy, a_dz = get_a_from_coord(i,3,a,def_scale)

            #x = coord[i,0]+a_dx
            #y = coord[i,1]+a_dy
            #z = coord[i,2]+a_dz

            x = coord[i][0]+a[i][0]*scale
            y = coord[i][1]+a[i][1]*scale
            z = coord[i][2]+a[i][2]*scale

            def_coord[i] = [x,y,z]

            #def_nodes.append(v.Sphere(c='white').scale(1.5*scale).pos([x,y,z]).alpha(alpha))
        """
        #meshes = []
        #nel = np.size(edof, axis = 0)
        
        

        #print(topo)

        #mesh = v.Mesh([def_coord, topo]).lw(1)
        




        ct = vtk.VTK_HEXAHEDRON

        celltypes = [ct] * nel

        ug=v.UGrid([def_coord, topo, celltypes])
        ug.points(def_coord)
        
        mesh = ug.tomesh().lw(1).alpha(alpha)

        #v.settings.useDepthPeeling = True

        #print(val)

        #print('Cell connectivity: ',mesh.faces())

        #elif val and val == 'nodal_values':
        if val and val == 'el_values':
            #print(val)
            #vmin, vmax = np.min(values), np.max(values)
            
            el_values = vdu.convert_el_values(edof,values)
            mesh.celldata["val"] = el_values

            mesh.cmap(colormap, "val", on="cells")
'''



#Start Calfem-vedo visualization
cfvv.show_and_wait()

