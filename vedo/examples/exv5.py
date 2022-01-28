# -*- coding: utf-8 -*-
"""
3D example using Vedo, plate elements

@author: Andreas Åmand
"""
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
"""
import os
import sys

os.system('clear')
sys.path.append("../")
sys.path.append("../../../calfem-python-develop/calfem")

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
#import calfem.core as cfc
import core as cfc
import numpy as np
#import vis_vedo as cfvv
import vis_vedo_no_qt as cfvv
from PyQt5 import Qt

#import calfem.geometry as cfg
#import calfem.mesh as cfm
#import calfem.vis as cfv
import geometry as cfg
import mesh as cfm
import vis as cfv
import utils as cfu
#import calfem.utils as cfu



d=0.1;
t=0.05

ncoord_x = 12+1;
ncoord_y = 12+1;


ncoord_init = ncoord_x*ncoord_y;

coord = np.zeros([ncoord_init,2]);
row = 0;


for y in range(ncoord_y):
    for x in range(ncoord_x):
        coord[row,:] = [x*d,y*d];
        row = row+1;

ncoord = np.size(coord,0);



dof = np.zeros([ncoord,3]);

it = 1;
dofs = [0,1,2]
for row in range(ncoord):
    for col in dofs:
        dof[row,col] = it;
        it = it + 1;
    #dof[row] = row;


ndof = np.size(dof,0)*np.size(dof,1);



nel_x = (ncoord_x-1);
nel_y = (ncoord_y-1);

#edof = np.zeros([nel_x*nel_y*nel_z,8*3]);
edof = np.zeros((nel_x*nel_y,4*3));
bc = np.zeros((ncoord_y*2*3*2-12,1));

x_step = 1;
y_step = ncoord_x;

it = 0;
bc_it = 0;
node = 0;


for row in range(nel_y):
    for el in range(nel_x):

        #edof[it,0] = it;
        edof[it,0:3] = dof[node,:];
        #edof[it,0] = dof[node];
        
        if el == 0:
            bc[bc_it,0] = int(dof[node,0])
            bc_it = bc_it + 1;
            bc[bc_it,0] = int(dof[node,1])
            bc_it = bc_it + 1;
            bc[bc_it,0] = int(dof[node,2])
            bc_it = bc_it + 1;
        
        node = node+x_step;
        edof[it,3:6] = dof[node,:];
        #edof[it,1] = dof[node];
        print(bc_it)
        print(node)
        if el == nel_x-1:
            bc[bc_it,0] = int(dof[node,0])
            bc_it = bc_it + 1;
            bc[bc_it,0] = int(dof[node,1])
            bc_it = bc_it + 1;
            bc[bc_it,0] = int(dof[node,2])
            bc_it = bc_it + 1;
        
        node = node+y_step;
        edof[it,6:9] = dof[node,:];
        #edof[it,2] = dof[node];
        
        if el == nel_x-1:
            bc[bc_it,0] = int(dof[node,0])
            bc_it = bc_it + 1;
            bc[bc_it,0] = int(dof[node,1])
            bc_it = bc_it + 1;
            bc[bc_it,0] = int(dof[node,2])
            bc_it = bc_it + 1;
        
        node = node-x_step;
        edof[it,9:12] = dof[node,:];
        #edof[it,3] = dof[node];
        
        if el == 0:
            bc[bc_it,0] = int(dof[node,0])
            bc_it = bc_it + 1;
            bc[bc_it,0] = int(dof[node,1])
            bc_it = bc_it + 1;
            bc[bc_it,0] = int(dof[node,2])
            bc_it = bc_it + 1;
        

        if el == nel_x-1:
            #node = node-z_step-y_step-2*x_step+y_step;
            #node = node-z_step-2*x_step;
            node = node-y_step+2
        else:
            node = node+x_step-y_step;
        
        it = it+1;


#coord = np.delete(coord,84,0)
#dof = np.delete(dof,np.size(dof,0)-2,0)
#edof = np.delete(edof,65,0)
#edof = np.delete(edof,66,0)
#edof = np.delete(edof,77,0)
#edof = np.delete(edof,78,0)

#print(edof)

edof = np.int_(edof)


#print(ex)






nnode = np.size(coord, axis = 0)
ndof = np.size(dof, axis = 0)*np.size(dof, axis = 1)
nel = np.size(edof, axis = 0)

ep=[t]

E=25*1000000000
v=0.2
eq=-25*1000

D = cfc.hooke(1,E,v);

ex, ey = cfc.coordxtr(edof,coord,dof)

#print(ex[0])

#K = np.int_(np.zeros((ndof,ndof)))
K = np.zeros((ndof,ndof))
#f = np.int_(np.zeros((ndof,1)))
f = np.zeros((ndof,1))

#print(f)

for eltopo, elx, ely in zip(edof, ex, ey):
    Ke,fe = cfc.platre(elx, ely, ep, D, eq)
    # Transposing fe due to error in platre
    fe = np.transpose(fe)
    cfc.assem(eltopo, K, Ke, f, fe)
"""
for i in range(nel):
    print(ex[i])
    print(ey[i])
    print(ep)
    print(D)
    print(eq)
    Ke, fe = cfc.platre(ex[i], ey[i], ep, D, eq)
    print(fe)
    #print(edof[i,:])
    #print(K)
    K, f = cfc.assem(edof[i,:],K,Ke,f,fe)
"""
bc = bc[:,0]
#for i in range(np.size(bc)):
    #print(int(bc[i,0]))
    #bc = int(bc[i,0])
bc = bc.astype(np.int64)
#print(bc)
bcVal = np.zeros((1,np.size(bc)))
#print(bcVal[0])

a,r = cfc.solveq(K, f, bc, bcVal[0])

ed = cfc.extract_eldisp(edof,a)
#print(ed)

es = np.zeros((5,nel))

#for i in range(nel):
#    es[:,i] = cfc.platrs(ex[i],ey[i],ep,D,ed[i])

#print(es)



cfvv.draw_mesh(edof,coord,dof,6,scale=0.002)
cfvv.show_and_wait()
cfvv.figure(2)
cfvv.draw_displaced_mesh(edof,coord,dof,6,a,scale=0.002)

#Start Calfem-vedo visualization
cfvv.show_and_wait()



