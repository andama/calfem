# -*- coding: utf-8 -*-

"""
CALFEM Vedo

Utils for 3D visualization in CALFEM using Vedo (https://vedo.embl.es/)

@author: Andreas Åmand & Jonas Lindemann
"""

import numpy as np
import vedo as v
import pyvtk
import vtk
import sys
#import webbrowser
from scipy.io import loadmat
import calfem.core as cfc

### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
# Tools, used in this file but can be accessed by a user as well (see exv4a.py/exv4b.py)

# Implementera nedanstående för att kontrollera att dim. stämmer för draw_mesh/draw_displaced_mesh
#def check_input(edof,coord,dof,element_type,a):

def get_coord_from_edof(edof_row,dof,element_type):
    if element_type == 1 or element_type == 2 or element_type == 5:
        edof_row1,edof_row2 = np.split(edof_row,2)
        coord1 = int(np.where(np.all(edof_row1==dof,axis=1))[0])
        coord2 = int(np.where(np.all(edof_row2==dof,axis=1))[0])
        return coord1, coord2
    elif element_type == 3 or element_type == 4:
        edof_row1,edof_row2,edof_row3,edof_row4,edof_row5,edof_row6,edof_row7,edof_row8 = np.split(edof_row,8)
        coord1 = int(np.where(np.all(edof_row1==dof,axis=1))[0])
        coord2 = int(np.where(np.all(edof_row2==dof,axis=1))[0])
        coord3 = int(np.where(np.all(edof_row3==dof,axis=1))[0])
        coord4 = int(np.where(np.all(edof_row4==dof,axis=1))[0])
        coord5 = int(np.where(np.all(edof_row5==dof,axis=1))[0])
        coord6 = int(np.where(np.all(edof_row6==dof,axis=1))[0])
        coord7 = int(np.where(np.all(edof_row7==dof,axis=1))[0])
        coord8 = int(np.where(np.all(edof_row8==dof,axis=1))[0])
        coords = np.array([coord1, coord2, coord3, coord4, coord5, coord6, coord7, coord8])
        return coords
    elif element_type == 6:
        edof_row1,edof_row2,edof_row3,edof_row4 = np.split(edof_row,4)
        coord1 = int(np.where(np.all(edof_row1==dof,axis=1))[0])
        coord2 = int(np.where(np.all(edof_row2==dof,axis=1))[0])
        coord3 = int(np.where(np.all(edof_row3==dof,axis=1))[0])
        coord4 = int(np.where(np.all(edof_row4==dof,axis=1))[0])
        coords = np.array([coord1, coord2, coord3, coord4])
        return coords

def get_a_from_coord(coord_row_num,num_of_deformations,a,scale=1):
    dx = a[coord_row_num*num_of_deformations]*scale
    dy = a[coord_row_num*num_of_deformations+1]*scale
    dz = a[coord_row_num*num_of_deformations+2]*scale
    return dx, dy, dz

def get_node_elements(coord,scale,alpha,t=None):
    nnode = np.size(coord, axis = 0)
    ncoord = np.size(coord, axis = 1)
    nodes = []
    for i in range(nnode):
        if ncoord == 3:
            node = v.Sphere(c='white').scale(1.5*scale).pos([coord[i,0],coord[i,1],coord[i,2]]).alpha(alpha)
            node.info = f"Node nr. {i}"
            nodes.append(node)
        elif ncoord == 2:
            node = v.Sphere(c='white').scale(1.5*scale).pos([coord[i,0],coord[i,1],0]).alpha(alpha)
            node.info = f"Node nr. {i}"
            nodes.append(node)
        #elif ncoord == 2 and t is not None:
        #    node = v.Sphere(c='white').scale(1.5*scale).pos([coord[i,0],coord[i,1],0]).alpha(alpha)
        #    node.info = f"Node nr. {i}"
        #    nodes.append(node)
        elif ncoord == 1:
            node = v.Sphere(c='white').scale(1.5*scale).pos([coord[i,0],0,0]).alpha(alpha)
            node.info = f"Node nr. {i}"
            nodes.append(node)
    return nodes



def convert_a(coord_old,coord_new,a,ndofs):

    ncoord = np.size(coord_old, axis=0)

    print(ncoord)

    a_new = np.zeros((ncoord,ndofs))

    #print(coord_old)
    #print(coord_new)

    print(coord_old[0])
    print(coord_new[0])

    #print(np.size(coord_old, axis=0))
    #print(np.size(coord_new, axis=0))

    indexes = []

    coord_hash_old = {}
    coord_hash_new = {}
    coord_hash_old_numbers = {}
    coord_hash_new_numbers = {}

    #node = np.zeros((ncoord, ndofs), dtype=int)


    for i in range(ncoord):
        #el_dof[i] = el_dofs[ (i*n_dofs_per_node):((i+1)*n_dofs_per_node) ]
        #node[i] = coord_old[ i ]
        #coord_hash_old[hash(tuple(node[i]))] = [ coord_old[i] ]
        #coord_hash_new[hash(tuple(node[i]))] = [ coord_new[i] ]
        coord_hash_old[hash(tuple(coord_old[i]))] = i
        coord_hash_new[hash(tuple(coord_new[i]))] = i
        #node_hash_a[hash(tuple(a_node[i]))] = a
        #node_hash_a[hash(tuple(el_dof[i]))] = a
        #node_hash_coords[hash(tuple(el_dof[i]))] = [elx[i]+a[i*3], ely[i]+a[i*3+1], elz[i]+a[i*3+2]]
        #node_hash_a[hash(tuple(a_upd[i]))] = [ a[i*3], a[i*3+1], a[i*3+2] ]
        #coord_hash_old_numbers[hash(tuple(coord_old[i]))] = -1
        #coord_hash_new_numbers[hash(tuple(coord_new[i]))] = -1
        #coord_hash_old_numbers[hash(tuple(coord_old[i]))] = [coord_old[i]]
        #coord_hash_new_numbers[hash(tuple(coord_new[i]))] = [coord_new[i]]


    #print(coord_hash_new.values())
    #print(coord_hash_old.values())
    #coord_count = 0

    for node_hash in coord_hash_old.keys():

        #coord_hash_old_numbers[node_hash] = coord_count
        #print(node_hash_numbers[node_hash])
        #node_hash_a[hash(tuple(a))] = a_upd
        #indexes.append(coord_hash_new_numbers[node_hash])
        index = coord_hash_new[node_hash]
        indexes.append(index)

        #if np.all(old_row) == np.all(new_row):
        #        index = j
        #        indexes.append(index)
            #index = np.where(np.all(old_row==new_row,axis=0))

        #coord_count +=1

        #coords.append(node_hash_coords[node_hash])
        #a_node_new.append(node_hash_a[node_hash])
        #a_node_new.append(node_hash_coords[node_hash])
        #print(node_hash_numbers.keys())
        #print(node_hash_coords)
        #a_new.append(node_hash_a[node_hash])
        #a_new.append(hash(node_hash_a[node_hash]))
        #a_new.append(hash(tuple(node_hash_a[node_hash])))

    node = 0

    for index in zip(indexes):
        a_new[index,0] = a[node*3]
        a_new[index,1] = a[node*3+1]
        a_new[index,2] = a[node*3+2]
        node += 1

    """
    for i in range(ncoord):
        old_row = coord_old[i]
        
        #print(old_row)
        for j in range(ncoord):
            new_row = coord_new[j]
            if np.all(old_row) == np.all(new_row):
                index = j
                indexes.append(index)
            #index = np.where(np.all(old_row==new_row,axis=0))
            
            #print(new_row)
            
        #if coord_old[i] == 
    """
    #print(indexes)

    return a_new










def convert_to_node_topo(edof, ex, ey, ez, a, n_dofs_per_node=3, ignore_first=False):
    """
    Written by: Jonas Lindemann
    Modified by: Andreas Åmand

    Routine to convert dof based topology and element coordinates to node based
    topology required for visualisation with VTK and other visualisation frameworks

    :param array edof: element topology [nel x (n_dofs_per_node)|(n_dofs_per_node+1)*n_nodes ]
    :param array ex: element x coordinates [nel x n_nodes]
    :param array ey: element y coordinates [nel x n_nodes]
    :param array ez: element z coordinates [nel x n_nodes]
    :param array a: global deformation [ndof]
    :param array n_dofs_per_node: number of dofs per node. (default = 3)
    :param boolean ignore_first: ignore first column of edof. (default = False)
    :return array coords: Array of node coordinates. [n_nodes x 3]
    :return array topo: Node topology. [nel x n_nodes]
    :return array node_dofs: Dofs for each node. [n_nodes x n_dofs_per_node]
    :return array a: global deformation [ndof] (reorderd according to )
    """

    node_hash_coords = {}
    node_hash_numbers = {}
    a_hash_numbers = {}
    node_hash_a = {}
    node_hash_dofs = {}
    el_hash_dofs = []

    nel, cols = edof.shape

    if ignore_first:
        tot_dofs = cols-1
    else:
        tot_dofs = cols

    n_nodes = int(tot_dofs / n_dofs_per_node)

    print("cols    =", tot_dofs)
    print("nel     =", nel)
    print("n_nodes =", n_nodes)

    #node_hash_a[hash(tuple(a))] = a
    #print(node_hash_a)

    tot_nnodes = int(np.size(a, axis = 0)/3)

    a_node = np.zeros((tot_nnodes, n_dofs_per_node))
    #print(np.size(a_node, axis = 0),np.size(a_node, axis = 1))

    for i in range(tot_nnodes):
        a_node[i,:] = [a[i*3], a[i*3+1], a[i*3+2]]
        
        #node_hash_a[hash(tuple(a_node[i]))] = a_node[i,:]

    #print(a_node)


    # Loopar igenom element
    for elx, ely, elz, dofs, a in zip(ex, ey, ez, edof, a_node):

        

        if ignore_first:
            el_dofs = dofs[1:]
        else:
            el_dofs = dofs

        # 0 1 2  3 4 5  6 7 8  9 12 11 

        el_dof = np.zeros((n_nodes, n_dofs_per_node), dtype=int)
        #a_upd = np.zeros((n_nodes, n_dofs_per_node), dtype=int)
        el_hash_topo = []

        
        # Loopar igenom elementets noder
        for i in range(n_nodes):
            el_dof[i] = el_dofs[ (i*n_dofs_per_node):((i+1)*n_dofs_per_node) ]
            node_hash_coords[hash(tuple(el_dof[i]))] = [elx[i], ely[i], elz[i]]
            #node_hash_a[hash(tuple(a_node[i]))] = a
            node_hash_a[hash(tuple(el_dof[i]))] = a
            #node_hash_coords[hash(tuple(el_dof[i]))] = [elx[i]+a[i*3], ely[i]+a[i*3+1], elz[i]+a[i*3+2]]
            #node_hash_a[hash(tuple(a_upd[i]))] = [ a[i*3], a[i*3+1], a[i*3+2] ]
            node_hash_numbers[hash(tuple(el_dof[i]))] = -1
            a_hash_numbers[hash(tuple(el_dof[i]))] = -1

            node_hash_dofs[hash(tuple(el_dof[i]))] = el_dof[i]
            el_hash_topo.append(hash(tuple(el_dof[i])))
            

        el_hash_dofs.append(el_hash_topo)

    coord_count = 0
    """
    #for i in range(tot_nnodes):
    for node_hash in node_hash_numbers.keys():
        node_hash_numbers[node_hash] = coord_count
        #node_hash_numbers[node_hash] = coord_count
        #node[i] = el_dofs[ (i*n_dofs_per_node):((i+1)*n_dofs_per_node) ]
        a_node[i] = node_hash_numbers[node_hash]
        coord_count +=1
        node_hash_a[hash(tuple(node[i]))] = a[i]
    """


    #for i in range


    coord_count = 0

    coords = []
    node_dofs = []

    #a_new = []

    #print(node_hash_numbers.keys())
    #print(len(node_hash_a))
    #print(node_hash_a)
    #print(node_hash_coords)

    a_node_new = []

    # Skapar global koordinatmartis baserat på hashes
    for node_hash in node_hash_numbers.keys():
        node_hash_numbers[node_hash] = coord_count
        #print(node_hash_numbers[node_hash])
        #node_hash_a[hash(tuple(a))] = a_upd
        node_dofs.append(node_hash_dofs[node_hash])

        coord_count +=1

        coords.append(node_hash_coords[node_hash])
        a_node_new.append(node_hash_a[node_hash])
        #a_node_new.append(node_hash_coords[node_hash])
        #print(node_hash_numbers.keys())
        #print(node_hash_coords)
        #a_new.append(node_hash_a[node_hash])
        #a_new.append(hash(node_hash_a[node_hash]))
        #a_new.append(hash(tuple(node_hash_a[node_hash])))

    a_count = 0

    for a_hash in a_hash_numbers.keys():
        a_hash_numbers[node_hash] = coord_count

        a_count +=1

        a_node_new.append(node_hash_a[a_hash])
        #a_node_new.append(node_hash_coords[node_hash])
        #print(node_hash_numbers.keys())
        #print(node_hash_coords)
        #a_new.append(node_hash_a[node_hash])
        #a_new.append(hash(node_hash_a[node_hash]))
        #a_new.append(hash(tuple(node_hash_a[node_hash])))



    #for i in range(coord_count)
    #    node_hash_a[hash(tuple(el_dof[i]))] = -1

    #for node_hash in node_hash_numbers.keys():
    #    a_node.append()



    topo = []
    #a_el = []

    #print(el_hash_dofs)
    #print(node_hash_numbers)

    # Skapar global topologimartis baserat på hashes
    for el_hashes in el_hash_dofs:
        topo.append([
            node_hash_numbers[el_hashes[0]], 
            node_hash_numbers[el_hashes[1]], 
            node_hash_numbers[el_hashes[2]], 
            node_hash_numbers[el_hashes[3]]
            ])
        topo.append([
            node_hash_numbers[el_hashes[4]], 
            node_hash_numbers[el_hashes[5]], 
            node_hash_numbers[el_hashes[6]], 
            node_hash_numbers[el_hashes[7]]
            ])
        topo.append([
            node_hash_numbers[el_hashes[0]],
            node_hash_numbers[el_hashes[3]],
            node_hash_numbers[el_hashes[7]],
            node_hash_numbers[el_hashes[4]]
            ])
        topo.append([
            node_hash_numbers[el_hashes[1]],
            node_hash_numbers[el_hashes[2]],
            node_hash_numbers[el_hashes[6]],
            node_hash_numbers[el_hashes[5]]
            ])
        topo.append([
            node_hash_numbers[el_hashes[0]],
            node_hash_numbers[el_hashes[1]],
            node_hash_numbers[el_hashes[5]],
            node_hash_numbers[el_hashes[4]]
            ])
        topo.append([
            node_hash_numbers[el_hashes[2]],
            node_hash_numbers[el_hashes[3]],
            node_hash_numbers[el_hashes[7]],
            node_hash_numbers[el_hashes[6]]
            ])

        #a_el.append(a[node_hash_numbers[el_hashes[0]]])
        #a_el.append(a[node_hash_numbers[el_hashes[1]]])
        #a_el.append(a[node_hash_numbers[el_hashes[2]]])
        #a_el.append(a[node_hash_numbers[el_hashes[3]]])
        #a_el.append(a[node_hash_numbers[el_hashes[4]]])
        #a_el.append(a[node_hash_numbers[el_hashes[5]]])
        #a_el.append(a[node_hash_numbers[el_hashes[6]]])
        #a_el.append(a[node_hash_numbers[el_hashes[7]]])
        

        """
        topo.append([
            node_hash_numbers[el_hashes[0]], 
            node_hash_numbers[el_hashes[1]], 
            node_hash_numbers[el_hashes[2]], 
            node_hash_numbers[el_hashes[3]]
            ]
        )
        """

        #print(coords)
    """
    a = a.tolist()
    print(a)

    for i in range(len(coords)):
        coords[i][0] = coords[i][0] + a[i*3]
        coords[i][1] = coords[i][1] + a[i*3+1]
        coords[i][2] = coords[i][2] + a[i*3+2]
    """
        
        
    #mesh = v.Mesh([def_coord[coords,:],[[0,1,2,3],[4,5,6,7],[0,3,7,4],[1,2,6,5],[0,1,5,4],[2,3,7,6]]],alpha=alpha).lw(1)

    return coords, topo, node_dofs, a_node_new






def ugrid_from_edof_ec(edof, ex, ey, ez, a, dofs_per_node=3, ignore_first=False):
    coords, topo, node_dofs = convert_to_node_topo_upd(edof, ex, ey, ez, dofs_per_node, ignore_first)

    npoint = coords.shape[0]
    nel = topo.shape[0]
    nnd = topo.shape[1]

    for i in range(npoint):
        #print([a[i*3],a[i*3+1],a[i*3+2]])
        coords[i][0] = coords[i][0] + a[i*3]
        coords[i][1] = coords[i][1] + a[i*3+1]
        coords[i][2] = coords[i][2] + a[i*3+2]

    if nnd == 4:
        ct = vtk.VTK_TETRA
    elif nnd == 8:
        ct = vtk.VTK_HEXAHEDRON
    else:
        print("Topology not supported.")

    celltypes = [ct] * nel

    return v.UGrid([coords, topo, celltypes])

def convert_to_node_topo_upd(edof, ex, ey, ez, dofs_per_node=3, ignore_first=False):
    """
    Routine to convert dof based topology and element coordinates to node based
    topology required for visualisation with VTK and other visualisation frameworks

    :param array edof: element topology [nel x (n_dofs_per_node)|(n_dofs_per_node+1)*n_nodes ]
    :param array ex: element x coordinates [nel x n_nodes]
    :param array ey: element y coordinates [nel x n_nodes]
    :param array ez: element z coordinates [nel x n_nodes]
    :param array n_dofs_per_node: number of dofs per node. (default = 3)
    :param boolean ignore_first: ignore first column of edof. (default = True)
    :return array coords: Array of node coordinates. [n_nodes x 3]
    :return array topo: Node topology. [nel x n_nodes]
    :return array node_dofs: Dofs for each node. [n_nodes x n_dofs_per_node]
    """

    node_hash_coords = {}
    node_hash_numbers = {}
    node_hash_dofs = {}
    el_hash_dofs = []

    nel, cols = edof.shape

    if ignore_first:
        tot_dofs = cols-1
    else:
        tot_dofs = cols

    n_nodes = int(tot_dofs / dofs_per_node)

    print("n_dofs_per_node =", dofs_per_node)
    print("cols    =", tot_dofs)
    print("nel     =", nel)
    print("n_nodes =", n_nodes)

    for elx, ely, elz, dofs in zip(ex, ey, ez, edof):

        if ignore_first:
            el_dofs = dofs[1:]
        else:
            el_dofs = dofs

        # 0 1 2  3 4 5  6 7 8  9 12 11 

        el_dof = np.zeros((n_nodes, dofs_per_node), dtype=int)
        el_hash_topo = []

        for i in range(n_nodes):
            el_dof[i] = el_dofs[ (i*dofs_per_node):((i+1)*dofs_per_node) ]
            node_hash_coords[hash(tuple(el_dof[i]))] = [elx[i], ely[i], elz[i]]
            node_hash_numbers[hash(tuple(el_dof[i]))] = -1
            node_hash_dofs[hash(tuple(el_dof[i]))] = el_dof[i]
            el_hash_topo.append(hash(tuple(el_dof[i])))

        el_hash_dofs.append(el_hash_topo)

    coord_count = 0

    coords = []
    node_dofs = []

    for node_hash in node_hash_numbers.keys():
        node_hash_numbers[node_hash] = coord_count
        node_dofs.append(node_hash_dofs[node_hash])
        coord_count +=1

        coords.append(node_hash_coords[node_hash])

    topo = []

    for el_hashes in el_hash_dofs:

        el_hash_topo = []

        for el_hash in el_hashes:
            el_hash_topo.append(node_hash_numbers[el_hash])

        topo.append(el_hash_topo)

        # topo.append([
        #     node_hash_numbers[el_hashes[0]], 
        #     node_hash_numbers[el_hashes[1]], 
        #     node_hash_numbers[el_hashes[2]], 
        #     node_hash_numbers[el_hashes[3]]
        #     ]
        # )

    return np.asarray(coords), np.asarray(topo), np.asarray(node_dofs)









