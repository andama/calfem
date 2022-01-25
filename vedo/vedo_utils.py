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

def check_input(edof,coord,dof,element_type,a=None,values=None):
    if element_type == 1 or element_type == 2 or element_type == 5:
        number_of_nodes_per_element = 2
    elif element_type == 3 or element_type == 4:
        number_of_nodes_per_element = 8
    elif element_type == 6:
        number_of_nodes_per_element = 4

    number_of_elements = np.size(edof, axis=0)
    number_of_degrees_of_freedom_per_element = np.size(edof, axis=1)

    number_of_coordinates = np.size(coord, axis=0)
    number_of_dimensions = np.size(coord, axis=1)

    number_of_degrees_of_freedom = np.size(dof, axis=0)*np.size(dof, axis=1)
    degrees_of_freedom_per_node = np.size(dof, axis=1)

    if a is not None:
        number_of_displacements = np.size(a, axis=0)

    if values is not None:
        number_of_values = np.size(values, axis=0)*np.size(values, axis=1)

        if number_of_values == number_of_elements:
            val = 'el_values'
        elif number_of_values == number_of_elements*number_of_nodes_per_element:
            val = 'nodal_values_by_el'
        elif number_of_values == number_of_coordinates:
            val = 'nodal_values'
        else:
            print("Invalid number of element-/nodal values, please make sure values correspond to total number of elements or nodes")
            sys.exit()


    # Implementera kontroll av dim. stämmer för draw_mesh/draw_displaced_mesh
    if element_type == 1 or element_type == 2 or element_type == 5:
        print(element_type)
    elif element_type == 3 or element_type == 4:
        print(element_type)
    elif element_type == 6:
        print(element_type)

    if a is None and values is None:
        print(1)
        return number_of_elements, \
            number_of_degrees_of_freedom_per_element, \
            number_of_coordinates, \
            number_of_dimensions, \
            number_of_degrees_of_freedom, \
            degrees_of_freedom_per_node,
    elif a is None:
        print(2)
        return number_of_elements, \
            number_of_degrees_of_freedom_per_element, \
            number_of_coordinates, \
            number_of_dimensions, \
            number_of_degrees_of_freedom, \
            degrees_of_freedom_per_node, \
            val
    elif values is None:
        print(3)
        return number_of_elements, \
            number_of_degrees_of_freedom_per_element, \
            number_of_coordinates, \
            number_of_dimensions, \
            number_of_degrees_of_freedom, \
            degrees_of_freedom_per_node, \
            number_of_displacements
    else:
        print('test')
        return number_of_elements, \
            number_of_degrees_of_freedom_per_element, \
            number_of_coordinates, \
            number_of_dimensions, \
            number_of_degrees_of_freedom, \
            degrees_of_freedom_per_node, \
            number_of_displacements, \
            val
            




def ugrid_from_edof_ec(edof, ex, ey, ez, ed=None, dofs_per_node=3, ignore_first=True):
    
    coords, topo, node_dofs, node_displ = convert_to_node_topo(edof, ex, ey, ez, ed, dofs_per_node, ignore_first)

    npoint = coords.shape[0]
    nel = topo.shape[0]
    nnd = topo.shape[1]

    if nnd == 4:
        ct = vtk.VTK_TETRA
    elif nnd == 8:
        ct = vtk.VTK_HEXAHEDRON
    else:
        print("Topology not supported.")

    celltypes = [ct] * nel

    return UGrid([coords, topo, celltypes])

def convert_to_node_topo(edof, ex, ey, ez, ed=None, dofs_per_node=3, ignore_first=True):
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
    node_hash_displ = {}
    el_hash_dofs = []


    nel, cols = edof.shape

    if ignore_first:
        tot_dofs = cols-1
    else:
        tot_dofs = cols

    n_nodes = int(tot_dofs / dofs_per_node)

    # print("n_dofs_per_node =", dofs_per_node)
    # print("cols    =", tot_dofs)
    # print("nel     =", nel)
    # print("n_nodes =", n_nodes)

    if ed is None:
        ed = np.zeros((nel,n_nodes*dofs_per_node))

    for elx, ely, elz, eed, dofs in zip(ex, ey, ez, ed, edof):

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
            displ_dofs = []
            for j in range(dofs_per_node):
                displ_dofs.append(eed[i*dofs_per_node+j])
            node_hash_displ[hash(tuple(el_dof[i]))] = displ_dofs
            #node_hash_displ[hash(tuple(el_dof[i]))] = [eed[i*dofs_per_node], eed[i*dofs_per_node+1], eed[i*dofs_per_node+2]]
            el_hash_topo.append(hash(tuple(el_dof[i])))

        el_hash_dofs.append(el_hash_topo)

    coord_count = 0

    coords = []
    node_dofs = []
    node_displ = []

    for node_hash in node_hash_numbers.keys():
        node_hash_numbers[node_hash] = coord_count
        node_dofs.append(node_hash_dofs[node_hash])
        node_displ.append(node_hash_displ[node_hash])
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

    if ed is None:
        return np.asarray(coords), np.asarray(topo), np.asarray(node_dofs)
    else:
        #print('test')
        return np.asarray(coords), np.asarray(topo), np.asarray(node_dofs), np.asarray(node_displ)






def convert_nodal_values(edof,topo,dof,coord,values):
    nel = np.size(edof, axis=0)
    nnode = np.size(coord, axis=0)
    #nodal_value_array = np.zeros((ncoord,8))
    nodal_value_array = np.zeros((nnode,1))
    print('Number of element values: ', np.size(values, axis=0))
    print('Number of values per element: ', np.size(values, axis=1))

    topo_hash = {}
    topo_num = {}
    dof_hash = {}
    dof_num = {}
    edof_hash = {}
    edof_num = {}
    

    for i in range(nel):
        #print(i,': ',topo[i])
        #topo_hash[i] = hash(tuple(topo[i]))
        #topo_num[i] = tuple(topo[i])
        for j in range(8):
            #topo_hash[hash(tuple(topo[i]))] = i
            #topo_num[tuple(topo[i])] = i
            topo_hash[i] = hash(tuple(topo[i]))
            topo_num[i] = tuple(topo[i])
            #print(i,topo_num[i])

    #for i in range(nnode):
    #    dof_hash[hash(tuple(dof[i]))] = i
        #dof_hash[i] = hash(tuple(dof[i]))
    #    dof_num[tuple(dof[i])] = i
        #dof_num[i] = tuple(dof[i])

    #for i in range(nel):
        #edof_hash[hash(tuple(edof[i]))] = i
    #    for j in range(8):
            #print(edof[i,j*3:j*3+3])
    #        edof_hash[hash(tuple(edof[i,j*3:j*3+3]))] = [i,j]
            #edof_hash[[i,j]] = hash(tuple(edof[i,j*3:j*3+3]))
    #        edof_num[tuple(edof[i,j*3:j*3+3])] = [i,j]
            #edof_num[[i,j]] = tuple(edof[i,j*3:j*3+3])

    #print(topo_num)
    #print(dof_num)
    #print(edof_num)

    test = {}

    for el, nodes in topo_num.items():
        it = 0
        for i in zip(nodes):
            #print(i[0])
            #test.append([i[0],el,it])

            # Node: [Element, Element node number, Value]
            #test[i[0]] = tuple([el,it,values[el,it]])
            #test[i[0]] = tuple([values[el,it]])
            test[tuple([el,it])] = i[0]
            #print(i[0],test[i[0]])

            it += 1

    #print(values)

    #print(test)
    #print(len(test))
    
    test2 = {}
    #print('ind')
    for i in range(nnode):
        test2[i] = []
        #print(test[i])
    



    for data, node in test.items():
        #el = data[0]
        #el_node = data[1]

        test2[node].append(values[data])





    for i in range(nnode):
        nodal_value_array[i] = np.mean(test2[i])
        #print(test2[i])
        #print(nodal_value_array[i])

    return nodal_value_array

    """
    for node, data in test.items():
        #ind = [x for x, z in zip(test.keys()) if z == node]
        print(ind)
        #test2[node].append(data[2])
        l = test2[node]
        #print(l)
        l.append(data[0])
        test2[node] = l
        #test2[node].append(data[0])
    """




    #print(test2)






    #print(test)
    #print(len(test))
    #    for e_key, e_val in dof_num.items():
    #        row = []
    #        print(e_key,d_key)
    #        if d_key in 
            #if e_key == d_key:
                #row.append(e_val)

    #    test.append(row)
        #print(key)
        #indexes_dof.append(edof_hash[key])
        #ind = [x for x, z in enumerate(indexes_dof) if z == j] 
        #nodes_per_el.append(ind)































































    #for node, el_num in test2.items():
    #    print(test2[node])









    #edof_hash = {}
    #edof_split_hash = {}
    #dof_hash = {}
    #dof_num = {}
    #edof_num = {}

    #edof_split = np.zeros((nel,8))
    #edof_split = []

    #print(np.split(edof[0],8))
    """
    for i in range(nnode):
        dof_hash[hash(tuple(dof[i]))] = i
        #dof_hash[i] = hash(tuple(dof[i]))
        dof_num[tuple(dof[i])] = i
        #dof_num[i] = tuple(dof[i])

    for i in range(nel):
        #edof_hash[hash(tuple(edof[i]))] = i
        for j in range(8):
            #print(edof[i,j*3:j*3+3])
            edof_hash[hash(tuple(edof[i,j*3:j*3+3]))] = [i,j]
            #edof_hash[[i,j]] = hash(tuple(edof[i,j*3:j*3+3]))
            edof_num[tuple(edof[i,j*3:j*3+3])] = [i,j]
            #edof_num[[i,j]] = tuple(edof[i,j*3:j*3+3])
    """
    #print(dof_hash)
    #print(dof_num)
    #print(len(dof_num))
    #print(edof_num)
    #print(len(edof_num))

    #test = []
    """
    keys1 = set(dof_num.keys())
    keys2 = set(edof_num.keys())
    common_keys = set(keys1).intersection(set(keys2))
    print(common_keys)
    
    #ind = 0
    for d_key, d_val in dof_hash.items():
        for e_key, e_val in dof_hash.items():
            if d_key == e_key:
                t = edof_hash[d_key]
                test.append(t)
    """

    #print(test)
    #print(len(test))
        #for e_key, e_val in dof_num.items():
        #    row = []
        #    print(e_key,d_key)
        #    if d_key in 
            #if e_key == d_key:
                #row.append(e_val)

        #test.append(row)
        #print(key)
        #indexes_dof.append(edof_hash[key])
        #ind = [x for x, z in enumerate(indexes_dof) if z == j] 
        #nodes_per_el.append(ind)


    #print(test)


    """
    indexes_dof = []

    for key in dof_hash.keys():
        #print(key)
        indexes_dof.append(edof_hash[key])

    print(indexes_dof)
    indexes_edof = []

    for key in edof_hash.keys():
        #print(key)
        indexes_edof.append(dof_hash[key])

    print(indexes_edof)
    nodes_per_el = []

    for j in range(nel):
        ind = [x for x, z in enumerate(indexes_dof) if z == j] 
        nodes_per_el.append(ind)

    print(nodes_per_el)
    #print('Nodes per element: ',nodes_per_el)
    #print('Indexes: ',len(indexes))
    #print('Number of DoF hashes: ',len(dof_hash))
    #print('Number of EDoF hashes: ',len(edof_hash))


    nodal_values = []
    """
    #for i in zip(nodes_per_el):
        #print('Index: ',i[0])




    #for i in range(nel):
        #print(edof[i,0:3])
        #test = dof_num(edof[i,0:3])
    #    print(test)








    


    #for i in range(nel):
        #edof_split[i,0],edof_split[i,1],edof_split[i,2],edof_split[i,3],edof_split[i,4],edof_split[i,5],edof_split[i,6],edof_split[i,7]
    #    split = np.split(edof[i],8)
    #    split_dofs = []
    #    for j in range(8):
    #        split_dofs.append([split[j]])
    #    edof_split.append(split_dofs)

    #print(edof_split)

    #for i in range(nel):
        #el_hash_old[hash(tuple(edof[i]))] = i
        #edof_split_hash[hash(tuple(edof_split[i]))] = i
    #    edof_hash[hash(tuple(edof[i]))] = i
        #dof_hash[hash(tuple(dof[i]))] = i
    #    edof_num[(tuple(edof[i]))] = i
        #coord_hash_old[hash(tuple(coord_old[i]))] = i
        #coord_hash_new[hash(tuple(coord_new[i]))] = i
        #elem[i] = i

    #print(edof_split_hash)
    #print(edof_num)

    









    
    #coord_hash_old = {}
    #coord_hash_new = {}
    #node_hash_numbers = {}
    #nodes = np.zeros((ncoord, 1), dtype=int)
    #print(nodes)

    #for i in range(ncoord):
        #el_hash_old[hash(tuple(edof[i]))] = i
    #    coord_hash_old[hash(tuple(coord_old[i]))] = i
    #    coord_hash_new[hash(tuple(coord_new[i]))] = i
        #elem[i] = i

    #print(coord_hash_old)

    #indexes = []

    #for node_hash in coord_hash_old.keys():
    #    index = coord_hash_new[node_hash]
    #    indexes.append(index)

    #print(np.size(indexes))

    #node = 0
    #for index in zip(indexes):
        #print(nodal_value_array[index,:])
    #    print(values[node,:])
    #    nodal_value_array.append(values[node,:])
    #    node += 1

    #print(nodal_value_array)
    




    #indexes = []

    #for el in el_hash_old.values():
    #    index = elem[el]
    #    indexes.append(index)

    #print(indexes)

    #coord_count = 0

    # Skapar global koordinatmartis baserat på hashes
    #for node_hash in node_hash_numbers.keys():
    #    node_hash_numbers[node_hash] = coord_count
        #node_dofs.append(node_hash_dofs[node_hash])
        #nodal_values.append(node_hash_numbers[node_hash])

    #    coord_count +=1

        #coords.append(node_hash_coords[node_hash])
        
    #print(nodal_values)


























"""
def convert_a(coord_old,coord_new,a,ndofs):
    ncoord = np.size(coord_old, axis=0)
    a_new = np.zeros((ncoord,ndofs))

    coord_hash_old = {}
    coord_hash_new = {}

    for i in range(ncoord):
        coord_hash_old[hash(tuple(coord_old[i]))] = i
        coord_hash_new[hash(tuple(coord_new[i]))] = i

    indexes = []

    for node_hash in coord_hash_old.keys():
        index = coord_hash_new[node_hash]
        indexes.append(index)

    node = 0
    for index in zip(indexes):
        if ndofs == 1:
            a_new[index] = a[node]
        elif ndofs == 3:
            a_new[index,0] = a[node*3]
            a_new[index,1] = a[node*3+1]
            a_new[index,2] = a[node*3+2]
        node += 1

    # Returns disp. by node, i.e. a_new = [number of nodes x degrees of freedom per node]
    return a_new
"""



def convert_el_values(edof,values):
    nel = np.size(edof, axis=0)
    el_values = np.zeros((nel*6,1))

    for i in range(nel):
        el_values[i*6,:] = values[i]
        el_values[i*6+1,:] = values[i]
        el_values[i*6+2,:] = values[i]
        el_values[i*6+3,:] = values[i]
        el_values[i*6+4,:] = values[i]
        el_values[i*6+5,:] = values[i]
    """
    el_hash_old = {}
    elem = {}

    for i in range(nel):
        el_hash_old[hash(tuple(edof[i]))] = i
        elem[i] = i

    indexes = []

    for el in el_hash_old.values():
        index = elem[el]
        indexes.append(index)

    el = 0
    for index in zip(indexes):
        i = index[0]*6
        el_values[i,:] = values[el]
        i += 1
        el_values[i,:] = values[el]
        i += 1
        el_values[i,:] = values[el]
        i += 1
        el_values[i,:] = values[el]
        i += 1
        el_values[i,:] = values[el]
        i += 1
        el_values[i,:] = values[el]
        el += 1
    """
    return el_values

# def convert_to_node_topo(edof, ex, ey, ez, n_dofs_per_node=3, ignore_first=False):
#     """
#     Written by: Jonas Lindemann
#     Modified by: Andreas Åmand

#     Routine to convert dof based topology and element coordinates to node based
#     topology required for visualisation with VTK and other visualisation frameworks

#     :param array edof: element topology [nel x (n_dofs_per_node)|(n_dofs_per_node+1)*n_nodes ]
#     :param array ex: element x coordinates [nel x n_nodes]
#     :param array ey: element y coordinates [nel x n_nodes]
#     :param array ez: element z coordinates [nel x n_nodes]
#     :param array a: global deformation [ndof]
#     :param array n_dofs_per_node: number of dofs per node. (default = 3)
#     :param boolean ignore_first: ignore first column of edof. (default = False)
#     :return array coords: Array of node coordinates. [n_nodes x 3]
#     :return array topo: Node topology. [nel x n_nodes]
#     :return array node_dofs: Dofs for each node. [n_nodes x n_dofs_per_node]
#     :return array a: global deformation [ndof] (reorderd according to )
#     """

#     node_hash_coords = {}
#     node_hash_numbers = {}
#     #a_hash_numbers = {}
#     #node_hash_a = {}
#     node_hash_dofs = {}
#     el_hash_dofs = []

#     nel, cols = edof.shape

#     if ignore_first:
#         tot_dofs = cols-1
#     else:
#         tot_dofs = cols

#     n_nodes = int(tot_dofs / n_dofs_per_node)

#     print("cols    =", tot_dofs)
#     print("nel     =", nel)
#     print("n_nodes =", n_nodes)

#     #node_hash_a[hash(tuple(a))] = a
#     #print(node_hash_a)

#     #tot_nnodes = int(np.size(a, axis = 0)/3)

#     #a_node = np.zeros((tot_nnodes, n_dofs_per_node))
#     #print(np.size(a_node, axis = 0),np.size(a_node, axis = 1))

#     #for i in range(tot_nnodes):
#     #    a_node[i,:] = [a[i*3], a[i*3+1], a[i*3+2]]
        
#         #node_hash_a[hash(tuple(a_node[i]))] = a_node[i,:]

#     #print(a_node)


#     # Loopar igenom element
#     for elx, ely, elz, dofs in zip(ex, ey, ez, edof):

        

#         if ignore_first:
#             el_dofs = dofs[1:]
#         else:
#             el_dofs = dofs

#         # 0 1 2  3 4 5  6 7 8  9 12 11 

#         el_dof = np.zeros((n_nodes, n_dofs_per_node), dtype=int)
#         #a_upd = np.zeros((n_nodes, n_dofs_per_node), dtype=int)
#         el_hash_topo = []

        
#         # Loopar igenom elementets noder
#         for i in range(n_nodes):
#             el_dof[i] = el_dofs[ (i*n_dofs_per_node):((i+1)*n_dofs_per_node) ]
#             node_hash_coords[hash(tuple(el_dof[i]))] = [elx[i], ely[i], elz[i]]
#             #node_hash_a[hash(tuple(a_node[i]))] = a
#             #node_hash_a[hash(tuple(el_dof[i]))] = a
#             #node_hash_coords[hash(tuple(el_dof[i]))] = [elx[i]+a[i*3], ely[i]+a[i*3+1], elz[i]+a[i*3+2]]
#             #node_hash_a[hash(tuple(a_upd[i]))] = [ a[i*3], a[i*3+1], a[i*3+2] ]
#             node_hash_numbers[hash(tuple(el_dof[i]))] = -1
#             #a_hash_numbers[hash(tuple(el_dof[i]))] = -1

#             node_hash_dofs[hash(tuple(el_dof[i]))] = el_dof[i]
#             el_hash_topo.append(hash(tuple(el_dof[i])))
            

#         el_hash_dofs.append(el_hash_topo)

#     coord_count = 0
#     """
#     #for i in range(tot_nnodes):
#     for node_hash in node_hash_numbers.keys():
#         node_hash_numbers[node_hash] = coord_count
#         #node_hash_numbers[node_hash] = coord_count
#         #node[i] = el_dofs[ (i*n_dofs_per_node):((i+1)*n_dofs_per_node) ]
#         a_node[i] = node_hash_numbers[node_hash]
#         coord_count +=1
#         node_hash_a[hash(tuple(node[i]))] = a[i]
#     """


#     #for i in range


#     coord_count = 0

#     coords = []
#     node_dofs = []

#     #a_new = []

#     #print(node_hash_numbers.keys())
#     #print(len(node_hash_a))
#     #print(node_hash_a)
#     #print(node_hash_coords)

#     #a_node_new = []

#     # Skapar global koordinatmartis baserat på hashes
#     for node_hash in node_hash_numbers.keys():
#         node_hash_numbers[node_hash] = coord_count
#         #print(node_hash_numbers[node_hash])
#         #node_hash_a[hash(tuple(a))] = a_upd
#         node_dofs.append(node_hash_dofs[node_hash])

#         coord_count +=1

#         coords.append(node_hash_coords[node_hash])
#         #a_node_new.append(node_hash_a[node_hash])
#         #a_node_new.append(node_hash_coords[node_hash])
#         #print(node_hash_numbers.keys())
#         #print(node_hash_coords)
#         #a_new.append(node_hash_a[node_hash])
#         #a_new.append(hash(node_hash_a[node_hash]))
#         #a_new.append(hash(tuple(node_hash_a[node_hash])))
#     """
#     a_count = 0

#     for a_hash in a_hash_numbers.keys():
#         a_hash_numbers[node_hash] = coord_count

#         a_count +=1

#         a_node_new.append(node_hash_a[a_hash])
#         #a_node_new.append(node_hash_coords[node_hash])
#         #print(node_hash_numbers.keys())
#         #print(node_hash_coords)
#         #a_new.append(node_hash_a[node_hash])
#         #a_new.append(hash(node_hash_a[node_hash]))
#         #a_new.append(hash(tuple(node_hash_a[node_hash])))
#     """


#     #for i in range(coord_count)
#     #    node_hash_a[hash(tuple(el_dof[i]))] = -1

#     #for node_hash in node_hash_numbers.keys():
#     #    a_node.append()



#     topo = []
#     #a_el = []

#     #print(el_hash_dofs)
#     #print(node_hash_numbers)

#     # Skapar global topologimartis baserat på hashes
#     for el_hashes in el_hash_dofs:
#         topo.append([
#             node_hash_numbers[el_hashes[0]], 
#             node_hash_numbers[el_hashes[1]], 
#             node_hash_numbers[el_hashes[2]], 
#             node_hash_numbers[el_hashes[3]]
#             ])
#         topo.append([
#             node_hash_numbers[el_hashes[4]], 
#             node_hash_numbers[el_hashes[5]], 
#             node_hash_numbers[el_hashes[6]], 
#             node_hash_numbers[el_hashes[7]]
#             ])
#         topo.append([
#             node_hash_numbers[el_hashes[0]],
#             node_hash_numbers[el_hashes[3]],
#             node_hash_numbers[el_hashes[7]],
#             node_hash_numbers[el_hashes[4]]
#             ])
#         topo.append([
#             node_hash_numbers[el_hashes[1]],
#             node_hash_numbers[el_hashes[2]],
#             node_hash_numbers[el_hashes[6]],
#             node_hash_numbers[el_hashes[5]]
#             ])
#         topo.append([
#             node_hash_numbers[el_hashes[0]],
#             node_hash_numbers[el_hashes[1]],
#             node_hash_numbers[el_hashes[5]],
#             node_hash_numbers[el_hashes[4]]
#             ])
#         topo.append([
#             node_hash_numbers[el_hashes[2]],
#             node_hash_numbers[el_hashes[3]],
#             node_hash_numbers[el_hashes[7]],
#             node_hash_numbers[el_hashes[6]]
#             ])

#         #a_el.append(a[node_hash_numbers[el_hashes[0]]])
#         #a_el.append(a[node_hash_numbers[el_hashes[1]]])
#         #a_el.append(a[node_hash_numbers[el_hashes[2]]])
#         #a_el.append(a[node_hash_numbers[el_hashes[3]]])
#         #a_el.append(a[node_hash_numbers[el_hashes[4]]])
#         #a_el.append(a[node_hash_numbers[el_hashes[5]]])
#         #a_el.append(a[node_hash_numbers[el_hashes[6]]])
#         #a_el.append(a[node_hash_numbers[el_hashes[7]]])
        

#         """
#         topo.append([
#             node_hash_numbers[el_hashes[0]], 
#             node_hash_numbers[el_hashes[1]], 
#             node_hash_numbers[el_hashes[2]], 
#             node_hash_numbers[el_hashes[3]]
#             ]
#         )
#         """

#         #print(coords)
#     """
#     a = a.tolist()
#     print(a)

#     for i in range(len(coords)):
#         coords[i][0] = coords[i][0] + a[i*3]
#         coords[i][1] = coords[i][1] + a[i*3+1]
#         coords[i][2] = coords[i][2] + a[i*3+2]
#     """
        
        
#     #mesh = v.Mesh([def_coord[coords,:],[[0,1,2,3],[4,5,6,7],[0,3,7,4],[1,2,6,5],[0,1,5,4],[2,3,7,6]]],alpha=alpha).lw(1)

#     return coords, topo, node_dofs





"""
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
"""








