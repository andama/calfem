# -*- coding: utf-8 -*-
"""
CALFEM VTK

Contains all the functions for 3D visualization in CALFEM using VTK

@author: Andreas Åmand
"""

import numpy as np
import vedo as v
import sys
from PyQt5 import Qt
from PyQt5 import uic
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor



class beamdata:
    beam3d_stresses = None
    def_coord = None



class tools:
    def get_coord_from_edof(edof_row,dof):
        edof_row1,edof_row2 = np.split(edof_row,2)
        coord1 = int(np.where(np.all(edof_row1==dof,axis=1))[0])
        coord2 = int(np.where(np.all(edof_row2==dof,axis=1))[0])
        return coord1, coord2

    def get_a_from_coord(coord_row_num,num_of_deformations,a,scale):
        dx = a[coord_row_num*num_of_deformations]*scale
        dy = a[coord_row_num*num_of_deformations+1]*scale
        dz = a[coord_row_num*num_of_deformations+2]*scale
        return dx, dy, dz



class MainWindow(Qt.QMainWindow):
    
    def __init__(self):
        Qt.QMainWindow.__init__(self)
        
        elements = beamdata.elements
        nodes = beamdata.nodes
        def_elements = beamdata.def_elements
        def_nodes = beamdata.def_nodes
        el_values = beamdata.el_values[:, 0]

        self.initialize(def_elements)

        self.render_geometry(nodes,elements)
        self.render_geometry(def_nodes,def_elements)

    def initialize(self,def_elements):
        uic.loadUi("QtVTKMainWindow.ui", self)
        self.frame = Qt.QFrame()
        self.layout = Qt.QVBoxLayout()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)

        # Create renderer and add the vedo objects and callbacks
        self.vp = v.Plotter(qtWidget=self.vtkWidget,bg2="blackboard",bg="black",axes=4)

        self.vp.show(def_elements, axes=1, viewup='y')

        #self.vp.addGlobalAxes(8)

        self.layout.addWidget(self.vtkWidget)
        self.frame.setLayout(self.layout)
        self.setCentralWidget(self.frame)

        self.show()

    def render_geometry(self,nodes,elements):
        
        nnode = np.size(nodes, axis = 0)
        for i in range(nnode):
            self.vp += nodes[i]

        nel = np.size(elements, axis = 0)
        for i in range(nel):
            self.vp += elements[i]

    def onClose(self):
        printc("..calling onClose")
        self.vtkWidget.close()



class beam3d:
    def geometry(edof,coord,dof,scale,alpha):
        
        nnode = np.size(coord, axis = 0)
        nodes = []
        for i in range(nnode):
            nodes.append(v.Sphere().scale(1.5*scale).pos([coord[i,0],coord[i,1],coord[i,2]]).alpha(alpha))

        nel = np.size(edof, axis = 0)
        elements = []

        
        res = 4
        for i in range(nel):
            coord1,coord2 = tools.get_coord_from_edof(edof[i,:],dof)

            tube = v.Cylinder([[coord[coord1,0],coord[coord1,1],coord[coord1,2]],[coord[coord2,0],coord[coord2,1],coord[coord2,2]]],r=scale,res=res).alpha(alpha)

            #tube = v.Cylinder(height=h/scale,res=res).scale(scale).pos([x,y,z]).orientation([dx,dy,dz]).alpha(alpha)

            elements.append(tube)

        beamdata.nodes = nodes
        beamdata.elements = elements

    def def_geometry(edof,coord,dof,a,el_values,label,scale=0.02,alpha=1,def_scale=1,nseg=2):

        ndof = np.size(a, axis = 0)
        ncoord = np.size(coord, axis = 0)
        def_nodes = []
        def_coord = np.zeros([ncoord,3])

        for i in range(0, ncoord):
            a_dx, a_dy, a_dz = tools.get_a_from_coord(i,6,a,def_scale)

            x = coord[i,0]+a_dx
            y = coord[i,1]+a_dy
            z = coord[i,2]+a_dz

            def_coord[i] = [x,y,z]

            def_nodes.append(v.Sphere().scale(1.5*scale).pos([x,y,z]).alpha(alpha))

        beamdata.def_coord = def_coord

        nel = np.size(edof, axis = 0)
        def_elements = []
        scale = 0.02
        res = 4

        def_coords = []

        vmin, vmax = np.min(el_values), np.max(el_values)

        #if nseg > 2:
        #    el_values = np.zeros((1,nseg))[0,:]

        el_values_array = np.zeros((1,4*res))[0,:]

        for i in range(nel):
            coord1,coord2 = tools.get_coord_from_edof(edof[i,:],dof)

            #tube = v.Cylinder(height=h/scale,res=res).scale(scale).pos([x,y,z]).orientation([dx,dy,dz])


            if nseg > 2:
                steps = np.float32(1/(nseg-1))

                dx = (def_coord[coord2,0]-def_coord[coord1,0])*steps
                dy = (def_coord[coord2,1]-def_coord[coord1,1])*steps
                dz = (def_coord[coord2,2]-def_coord[coord1,2])*steps

                for j in range(nseg-1):
                    x1 = def_coord[coord1,0]+dx*j
                    y1 = def_coord[coord1,1]+dy*j
                    z1 = def_coord[coord1,2]+dz*j

                    x2 = def_coord[coord1,0]+dx*(j+1)
                    y2 = def_coord[coord1,1]+dy*(j+1)
                    z2 = def_coord[coord1,2]+dz*(j+1)

                    tube = v.Cylinder([[x1,y1,z1],[x2,y2,z2]],r=scale,res=res).alpha(alpha)
                    def_elements.append(tube)

                    el_value1 = el_values[nseg*i,:]
                    el_value2 = el_values[nseg*i+1,:]

                    el_values_array[1] = el_value1
                    el_values_array[3] = el_value1
                    el_values_array[5] = el_value1
                    el_values_array[7] = el_value1
                    el_values_array[12] = el_value1
                    el_values_array[13] = el_value1
                    el_values_array[14] = el_value1
                    el_values_array[15] = el_value1

                    el_values_array[0] = el_value2
                    el_values_array[2] = el_value2
                    el_values_array[4] = el_value2
                    el_values_array[6] = el_value2
                    el_values_array[8] = el_value2
                    el_values_array[9] = el_value2
                    el_values_array[10] = el_value2
                    el_values_array[11] = el_value2

                    tube.cmap("jet", el_values_array, on="points", vmin=vmin, vmax=vmax).addScalarBar(label)


                #for k in range(nseg):
                #    el_value[k] = el_values[nseg*i,:]



            else:
                tube = v.Cylinder([[def_coord[coord1,0],def_coord[coord1,1],def_coord[coord1,2]],[def_coord[coord2,0],def_coord[coord2,1],def_coord[coord2,2]]],r=scale,res=res).alpha(alpha)
                def_elements.append(tube)

                el_value1 = el_values[2*i,:]
                el_value2 = el_values[2*i+1,:]

                el_values_array[1] = el_value1
                el_values_array[3] = el_value1
                el_values_array[5] = el_value1
                el_values_array[7] = el_value1
                el_values_array[12] = el_value1
                el_values_array[13] = el_value1
                el_values_array[14] = el_value1
                el_values_array[15] = el_value1

                el_values_array[0] = el_value2
                el_values_array[2] = el_value2
                el_values_array[4] = el_value2
                el_values_array[6] = el_value2
                el_values_array[8] = el_value2
                el_values_array[9] = el_value2
                el_values_array[10] = el_value2
                el_values_array[11] = el_value2

                tube.cmap("jet", el_values_array, on="points", vmin=vmin, vmax=vmax).addScalarBar(label)


                

        beamdata.def_nodes = def_nodes
        beamdata.def_elements = def_elements
        beamdata.el_values = el_values

