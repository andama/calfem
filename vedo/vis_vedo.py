# -*- coding: utf-8 -*-
"""
CALFEM VTK

Contains all the functions for 3D visualization in CALFEM using VTK

@author: Andreas Åmand
"""

"""
# -*- coding: utf-8 -*-

class BeamViz:
    def __init__(self):
        self.elements
        self.nodes
        ...

    def init_vedo_geometry(self):
        ...
    def render_geometry(self):
        ...

class BeamVizWindow(Qt.QMainWindow):
    def __init__(self):
        self.beam_viz = BeamViz()
        ...

    def draw_geometry(self):
        self.beam_viz.render_geometry(...)

    def draw_displaced_geometry(self):
        self.beam_viz.render_geometry(...)

def draw_geometry(...):
    viz_window = BeamVizWindow(...)
    viz_window.draw_geometry(...)

def draw_displaced_geometry(...):
    viz_window = BeamVizWindow(...)
    viz_window.draw_displaced_geometry(...)

"""

import numpy as np
import vedo as v
#import polyscope as p
import sys
from PyQt5 import Qt
from PyQt5 import uic
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

"""
class data:
    beam_3D_nodes = None
    beam_3D_elements = None
    beam_3D_def_nodes = None
    beam_3D_def_elements = None
    beam_3D_el_values = None

    solid_3D_mesh = None
    solid_3D_def_mesh = None
    solid_3D_el_values = None
"""

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



class VedoMainWindow(Qt.QMainWindow):
    
    def __init__(self):
        """
        self.beam3d = beam3d()
        self.solid3d = solid3d()
        
        self.beam_elements = self.beam3d.elements
        print(self.beam_elements)
        self.beam_nodes = self.beam3d.nodes
        self.beam_displaced_elements = self.beam3d.displaced_elements
        self.beam_displaced_nodes = self.beam3d.displaced_nodes
        self.beam_element_values = self.beam3d.element_values
        
        
        elements = beamdata.elements
        nodes = beamdata.nodes
        def_elements = beamdata.def_elements
        def_nodes = beamdata.def_nodes
        el_values = beamdata.el_values[:, 0]
        

        beam_3D_nodes = data.beam_3D_nodes
        beam_3D_elements = data.beam_3D_elements
        beam_3D_def_nodes = data.beam_3D_def_nodes
        beam_3D_def_elements = data.beam_3D_def_elements
        beam_3D_el_values = data.beam_3D_el_values

        solid_3D_mesh = data.solid_3D_mesh
        solid_3D_def_mesh = data.solid_3D_def_mesh
        solid_3D_el_values = data.solid_3D_el_values
        """

        Qt.QMainWindow.__init__(self)
        
        #self.initialize(def_elements)
        self.initialize()
        #self.render_beam_geometry(beam_3D_nodes,beam_3D_elements)
        #self.render_beam_geometry(beam_3D_def_nodes,beam_3D_def_elements)
        #self.render_solid_geometry(solid_3D_mesh)
        #self.render_solid_geometry(solid_3D_def_mesh)
        #self.render_geometry(nodes,elements)
        #self.render_geometry(def_nodes,def_elements)

    def initialize(self):
        uic.loadUi("QtVTKMainWindow.ui", self)
        self.frame = Qt.QFrame()
        self.layout = Qt.QVBoxLayout()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)

        #p.set_program_name("vedo using polyscope")
        #p.set_up_dir("z_up")
        #p.init()

        # Create renderer and add the vedo objects and callbacks
        self.plotter = v.Plotter(qtWidget=self.vtkWidget,bg2="blackboard",bg="black",axes=4)

        #self.vp.show(def_elements, axes=1, viewup='y')
        self.plotter.show(axes=1, viewup='y')

        #self.vp.addGlobalAxes(8)

        self.layout.addWidget(self.vtkWidget)
        self.frame.setLayout(self.layout)
        self.setCentralWidget(self.frame)

        self.show()

    def render_beam_geometry(self,beam_nodes,beam_elements):
        
        nnode = np.size(beam_nodes, axis = 0)
        for i in range(nnode):
            self.plotter += beam_nodes[i]

        nel = np.size(beam_elements, axis = 0)
        for i in range(nel):
            self.plotter += beam_elements[i]

    def render_solid_geometry(self,solid_mesh):

        #nel = np.size(elements, axis = 0)
        for i in range(1):
            self.plotter += solid_mesh
            #ps_mesh = p.register_surface_mesh("My vedo mesh",solid_mesh.points(),solid_mesh.faces(),color=[0.5,0,0],smooth_shade=False)

        #ps_mesh.add_scalar_quantity("heights", solid_mesh.points()[:,2], defined_on='vertices')
        #p.show()

    def onClose(self):
        printc("..calling onClose")
        self.vtkWidget.close()



class beam3d:
    #def __init__(self,edof,coord,dof,a,el_values,label,nseg):
    """
    def __init__(self):
        #self.MainWindow = MainWindow():
        #self.elements
        #self.nodes = []
        self.displaced_elements = []
        self.displaced_nodes = []
        self.element_values = []
    """

    def draw_geometry(edof,coord,dof,scale=0.2,alpha=1):
        
        nnode = np.size(coord, axis = 0)
        #nodes = []
        nodes = []
        for i in range(nnode):
            nodes.append(v.Sphere().scale(1.5*scale).pos([coord[i,0],coord[i,1],coord[i,2]]).alpha(alpha))

        nel = np.size(edof, axis = 0)
        #elements = []
        elements = []
        
        res = 4
        for i in range(nel):
            coord1,coord2 = tools.get_coord_from_edof(edof[i,:],dof)

            tube = v.Cylinder([[coord[coord1,0],coord[coord1,1],coord[coord1,2]],[coord[coord2,0],coord[coord2,1],coord[coord2,2]]],r=scale,res=res).alpha(alpha)

            #tube = v.Cylinder(height=h/scale,res=res).scale(scale).pos([x,y,z]).orientation([dx,dy,dz]).alpha(alpha)

            elements.append(tube)

        #self.MainWindow.beam_nodes = nodes
        #self.MainWindow.beam_nodes = elements
        data.beam_3D_nodes = nodes
        data.beam_3D_elements = elements

    def draw_displaced_geometry(edof,coord,dof,a,el_values,label,scale=0.02,alpha=1,def_scale=1,nseg=2):

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

        #beamdata.def_coord = def_coord

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


                
        #self.MainWindow.beam_displaced_nodes = def_nodes
        #self.MainWindow.beam_displaced_elements = def_elements
        #self.MainWindow.beam_element_values = el_values
        data.beam_3D_def_nodes = def_nodes
        data.beam_3D_def_elements = def_elements
        data.beam_3D_el_values = el_values


def init_app():
    global vedo_app
    vedo_app = Qt.QApplication.instance()

    if vedo_app is None:
        print("No QApplication instance found. Creating one.")
        # if it does not exist then a QApplication is created
        vedo_app = Qt.QApplication(sys.argv)
    else:
        print("QApplication instance found.")

    return vedo_app

class VedoPlotWindow:
    __instance = None
    @staticmethod 
    def intance():
        """ Static access method. """
        if VedoPlotWindow.__instance == None:
            VedoPlotWindow()

        return VedoPlotWindow.__instance

    def __init__(self):
        """ Virtually private constructor. """
        if VedoPlotWindow.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            VedoPlotWindow.__instance = self
            self.plot_window = VedoMainWindow()

def draw_geometry(edof,coord,dof,scale,alpha,export=False):
    mesh = v.Mesh([coord,[[0,1,2,3],[4,5,6,7],[0,3,7,4],[1,2,6,5],[0,1,5,4],[2,3,7,6]]],alpha=alpha)

    if export==True:
        v.io.write(mesh, "solid.vtk")
        #print(export)

    app = init_app()
    plot_window = VedoPlotWindow.intance().plot_window
    plot_window.render_solid_geometry(mesh)
"""
class solid3d:
    
    #def __init__(self):
    #    self.mesh = []
    #    self.displaced_mesh = []
    

    def draw_geometry(edof,coord,dof,scale,alpha):
        app = init_app()
        plot_window = VedoPlotWindow.intance().plot_window
        mesh = v.mesh.Mesh([coord,[[0,1,2,3],[4,5,6,7],[0,3,7,4],[1,2,6,5],[0,1,5,4],[2,3,7,6]]],alpha=alpha)
        plot_window.render_solid_geometry(mesh)
        #data.solid_3D_mesh = mesh

    def draw_displaced_geometry(edof,coord,dof,a,el_values,label,scale=0.02,alpha=1,def_scale=1,nseg=2):
        ncoord = np.size(coord, axis = 0)
        def_nodes = []
        def_coord = np.zeros([ncoord,3])

        for i in range(0, ncoord):
            a_dx, a_dy, a_dz = tools.get_a_from_coord(i,6,a,def_scale)

            x = coord[i,0]+a_dx
            y = coord[i,1]+a_dy
            z = coord[i,2]+a_dz

            def_coord[i] = [x,y,z]

            #def_nodes.append(v.Sphere().scale(1.5*scale).pos([x,y,z]).alpha(alpha))
            
        mesh = v.Mesh([def_coord,[[0,1,2,3],[4,5,6,7],[0,3,7,4],[1,2,6,5],[0,1,5,4],[2,3,7,6]]],alpha=alpha)
        data.solid_3D_def_mesh = mesh
"""
#Start Calfem-vedo visualization
def show_and_wait():
    #app = Qt.QApplication(sys.argv)
    #window = MainWindow()
    app = init_app()
    app.exec_()

