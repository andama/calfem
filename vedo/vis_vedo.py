# -*- coding: utf-8 -*-
"""
CALFEM VTK

Contains all the functions for 3D visualization in CALFEM using VTK

@author: Andreas Åmand
"""
"""
import numpy as np
import vtk
import PyQt5
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
#from PyQt5 import uic
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.Qt import QApplication, QUrl, QDesktopServices
import webbrowser
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
"""
import numpy as np
import vedo as v

#mlab.test_contour3d()

#class vedo_int:
   
"""
# Create a scalar field: the distance from point (15,15,15)
X, Y, Z = np.mgrid[:30, :30, :30]
v.scalar_field = ((X-15)**2 + (Y-15)**2 + (Z-15)**2)/225

# Create the Volume from the numpy object
vol = v.Volume(scalar_field)

# Generate the surface that contains all voxels in range [1,2]
lego = v.vol.legosurface(1,2).addScalarBar()

v.show(lego, axes=True)
"""

import sys
from PyQt5 import Qt
from PyQt5 import uic
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
#from vedo import Plotter, Cone, printc

class MainWindow(Qt.QMainWindow):
    
    def __init__(self,nodes,elements,def_nodes,def_elements):
        
        
        """
        # Load colors
        self.colors = vtk.vtkNamedColors()
        
        # Create container
        self.vl = QtWidgets.QVBoxLayout()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.vtkWidget.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.vl.addWidget(self.vtkWidget)
        
        # Create renderer & render window
        self.ren = vtk.vtkRenderer()
        self.renwin = self.vtkWidget.GetRenderWindow()
        self.renwin.AddRenderer(self.ren)
        self.iren = self.renwin.GetInteractor()
        self.iren.SetRenderWindow(self.renwin)

        # Setting frame
        self.frame.setLayout(self.vl)
        
        # Rotate default camera
        self.ren.GetActiveCamera().Azimuth(45)
        self.ren.GetActiveCamera().Pitch(-45)

        # Add gradient
        self.ren.GradientBackgroundOn()

        # Axis widget
        self.axesWidget()

        # Orientation widget
        self.orientation()
        
        # Starting render
        self.ren.ResetCamera()
        self.iren.Start()
        """
        #Qt.QMainWindow.__init__(self, parent)
        Qt.QMainWindow.__init__(self)
        

        normal_stresses = data.beam3d_stresses[:, 0]
        self.initialize(def_elements)
        #print(normal_stresses)

        self.render_geometry(nodes,elements,normal_stresses)
        self.render_geometry(def_nodes,def_elements,normal_stresses)
        #self.render_geometry(def_nodes,def_elements,es,A,Iy,Iz)
        #self.render_stresses(stresses)
        #self.render()

    def initialize(self,def_elements):
        uic.loadUi("QtVTKMainWindow.ui", self)
        self.frame = Qt.QFrame()
        self.layout = Qt.QVBoxLayout()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)

        # Create renderer and add the vedo objects and callbacks
        self.vp = v.Plotter(qtWidget=self.vtkWidget,bg2="blackboard",bg="black",axes=4)
        #self.id1 = self.vp.addCallback("mouse click", self.onMouseClick)
        #self.id2 = self.vp.addCallback("key press",   self.onKeypress)
        #self.vp += v.Cone()

        b = v.Box(pos=(0,0,0), length=8, width=9, height=7).alpha(0.1)
        axs = v.Axes(b, xyPlaneColor="White")
        self.vp += axs
        #self.vp.show()

        doc = v.Text2D(__doc__, pos="bottom-right")

        ### SE ÖVER
        self.vp.show(doc, at=0)

        #self.vp.show(def_elements, __doc__,
        #     axes=dict(zLabelSize=.04, numberOfDivisions=10),
        #     elevation=-80, bg='blackboard',
        #).close()
        

        self.layout.addWidget(self.vtkWidget)
        #self.layout.addWidget(button)
        self.frame.setLayout(self.layout)
        self.setCentralWidget(self.frame)

        

    def render_geometry(self,nodes,elements,normal_stresses):
        
        nnode = np.size(nodes, axis = 0)
        for i in range(nnode):
            self.vp += nodes[i]

        nel = np.size(elements, axis = 0)
        for i in range(nel):
            self.vp += elements[i]
                       # <--- show the vedo rendering


        #self.show(elements, __doc__,
        #     axes=dict(zLabelSize=.04, numberOfDivisions=10),
        #     elevation=-80, bg='blackboard',
        #).close()

        #def_elements[i].pointdata["myzscalars"] = def_elements[i].points()[:, 2]
        #def_elements[i].pointdata["myzscalars"] = es
        #print(def_elements[i].points()[:, 2])
        #print(es[:,0])
        """
        n = es[:,0]
        vy = es[:,1]
        vz = es[:,2]
        t = es[:,3]
        my = es[:,4]
        mz = es[:,5]
        """
        print(normal_stresses)
        #print(normal_stresses[:, 0])
        #print(elements[i].points())

        for i in range(nel):
            #elements[i].pointdata["myzscalars"] = np.transpose(normal_stresses[:, 0])
            elements[i].pointdata["myzscalars"] = normal_stresses[2*i:2*i+2]
            #elements[i].pointdata["myzscalars"] = elements[i].points()[:, 2]
            elements[i].cmap("jet", "myzscalars", on="points")
        
        #def_elements[i].cmap("jet", "myzscalars", on="points")

        self.show()

        #self.vtkWidget.addScalarBar().show(axes=1)

        # Set-up the rest of the Qt window
        #button = Qt.QPushButton("My Button makes the cone red")
        #button.setToolTip('This is an example button')
        #button.clicked.connect(self.onClick)

    #def render_def_geometry(self,def_nodes,def_elements,stresses=None):
        
    #    nnode = np.size(def_nodes, axis = 0)
    #    for i in range(nnode):
    #        self.vp += def_nodes[i]

    #    nel = np.size(def_elements, axis = 0)
    #    for i in range(nel):
    #        self.vp += def_elements[i]
        
    #def render(self):
        #e = v.Volume(dataurl+"embryo.tif").isosurface()
        #e.normalize().shift(-2,-1.5,-2).c("gold")
        #self.show(e, __doc__, viewup='z')                     # <--- show the Qt Window
        #self.show()

    #def onMouseClick(self, evt):
    #    printc("You have clicked your mouse button. Event info:\n", evt, c='y')

    #def onKeypress(self, evt):
    #    printc("You have pressed key:", evt.keyPressed, c='b')

    #@Qt.pyqtSlot()
    #def onClick(self):
    #    printc("..calling onClick")
    #    self.vp.actors[0].color('red').rotateZ(40)
    #    self.vp.interactor.Render()

    def onClose(self):
        #Disable the interactor before closing to prevent it
        #from trying to act on already deleted items
        printc("..calling onClose")
        self.vtkWidget.close()


class beam3d:
    def geometry(edof,coord,dof):
        
        nnode = np.size(coord, axis = 0)
        nodes = []
        for i in range(nnode):
            nodes.append(v.Sphere().scale(0.03).pos([coord[i,0],coord[i,1],coord[i,2]]).alpha(0.2))
        #print(coord)

        nel = np.size(edof, axis = 0)
        elements = []
        scale = 0.02
        for i in range(nel):
            coord1,coord2 = tools.get_coord_from_edof(edof[i,:],dof)
            dx = coord[coord2,0]-coord[coord1,0]
            dy = coord[coord2,1]-coord[coord1,1]
            dz = coord[coord2,2]-coord[coord1,2]
            x = coord[coord1,0] + 0.5*dx
            y = coord[coord1,1] + 0.5*dy
            z = coord[coord1,2] + 0.5*dz
            h = np.sqrt(dx*dx+dy*dy+dz*dz)

            #elements.append(v.Cylinder(height=h,scale=0.02))
            #elements.append(v.Cylinder(height=h/scale).scale(scale).pos([x,y,z]).orientation([dx,dy,dz]).alpha(0.2))
            elements.append(v.Line(p0=[x-0.5*dx,y-0.5*dy,z-0.5*dz], p1=[x+0.5*dx,y+0.5*dy,z+0.5*dz],closed=True).alpha(0.2).renderLinesAsTubes(True))
            #c1.orientation([1,1,1], rotation=20).pos([2,2,0])

        return nodes, elements

    def def_geometry(edof,coord,dof,a,def_scale):

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

            def_nodes.append(v.Sphere().scale(0.03).pos([x,y,z]))


        nel = np.size(edof, axis = 0)
        def_elements = []
        scale = 0.02

        #def_elements = v.Lines()

        for i in range(nel):
            coord1,coord2 = tools.get_coord_from_edof(edof[i,:],dof)

            dx = def_coord[coord2,0]-def_coord[coord1,0]
            dy = def_coord[coord2,1]-def_coord[coord1,1]
            dz = def_coord[coord2,2]-def_coord[coord1,2]
            x = def_coord[coord1,0] + 0.5*dx
            y = def_coord[coord1,1] + 0.5*dy
            z = def_coord[coord1,2] + 0.5*dz
            h = np.sqrt(dx*dx+dy*dy+dz*dz)

            #def_elements.append(v.Cylinder(height=h/scale).scale(scale).pos([x,y,z]).orientation([dx,dy,dz]).addElevationScalars(lowPoint=(0,0,0), highPoint=(1,1,1)))
            #def_elements.append(v.Cylinder(height=h/scale).scale(scale).pos([x,y,z]).orientation([dx,dy,dz]))
            #def_elements[i].addScalarBar().show(axes=1)


            #Stresses
            lut = v.buildLUT([
                #(-2, 'pink'      ),  # up to -2 is pink
                (0.0, 'pink'      ),  # up to 0 is pink
                (0.4, 'green', 0.5),  # up to 0.4 is green with alpha=0.5
                (0.7, 'darkblue'  ),
                #( 2, 'darkblue'  ),
               ],
               vmin=-1.2, belowColor='lightblue',
               vmax= 0.7, aboveColor='grey',
               nanColor='red',
               interpolate=False,
              )
            #data = es
            #mesh = v.mesh.cmap(lut, data).addScalarBar3D(title='My 3D scalarbar', c='white')

            #mesh.scalarbar.scale(1.5).rotateX(90).y(1) # make it bigger and place it
            #scalars = es # let element stresses be the scalar
            #vmin, vmax = np.min(es), np.max(es)
            #cmap = cmap(stresses, input_array=None, on='points', arrayName='Stresses', vmin=vmin, vmax=vmax, alpha=1, n=256)
            #col = v.colors.colorMap([vmin,0,vmax], name='jet', vmax=vmax, vmin=vmin)

            ### Gamla
            #def_elements.append(v.Cylinder(height=h/scale).scale(scale).pos([x,y,z]).orientation([dx,dy,dz]))

            def_elements.append(v.Line(p0=[x-0.5*dx,y-0.5*dy,z-0.5*dz], p1=[x+0.5*dx,y+0.5*dy,z+0.5*dz],closed=True).renderLinesAsTubes(True))
            #def_elements[i].cmap(lut, data).addScalarBar3D(title='My 3D scalarbar', c='white')
            #def_elements[i].scalarbar.scale(1.5).rotateX(90).y(1) # make it bigger and place it
            #msg1 = v.Text2D("Scalar originally defined on points..", pos="top-center")

            ### SE ÖVER
            #def_elements[i].pointdata["myzscalars"] = def_elements[i].points()[:, 2]
            #def_elements[i].pointdata["myzscalars"] = es
            #print(def_elements[i].points()[:, 2])
            #def_elements[i].cmap("jet", "myzscalars", on="points")
            
        
        #lut = vtk.vtkLookupTable()
        #lut.SetTableRange(vmin, vmax)
        #n = len(scalars)
        #lut.SetNumberOfTableValues(n)
        #lut.Build()

        return def_nodes, def_elements

    def el_values(edof,es,edi,eci,e,v,a,iy,iz,hy,hz):
        #ntot = np.size(es, axis = 0)
        nel = np.size(edof, axis = 0)
        nseg = np.int32(np.size(es, axis = 0)/nel)

        #print(ntot)
        #print(nel)
        #print(n)

        n = es[:,0]
        vy = es[:,1]
        vz = es[:,2]
        t = es[:,3]
        my = es[:,4]
        mz = es[:,5]

        normal_stresses = np.zeros((nel*nseg,1))

        for i in range(nel*nseg):
            stress = n[i]/a + my[i]/iy*hz + mz[i]/iz*hy
            normal_stresses[i] = stress

        data.beam3d_stresses = normal_stresses

        #print(normal_stresses)
        #return normal_stresses






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

    #def get_coord_from_global()




class data:
    beam3d_stresses = None