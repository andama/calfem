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
    
    def __init__(self, nodes, elements, parent=None):
        
        
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
        Qt.QMainWindow.__init__(self, parent)
        self.start(nodes,elements)
        self.render()
    def start(self,nodes,elements):
        uic.loadUi("QtVTKMainWindow.ui", self)
        self.frame = Qt.QFrame()
        self.layout = Qt.QVBoxLayout()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)

        # Create renderer and add the vedo objects and callbacks
        self.vp = v.Plotter(qtWidget=self.vtkWidget,bg2="blackboard",bg="black",axes=4)
        #self.id1 = self.vp.addCallback("mouse click", self.onMouseClick)
        #self.id2 = self.vp.addCallback("key press",   self.onKeypress)
        #self.vp += v.Cone()
        nnode = np.size(nodes, axis = 0)
        for i in range(nnode):
            self.vp += nodes[i]

        nel = np.size(elements, axis = 0)
        for i in range(nel):
            self.vp += elements[i]
        b = v.Box(pos=(0,0,0), length=8, width=9, height=7).alpha(0.1)
        axs = v.Axes(b, xyPlaneColor="White")
        self.vp += axs
        self.vp.show()                  # <--- show the vedo rendering

        # Set-up the rest of the Qt window
        #button = Qt.QPushButton("My Button makes the cone red")
        #button.setToolTip('This is an example button')
        #button.clicked.connect(self.onClick)
        self.layout.addWidget(self.vtkWidget)
        #self.layout.addWidget(button)
        self.frame.setLayout(self.layout)
        self.setCentralWidget(self.frame)
    def render(self):
        #e = v.Volume(dataurl+"embryo.tif").isosurface()
        #e.normalize().shift(-2,-1.5,-2).c("gold")
        #self.show(e, __doc__, viewup='z')                     # <--- show the Qt Window
        self.show() 
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
            nodes.append(v.Sphere().scale(0.03).pos([coord[i,0],coord[i,1],coord[i,2]]))
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
            elements.append(v.Cylinder(height=h/scale).scale(scale).pos([x,y,z]).orientation([dx,dy,dz]))
            #c1.orientation([1,1,1], rotation=20).pos([2,2,0])


        return nodes, elements

class tools:
    def get_coord_from_edof(edof_row,dof):
        edof_row1,edof_row2 = np.split(edof_row,2)
        coord1 = int(np.where(np.all(edof_row1==dof,axis=1))[0])
        coord2 = int(np.where(np.all(edof_row2==dof,axis=1))[0])
        return coord1, coord2