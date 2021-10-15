# -*- coding: utf-8 -*-
"""
CALFEM VTK

Contains all the functions for 3D visualization in CALFEM

@author: Andreas Åmand
"""

import numpy as np
import vtk


#import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import uic
from PyQt5 import QtCore, QtGui, QtWidgets
import PyQt5
#from PyQt5.QtWidgets import QApplication, QMainWindow
#from PyQt5 import QtGui
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

#class vtk_rendering:

"""
    Function to create essential VTK objects
"""
def vtk_initialize(self):
    # Load UI
    uic.loadUi("QtVTKMainWindow.ui", self)
    
    # Create container/frame
    self.vl = QtWidgets.QVBoxLayout()
    self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
    self.vtkWidget.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    #self.vtkWidget = vtk.vtkRenderWindowInteractor()
    self.vl.addWidget(self.vtkWidget)
    
    # Create renderer & render window
    self.ren = vtk.vtkRenderer()
    self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
    self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
    
    #self.defaultview = self.ren.GetActiveCamera().GetViewUp()
    #self.ren.GetActiveCamera().SetViewUp(1,1,1)
    
    
    self.reset.clicked.connect(lambda: reset_camera(self))
    
    #self.reset.clicked.connect(reset_camera(self))
    
    #self.joystick.stateChanged.connect(mode_joystick(self))
    
    self.statusBar().showMessage('VTK renderer initialized')
    






    
    
"""
    Functions to create VTK actors, mappers & color profiles
""" 
def vtk_mapper1(source):
    # Create a mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(source.GetOutputPort())
    return mapper

def vtk_mapper2(source):
    # Create a mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(source)
    return mapper

def vtk_actor(mapper):
    # Create an actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetLineWidth(4)
    return actor 
    
def vtk_colors(renderer):
    colors = vtk.vtkNamedColors()
    renderer.SetBackground(colors.GetColor3d("black"))

def vtk_widget(vl,render_window):
    vtk_widget = QVTKRenderWindowInteractor(rw=render_window)
    vl.addWidget(vtk_widget)
    vtk_widget.Initialize()
    #vtk_widget.Start()





"""
    Function to create a widget showing axis of orientation in lower right corner
""" 
def MakeAxesActor(scale, xyzLabels):
    axes = vtk.vtkAxesActor()
    axes.SetScale(scale[0], scale[1], scale[2])
    axes.SetShaftTypeToCylinder()
    axes.SetXAxisLabelText(xyzLabels[0])
    axes.SetYAxisLabelText(xyzLabels[1])
    axes.SetZAxisLabelText(xyzLabels[2])
    axes.SetCylinderRadius(0.5 * axes.GetCylinderRadius())
    axes.SetConeRadius(1.025 * axes.GetConeRadius())
    axes.SetSphereRadius(1.5 * axes.GetSphereRadius())
    tprop = axes.GetXAxisCaptionActor2D().GetCaptionTextProperty()
    tprop.ItalicOn()
    tprop.ShadowOn()
    tprop.SetFontFamilyToTimes()
    # Use the same text properties on the other two axes.
    axes.GetYAxisCaptionActor2D().GetCaptionTextProperty().ShallowCopy(tprop)
    axes.GetZAxisCaptionActor2D().GetCaptionTextProperty().ShallowCopy(tprop)
    return axes
    
    
    

"""
    Buttons
"""

# Reset camera
def reset_camera(self):
    print("reset camera")
    
    self.ren.ResetCamera()
    #self.ren.GetActiveCamera().SetViewUp(self.defaultview)
    #self.ren.GetActiveCamera().SetViewUp(1,1,1)
    #self.ren.GetActiveCamera().OrthogonalizeViewUp()
    #self.ren.UpdateViewport()
    #self.ren.GetActiveCamera().SetRoll(0)
    #self.ren.GetActiveCamera().SetViewAngle(0)
    #self.ren.GetActiveCamera().Yaw(90)
    self.iren.Render()
    #self.ren.Transparent()

"""
    Checkboxes, rendering
"""

# Shows a wireframe model
def show_wireframe():
    print("show wireframe")

# Shows a grid in given plane
def show_grid():
    print("show grid")
    
# Show xyz-axis
def show_axis():
    print("show axis")
    
    
"""
    Checkboxes, interaction modes
"""
    
# Joystick Mode
def mode_joystick(self):
    print("mode joystick")
    self.vtkWidget.SetInteractorStyle(vtk.vtkInteractorStyleJoystickCamera())
    #if self.joystick.isChecked():
    #    self.vtkWidget.SetInteractorStyle(vtk.vtkInteractorStyleJoystickCamera())
    #else:
    #    self.vtkWidget.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    
# Actor modes
def mode_actor():
    print("mode actor")









    
"""
    Depreciated functions
"""   
    
"""
def vtk_container(self):
    uic.loadUi("qtwindow.ui", self)
    #self.frame = QtWidgets.QFrame()
    #self.setCentralWidget(self.frame)
    #self.frame = QWidget()
    #self.frame = rendering()
    #self.vl = QVBoxLayout(self.frame)
    self.vl = QtWidgets.QVBoxLayout()
    self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
    self.vl.addWidget(self.vtkWidget)
    #self.container = QWidget()
    #self.vl = QVBoxLayout(self.container)
    #self.setCentralWidget(self.container)
    #self.resize(1200, 800)
    #self.resize(640, 480)
    return self.frame,self.vl

def vtk_importer():
    importer = vtk.vtkGLTFImporter()
    #importer.setFileName("qtwindow.ui")
    importer.SetFileName("qtwindow.ui")
    importer.Read()
    return importer

def vtk_renderer(self):
    self.ren = vtk.vtkRenderer()
    self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
    self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
    #renderer = importer.GetRenderer()
    #render_window = importer.GetRenderWindow()
    #return renderer,render_window
    return self.ren,self.iren
"""

"""

"""    



"""
# Fungerar inte i nuläget
def render(actor):
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(vtk.namedColors.GetColor3d("SlateGray"))

    window = vtk.vtkRenderWindow()
    window.SetWindowName("ColoredLines")
    window.AddRenderer(renderer)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)

    window.Render()
    interactor.Start()

# Fungerar inte i nuläget
def orientation():
    linesPolyData = vtk.vtkPolyData()

    # Create three points
    origin = [0.0, 0.0, 0.0]
    p0 = [0.5, 0.0, 0.0]
    p1 = [0.0, 0.5, 0.0]
    p2 = [0.0, 0.0, 0.5]
    
    pts = vtk.vtkPoints()
    pts.InsertNextPoint(origin)
    pts.InsertNextPoint(p0)
    pts.InsertNextPoint(p1)
    pts.InsertNextPoint(p2)
    
    line0 = vtk.vtkLine()
    line0.GetPointIds().SetId(0, 0)
    line0.GetPointIds().SetId(1, 1)

    line1 = vtk.vtkLine()
    line1.GetPointIds().SetId(0, 0)
    line1.GetPointIds().SetId(1, 2)
    
    line2 = vtk.vtkLine()
    line2.GetPointIds().SetId(0, 0)
    line2.GetPointIds().SetId(1, 3)
    
    lines = vtk.vtkCellArray()
    lines.InsertNextCell(line0)
    lines.InsertNextCell(line1)
    lines.InsertNextCell(line2)
    
    linesPolyData.SetLines(lines)
    
    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    
    try:
        colors.InsertNextTupleValue(vtk.namedColors.GetColor3ub("Red"))
        colors.InsertNextTupleValue(vtk.namedColors.GetColor3ub("Blue"))
        colors.InsertNextTupleValue(vtk.namedColors.GetColor3ub("Green"))
    except AttributeError:
        colors.InsertNextTypedTuple(vtk.namedColors.GetColor3ub("Red"))
        colors.InsertNextTypedTuple(vtk.namedColors.GetColor3ub("Blue"))
        colors.InsertNextTypedTuple(vtk.namedColors.GetColor3ub("Green"))
        
    linesPolyData.GetCellData().SetScalars(colors)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(linesPolyData)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetLineWidth(4)

# Ska visa en grid i xz-planet/annat plan
def grid(x=None,y=None,z=None):
    rgrid = vtk.vtkRectangularGrid()
    
def bc(xu=0,yv=0,zw=0,xr=0,yr=0,zr=0,coord=[None,None,None]):
    print('test')
"""