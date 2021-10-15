# -*- coding: utf-8 -*-
"""
CALFEM VTK

Contains all the functions for 3D visualization in CALFEM

@author: Andreas Åmand
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


class beam:
    def set_geometry(edof,coord,dof):
        linesPolyData = vtk.vtkPolyData()
        pts = vtk.vtkPoints()
        lines = vtk.vtkCellArray()

        spheres = []
        ncoord = np.size(coord, axis = 0)
        for i in range(ncoord):
            pts.InsertNextPoint(coord[i])
            spheres.append(vtk.vtkSphereSource())
            spheres[i].SetCenter(coord[i])
            spheres[i].SetRadius(0.05)
        linesPolyData.SetPoints(pts)

        nel = np.size(edof, axis = 0)
        line = [] 
        for i in range(nel):
            line.append(vtk.vtkLine())
            line[i].GetPointIds().SetId(0, i)
            line[i].GetPointIds().SetId(0, i+1)
            lines.InsertNextCell(line[i])
        linesPolyData.SetLines(lines)

        return linesPolyData,spheres


class MainWindow(QMainWindow):
    
    def __init__(self,linesPolyData=None,spheres=None):
    #def __init__(beam):
        #self.model = beam
        
        super().__init__()
        #coord,dof,edof = model.model()

        
        #print(coord)
        #self.init_gui()
        self.vtk_initialize()
        #linesPolyData,spheres = self.vtk_actor(coord,dof,edof)
        #if linesPolyData != None & spheres != None:
            
        if linesPolyData is None:
            self.statusBar().showMessage('VTK renderer initialized, no input data...')
        else:
            self.vtk_render(linesPolyData,spheres)

    #def init_gui(self):

    def vtk_initialize(self):
        # Load UI
        uic.loadUi("QtVTKMainWindow.ui", self)

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


        # ----- File buttons -----
        # ----- Select buttons -----
        # ----- View buttons -----

        # Reset camera
        self.actionReset_Camera.triggered.connect(lambda: self.reset_camera())

        # Show axis
        self.actionShow_Axis.triggered.connect(lambda: self.show_axis())

        # Show grid
        self.actionShow_Grid.triggered.connect(lambda: self.show_grid())

        # Wireframe
        self.actionWireframe.triggered.connect(lambda: self.show_wireframe())

        # ----- Mode buttons -----
        # ----- Help buttons -----
        
        # Documentation
        #self.actionCALFEM_for_Python_documentation.triggered.connect(lambda: QDesktopServices.openUrl("https://calfem-for-python.readthedocs.io/en/latest/"))
        self.actionCALFEM_for_Python_documentation.triggered.connect(lambda: webbrowser.open('https://calfem-for-python.readthedocs.io/en/latest/'))
        
        #self.defaultview = self.ren.GetActiveCamera().GetViewUp()
        #self.ren.GetActiveCamera().SetViewUp(1,1,1)
        
        
        #self.reset.clicked.connect(lambda: reset_camera(self))
        
        #self.reset.clicked.connect(reset_camera(self))
        
        #self.joystick.stateChanged.connect(mode_joystick(self))
        
        self.statusBar().showMessage('VTK renderer initialized')
        
    def vtk_render(self,linesPolyData,spheres):
        #mapper = cfvv.vtk_mapper_lines(linesPolyData)

        

        # Line actors for elements with 2 nodes
        actor = self.vtk_actor_lines(linesPolyData)
        self.ren.AddActor(actor)

        # Node actors for elements with 2 nodes
        nsph = np.size(spheres, axis = 0)
        for i in range(nsph):
            sphere_actor = self.vtk_actor_objects(spheres[i])
            sphere_actor.GetProperty().SetColor(self.colors.GetColor3d('Red'))
            self.ren.AddActor(sphere_actor)

        
        # Setting frame, starting render
        #self.frame.setLayout(self.vl)
        #self.ren.ResetCamera()
        #self.iren.Start()
        
        self.reset_camera()
        

        self.statusBar().showMessage('Objects rendered')

    def vtk_actor_objects(self,source):
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetLineWidth(4)
        return actor

    def vtk_actor_lines(self,source):
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(source)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetLineWidth(4)
        return actor
        
    # Create a widget showing axis of orientation in lower right corner
    def axesWidget(self, scale=None, xyzLabels=None):
        scale = [1.0, 1.0, 1.0]
        xyzLabels = ['X', 'Y', 'Z']
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
        axes.GetYAxisCaptionActor2D().GetCaptionTextProperty().ShallowCopy(tprop)
        axes.GetZAxisCaptionActor2D().GetCaptionTextProperty().ShallowCopy(tprop)

        # Skapa orientation marker
        self.om = vtk.vtkOrientationMarkerWidget()
        self.om.SetOrientationMarker(axes)
        self.om.SetViewport(0.8, 0, 1.0, 0.2)
        self.om.SetInteractor(self.iren)
        self.om.EnabledOn()
        self.om.InteractiveOn()
        #return axes

    def orientation(self):
        axes = vtk.vtkAxes()
        axes.SetOrigin(0, 0, 0)
        axesMapper = vtk.vtkPolyDataMapper()
        axesMapper.SetInputConnection(axes.GetOutputPort())
        axesActor = vtk.vtkActor()
        axesActor.SetMapper(axesMapper)

        # Create the 3D text and the associated mapper and follower (a type of actor).  Position the text so it is displayed over the origin of the axes.
        atext = vtk.vtkVectorText()
        atext.SetText('Origin')
        textMapper = vtk.vtkPolyDataMapper()
        textMapper.SetInputConnection(atext.GetOutputPort())
        textActor = vtk.vtkFollower()
        textActor.SetMapper(textMapper)
        textActor.SetScale(0.2, 0.2, 0.2)
        textActor.AddPosition(0, -0.1, 0)
        textActor.GetProperty().SetColor(self.colors.GetColor3d('White'))

        self.ren.AddActor(axesActor)
        self.ren.AddActor(textActor)

    # Reset camera
    def reset_camera(self):
        print("reset camera")
        # återställer zoom & position
        self.ren.ResetCamera()
        #self.ren.GetActiveCamera().SetViewUp(self.defaultview)
        #self.ren.GetActiveCamera().SetViewUp(1,1,1)
        #self.ren.GetActiveCamera().OrthogonalizeViewUp()
        #self.ren.UpdateViewport()
        #self.ren.GetActiveCamera().SetRoll(0)
        #self.ren.GetActiveCamera().SetViewAngle(0)
        #self.ren.GetActiveCamera().Yaw(90)
        #self.ren.GetActiveCamera().Azimuth(45)
        #self.ren.GetActiveCamera().Pitch(-45)
        #self.ren.GetActiveCamera().SetViewAngle(45)
        #self.ren.GetActiveCamera().SetEyeAngle(45) # återställer zoom?
        #self.ren.GetActiveCamera().SetFocalPoint(0,0,0)
        #self.ren.GetActiveCamera().SetViewUp(-0.3,-0.3,-0)
        #self.ren.GetActiveCamera().SetPosition(0,0,0)
        self.iren.Render()
        #self.ren.Transparent()

    # Shows a wireframe model
    def show_wireframe():
        print("show wireframe")

    # Shows a grid in given plane
    def show_grid():
        print("show grid")
        
    # Show xyz-axis
    def show_axis(self):
        print("show axis")
        if self.om.EnabledOn():
            self.om.EnabledOff()
        else:
            self.om.EnabledOn()
        
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