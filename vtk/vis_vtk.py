# -*- coding: utf-8 -*-
"""
CALFEM VTK

Contains all the functions for 3D visualization in CALFEM using VTK

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

class tools:
    def get_coord_from_edof(edof_row,dof):

        print(edof_row)

        #ndof = np.length

        #edof_row1 = edof_row[0:6]
        #edof_row2 = edof_row[6:12]

        edof_row1,edof_row2 = np.split(edof_row,2)

        #print(edof_row1==dof)
        #print(edof_row2==dof)

        #print(edof_row1)
        #print(edof_row2)

        coord1 = int(np.where(np.all(edof_row1==dof,axis=1))[0])
        coord2 = int(np.where(np.all(edof_row2==dof,axis=1))[0])

        #print(coord1)
        #print(coord2)

        #coord1 = 0
        #coord2 = 1

        return coord1, coord2
    """
    def getTransformFromAxes(xaxis, yaxis, zaxis):
 
        t = vtk.vtkTransform()
        m = vtk.vtkMatrix4x4()
     
        axes = np.array([xaxis, yaxis, zaxis]).transpose().copy()
        vtk.vtkMath.Orthogonalize3x3(axes, axes)
     
        for r in xrange(3):
            for c in xrange(3):
                m.SetElement(r, c, axes[r][c])
     
        t.SetMatrix(m)
        return t
    """


class beam:
    def set_geometry(edof,coord,dof):
        colors = vtk.vtkNamedColors()
        #linesPolyData = vtk.vtkPolyData()
        #pts = vtk.vtkPoints()
        #lines = vtk.vtkCellArray()

        node_actors = []
        ncoord = np.size(coord, axis = 0)
        for i in range(ncoord):
            #pts.InsertNextPoint(coord[i])
            node = vtk.vtkSphereSource()
            node.SetCenter(coord[i])
            node.SetRadius(0.05)

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(node.GetOutputPort())
            node_actors.append(vtk.vtkActor())
            node_actors[i].GetProperty().SetColor(colors.GetColor3d('Red'))
            node_actors[i].SetMapper(mapper)
            


            #node_actor = self.actor(spheres[i])
            
            #self.ren.AddActor(node_actor)
        #linesPolyData.SetPoints(pts)

        element_actors = []
        nel = np.size(edof, axis = 0)
        print(nel)
        for i in range(nel):
            #dofs1 = [edof[i,0], edof[i,1], edof[i,2], edof[i,3], edof[i,4], edof[i,5]]
            #dofs2 = [edof[i,6], edof[i,7], edof[i,8], edof[i,9], edof[i,10], edof[i,11]]
                #print(edof[i,:])
            coord1,coord2 = tools.get_coord_from_edof(edof[i,:],dof)
            #print(coord1)
            #print(coord2)
            element = vtk.vtkCylinderSource()
            element.SetResolution(15)
            element.SetRadius(0.02)

            #dx = coord[i+1,0]-coord[i,0]
            dx = coord[coord2,0]-coord[coord1,0]
            #print(dx)
            #dy = coord[i+1,1]-coord[i,1]
            dy = coord[coord2,1]-coord[coord1,1]
            #print(dy)
            #dz = coord[i+1,2]-coord[i,2]
            dz = coord[coord2,2]-coord[coord1,2]
            #print(dz)
            h = np.sqrt(dx*dx+dy*dy+dz*dz)
            #x = coord[i,0] + 0.5*dx
            #y = coord[i,1] + 0.5*dy
            #z = coord[i,2] + 0.5*dz

            x = coord[coord1,0] + 0.5*dx
            y = coord[coord1,1] + 0.5*dy
            z = coord[coord1,2] + 0.5*dz

            element.SetHeight(h)
            #element.SetCenter(x,y,z)

            #n = vtk.vtkMath.Normalize([dx,dy,dz])
            #print(n)

            
            #Works for horizontal elements
            anglex = vtk.vtkMath.AngleBetweenVectors([0, dy, dz], [0,0,1])*180/np.pi
            #angle = angle_rad*180/np.pi
            #print(anglex)

            angley = vtk.vtkMath.AngleBetweenVectors([dx, 0, dz], [0,0,1])*180/np.pi
            #angle = angle_rad*180/np.pi
            #print(angley)

            anglez = vtk.vtkMath.AngleBetweenVectors([dx, dy, 0], [0,0,1])*180/np.pi
            #angle = angle_rad*180/np.pi
            #print(anglez)
            
            """
            startPoint = [coord[coord1,0], coord[coord1,1], coord[coord1,2]]
            endPoint = [coord[coord2,0], coord[coord2,1], coord[coord2,2]]
            #rng = vtk.vtkMinimalStandardRandomSequence()
            #rng.SetSeed(8775070)  # For testing.
            print(startPoint)
            print(endPoint)

            # Compute a basis
            normalizedX = [0] * 3
            normalizedY = [0] * 3
            normalizedZ = [0] * 3
            print(normalizedX)
            print(normalizedY)
            print(normalizedZ)

            vtk.vtkMath.Subtract(endPoint, startPoint, normalizedX)
            length = vtk.vtkMath.Norm(normalizedX)
            vtk.vtkMath.Normalize(normalizedX)
            print(normalizedX)
            print(normalizedY)
            print(normalizedZ)

            # The Z axis is an arbitrary vector cross X
            #arbitrary = [0] * 3
            #for i in range(0, 3):
            #    rng.Next()
            #    arbitrary[i] = rng.GetRangeValue(-10, 10)
            vtk.vtkMath.Cross(normalizedX, normalizedY, normalizedZ)
            vtk.vtkMath.Normalize(normalizedZ)

            # The Y axis is Z cross X
            vtk.vtkMath.Cross(normalizedZ, normalizedX, normalizedY)

            matrix = vtk.vtkMatrix4x4()

            matrix.Identity()
            for j in range(0, 3):
                matrix.SetElement(j, 0, normalizedX[j])
                matrix.SetElement(j, 1, normalizedY[j])
                matrix.SetElement(j, 2, normalizedZ[j])
            print(matrix)
            """

            """
            Från OrientedCylinder
            # Generate a random start and end point
            startPoint = [0] * 3
            endPoint = [0] * 3
            rng = vtkMinimalStandardRandomSequence()
            rng.SetSeed(8775070)  # For testing.
            for i in range(0, 3):
                rng.Next()
                startPoint[i] = rng.GetRangeValue(-10, 10)
                rng.Next()
                endPoint[i] = rng.GetRangeValue(-10, 10)

            # Compute a basis
            normalizedX = [0] * 3
            normalizedY = [0] * 3
            normalizedZ = [0] * 3

            # The X axis is a vector from start to end
            vtkMath.Subtract(endPoint, startPoint, normalizedX)
            length = vtkMath.Norm(normalizedX)
            vtkMath.Normalize(normalizedX)

            # The Z axis is an arbitrary vector cross X
            arbitrary = [0] * 3
            for i in range(0, 3):
                rng.Next()
                arbitrary[i] = rng.GetRangeValue(-10, 10)
            vtkMath.Cross(normalizedX, arbitrary, normalizedZ)
            vtkMath.Normalize(normalizedZ)

            # The Y axis is Z cross X
            vtkMath.Cross(normalizedZ, normalizedX, normalizedY)
            matrix = vtkMatrix4x4()

            # Create the direction cosine matrix
            matrix.Identity()
            for i in range(0, 3):
                matrix.SetElement(i, 0, normalizedX[i])
                matrix.SetElement(i, 1, normalizedY[i])
                matrix.SetElement(i, 2, normalizedZ[i])

            # Apply the transforms
            transform = vtkTransform()
            transform.Translate(startPoint)  # translate to starting point
            transform.Concatenate(matrix)  # apply direction cosines
            transform.RotateZ(-90.0)  # align cylinder to x axis
            transform.Scale(1.0, length, 1.0)  # scale along the height vector
            transform.Translate(0, .5, 0)  # translate to start of cylinder

            # Transform the polydata
            transformPD = vtkTransformPolyDataFilter()
            transformPD.SetTransform(transform)
            transformPD.SetInputConnection(cylinderSource.GetOutputPort())
            """
            """
            m = vtk.vtkMatrix4x4()

            
            axes = [[dx, dy, dz], [dx, dy, dz], [dx, dy, dz]]
            axes = [[dx, dx, dx], [dy, dy, dy], [dz, dz, dz]]

            #print(axes)
            print(vtk.vtkMath.Orthogonalize3x3(axes, axes))
         
            for r in range(3):
                for c in range(3):
                    m.SetElement(r, c, axes[r][c])
            """
            
            

            #normalizedX = [dx,dx,dx]
            #normalizedY = [dy,dy,dy]
            #normalizedZ = [dz,dz,dz]

            #vtk.vtkMath.Normalize(normalizedX)
            #vtk.vtkMath.Normalize(normalizedY)
            #vtk.vtkMath.Normalize(normalizedZ)

            #vtk.vtkMath.Normalize(dx)
            #vtk.vtkMath.Normalize(dy)
            #vtk.vtkMath.Normalize(dz)

            #vtkMath.Cross(normalizedZ, normalizedX, normalizedY)
            #matrix = vtk.vtkMatrix4x4()
            #matrix.Identity()

            #matrix = [[0.61782155,   0.78631834,  0,      -0.33657291],
            #            [0.78631834,  -0.61782155,  0,       1.04497454],
            #            [0,            0,          -1,       0],
            #            [0,            0,           0,       1]]


            #matrix.SetElement(0, 0, normalizedX[0])
            #matrix.SetElement(0, 1, normalizedY[0])
            #matrix.SetElement(0, 2, normalizedZ[0])
            
            #matrix.SetElement(1, 0, normalizedX[1])
            #matrix.SetElement(1, 1, normalizedY[1])
            #matrix.SetElement(1, 2, normalizedZ[1])

            #matrix.SetElement(2, 0, normalizedX[2])
            #matrix.SetElement(2, 1, normalizedY[2])
            #matrix.SetElement(2, 2, normalizedZ[2])

            #print(matrix)
            #for k in range(0, 3):
            #    matrix.SetElement(k, 0, vtk.vtkMath.Normalize(dx))
            #    matrix.SetElement(k, 1, vtk.vtkMath.Normalize(dy))
            #    matrix.SetElement(k, 2, vtk.vtkMath.Normalize(dz))

            #t = tools.getTransformFromAxes(dx,dy,dz)
            """
            pos = [x,y,z]
            rotation = 0 # around new axis
            initaxis = [0,0,1] # old object's axis
            newaxis = [dx,dy,dz]
            crossvec = np.cross(initaxis, newaxis)
            angle = np.arccos(np.dot(initaxis, newaxis))
            T = vtk.vtkTransform()
            T.PostMultiply()
            #T.Translate(-pos)
            if rotation:
                T.RotateWXYZ(rotation, initaxis)
            T.RotateWXYZ(np.rad2deg(angle), crossvec)
            T.Translate(pos)
            """

            transform = vtk.vtkTransform()
            #transform.RotateWXYZ(90,1,1,1)
            #transform.PreMultiply()
            transform.PostMultiply()
            #transform.SetMatrix(t)
            #transform.Concatenate(matrix)
            
            #Works for horizontal elements
            transform.RotateX(anglez)
            transform.RotateY(anglex)
            transform.RotateZ(angley)
            #transform.PreMultiply()
            #transform.PostMultiply()
            #transform.RotateY(-30)
            #transform.RotateZ(-90)
            
            transform.Translate(x,y,z)
            #transform.Update()
            tf = vtk.vtkTransformPolyDataFilter()
            tf.SetTransform(transform)
            tf.SetInputConnection(element.GetOutputPort())
            tf.Update()

            

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(tf.GetOutputPort())


            element_actors.append(vtk.vtkActor())
            element_actors[i].GetProperty().SetColor(colors.GetColor3d('White'))
            element_actors[i].SetMapper(mapper)


        return element_actors, node_actors


            #element_actor = self.actor(elements[i])
            
            #element_actor.SetAxis(1,1,1)
            #self.ren.AddActor(element_actor)

        #nel = np.size(elements, axis = 0)
            
        #element_actor = self.vtk_actor_objects(linesPolyData)
        #self.ren.AddActor(element_actor)

        # Node actors for elements with 2 nodes
        #nnode = np.size(spheres, axis = 0)
            
    """
    def actor(self,source):
        #mapper = vtk.vtkPolyDataMapper()
        #mapper.SetInputConnection(tf.GetOutputPort())
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        #actor.GetProperty().SetLineWidth(4)
        return actor
    
    #elements[i].ApplyTransform(transform)

    #elements[i].SetAxis(1,1,1)

    #impl = vtk.vtkCylinder()
    #elements[i].SetImplicitFunction(impl)

    #elements[i].getActor.SetAxis(dx,dy,dz)


    #transform = vtk.vtkTransform()

    #transform.Translate(dx, dy, dz)

    #elements[i].SetTransform(transform)

            
    elements.append(vtk.vtkLine())
    elements[i].setLineWidth(10)
    elements[i].GetPointIds().SetId(0, i)
    elements[i].GetPointIds().SetId(0, i+1)
    lines.InsertNextCell(elements[i])
    """     
    #linesPolyData.SetLines(lines)
        
        

        


class MainWindow(QMainWindow):
    
    def __init__(self,elements=None,nodes=None):
    #def __init__(beam):
        #self.model = beam
        
        super().__init__()
        #coord,dof,edof = model.model()

        
        #print(coord)
        #self.init_gui()
        self.vtk_initialize()
        #linesPolyData,spheres = self.vtk_actor(coord,dof,edof)
        #if linesPolyData != None & spheres != None:
            
        if elements is None:
            self.statusBar().showMessage('VTK renderer initialized, no input data...')
        else:
            self.vtk_render(elements, nodes)

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
    """
    def vtk_actor_objects(self,source):
        #mapper = vtk.vtkPolyDataMapper()
        #mapper.SetInputConnection(tf.GetOutputPort())
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        #actor.GetProperty().SetLineWidth(4)
        return actor
    
    def vtk_actor_lines(self,source):
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(source)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        #actor.GetProperty().SetLineWidth(4)
        return actor
    """
    def vtk_render(self,elements,nodes):
        #mapper = cfvv.vtk_mapper_lines(linesPolyData)

        

        # Line actors for elements with 2 nodes
        #actor = self.vtk_actor_lines(linesPolyData)
        """
        nel = np.size(elements, axis = 0)
        for i in range(nel):
            element_actor = self.vtk_actor_objects(elements[i])
            element_actor.GetProperty().SetColor(self.colors.GetColor3d('Blue'))
            #element_actor.SetAxis(1,1,1)
            self.ren.AddActor(element_actor)
        #element_actor = self.vtk_actor_objects(linesPolyData)
        #self.ren.AddActor(element_actor)

        # Node actors for elements with 2 nodes
        nnode = np.size(spheres, axis = 0)
        for i in range(nnode):
            node_actor = self.vtk_actor_objects(spheres[i])
            node_actor.GetProperty().SetColor(self.colors.GetColor3d('Red'))
            self.ren.AddActor(node_actor)

        """
        # Setting frame, starting render
        #self.frame.setLayout(self.vl)
        #self.ren.ResetCamera()
        #self.iren.Start()

        nel = np.size(elements, axis = 0)
        for i in range(nel):
            self.ren.AddActor(elements[i])


        ncoord = np.size(nodes, axis = 0)
        for i in range(ncoord):
            self.ren.AddActor(nodes[i])
        
        self.reset_camera()
        

        self.statusBar().showMessage('Objects rendered')
        
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
        #axesActor.GetProperty().SetColor(self.colors.GetColor3d('Blue'))

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
        self.iren.Render()

    # Shows a wireframe model
    def show_wireframe():
        print("show wireframe")

    # Shows a grid in given plane
    def show_grid():
        print("show grid")
        
    # Show xyz-axis
    def show_axis(self):
        print("show axis")
        #self.om.EnabledOff()
        #if self.om.EnabledOn() is True:
        if self.om.EnabledOn():
            self.om.EnabledOff()
        else:
            self.om.EnabledOn()
        self.iren.Render()
        
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