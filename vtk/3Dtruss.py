#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 14:08:07 2021

@author: Andreas Åmand
"""

import sys
import vtk
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import uic
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import calfem.core as cfc
import numpy as np
import vis_vtk as cfvv
#from ui_form import UI_Form

#from vtkmodules.vtkCommonDataModel import (
#    vtkCellArray,
#    vtkLine,
#    vtkPolyData
#)




#class MainWindow(QWidget):
class MainWindow(QMainWindow):
#class MainWindow(QtWidgets.QWidget):
    
    def __init__(self, parent=None):
        #QtGui.QMainWindow.__init__(self, parent)
        
        super().__init__()

        self.init_gui()
    """    
    def init_gui(self):
        #uic.loadUi("qtwindow.ui", self)
        #self.frame = QtWidgets.QFrame()
        #self.frame = QWidget()
        #self.frame = rendering()

        #self.vl = QtWidgets.QVBoxLayout()
        self.vl = QVBoxLayout(self.frame)
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.vl.addWidget(self.vtkWidget)

        self.ren = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()

        # Create source
        source = vtk.vtkSphereSource()
        source.SetCenter(0, 0, 0)
        source.SetRadius(5.0)

        # Create a mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())

        # Create an actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        self.ren.AddActor(actor)

        self.ren.ResetCamera()

        self.frame.setLayout(self.vl)
        self.setCentralWidget(self.frame)
        #self.SetCenter(self.frame)
        #self.setCentralWidget(self.frame)
        
        self.setWindowTitle("Main Window")

        self.show()
        self.iren.Initialize()
    """


    def init_gui(self):
        #self.setGeometry(190, 300, 300, 200)
        #self.setWindowTitle("Test")
        #uic.loadUi("qtwindow.ui", self)
        # NEDANSTÅENDE RAD FUNKAR
        #container,vl = cfvv.vtk_container(self)
        #importer = cfvv.vtk_importer()
        #renderer,render_window = cfvv.vtk_renderer(importer)
        # NEDANSTÅENDE RAD FUNKAR
        #renderer,render_window = cfvv.vtk_renderer(self)
        
        cfvv.vtk_initialize(self)
        
        # Interaktion
        #self.reset.clicked.connect(cfvv.reset_camera(self.renderer))
        
        
        self.actor.stateChanged.connect(cfvv.mode_actor)
        self.axis.stateChanged.connect(cfvv.show_axis)
        self.grid.stateChanged.connect(cfvv.show_grid)
        self.wireframe.stateChanged.connect(cfvv.show_wireframe)
        
        #cfvv.orientation()
        
        # Create source
        ball = vtk.vtkSphereSource()
        ball.SetCenter(0, 0, 0)
        ball.SetRadius(5.0)
        
        
        
        
        linesPolyData = vtk.vtkPolyData()
        
        origin = [0.0, 0.0, 0.0]
        p0 = [5.5, 0.0, 0.0]
        p1 = [0.0, 5.5, 0.0]
        p2 = [0.0, 0.0, 5.5]
        
        pts = vtk.vtkPoints()
        pts.InsertNextPoint(origin)
        pts.InsertNextPoint(p0)
        pts.InsertNextPoint(p1)
        pts.InsertNextPoint(p2)
        
        linesPolyData.SetPoints(pts)
        
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
        
        
        axes = cfvv.MakeAxesActor([1.0, 1.0, 1.0], ['X', 'Y', 'Z'])
        om2 = vtk.vtkOrientationMarkerWidget()
        om2.SetOrientationMarker(axes)
        # Position lower right in the viewport.
        om2.SetViewport(0.8, 0, 1.0, 0.2)
        om2.SetInteractor(self.iren)
        om2.EnabledOn()
        om2.InteractiveOn()
        
        
        
        
        mapper1 = cfvv.vtk_mapper1(ball)
        mapper2 = cfvv.vtk_mapper2(linesPolyData)
        #mapper.SetInputData(linesPolyData)
        
        actor1 = cfvv.vtk_actor(mapper1)
        actor2 = cfvv.vtk_actor(mapper2)
        
        self.ren.AddActor(actor1)
        self.ren.AddActor(actor2)
        #cfvv.vtk_widget(vl,render_window)
        
        #self.renderer.ResetCamera()

        self.frame.setLayout(self.vl)
        #self.setCentralWidget(container)
        
        
        
        
        
        
        
        

        self.show()
        self.iren.Initialize()
        
        
        
        #self.vtkWidget.SetSize(600, 600)
        #self.vtkWidget.Render()
        #self.iren.Start()
        
        
        self.ren.GetActiveCamera().Azimuth(45)
        self.ren.GetActiveCamera().Pitch(-22.5)
        self.ren.ResetCamera()




class truss:
    def main():
        
        """
        K = np.zeros([114,114])
        f = np.zeros([114,1])
        f[8,0] = -3000
        f[14,0] = -3000
        f[20,0] = -3000
        f[26,0] = -3000
        f[32,0] = -3000
        f[50,0] = -3000
        f[56,0] = -3000
        f[62,0] = -3000
        f[68,0] = -3000
        f[74,0] = -3000
        print(K)

        edof_beams = np.array([
            # vänster sida längst ner
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 
            [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
            [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
            [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
            [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36],
            [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42],
            # höger sida längst ner
            [43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
            [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
            [55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66],
            [61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72],
            [67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78],
            [73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84],
            # botten mellan vänster & höger
            [7, 8, 9, 10, 11, 12, 49, 50, 51, 52, 53, 54], 
            [13, 14, 15, 16, 17, 18, 55, 56, 57, 58, 59, 60],
            [19, 20, 21, 22, 23, 24, 61, 62, 63, 64, 65, 66],
            [25, 26, 27, 28, 29, 30, 67, 68, 69, 70, 71, 72],
            [31, 32, 33, 34, 35, 36, 73, 74, 75, 76, 77, 78]
        ])

        edof_bars = np.array([
            # vänster sida, vertikala/sneda
            [1, 2, 3, 85, 86, 87],
            [7, 8, 9, 85, 86, 87],
            [13, 14, 15, 85, 86, 87],
            [13, 14, 15, 88, 89, 90],
            [19, 20, 21, 88, 89, 90],
            [19, 20, 21, 91, 92, 93],
            [19, 20, 21, 94, 95, 96],
            [25, 26, 27, 94, 95, 96],
            [25, 26, 27, 97, 98, 99],
            [31, 32, 33, 97, 98, 99],
            [37, 38, 39, 97, 98, 99],
            # vänster sida, överst
            [85, 86, 87, 88, 89, 90],
            [88, 89, 90, 91, 92, 93],
            [91, 92, 93, 94, 95, 96],
            [94, 95, 96, 97, 98, 99],
            # höger sida, vertikala/sneda
            [43, 44, 45, 100, 101, 102],
            [49, 50, 51, 100, 101, 102],
            [55, 56, 57, 100, 101, 102],
            [55, 56, 57, 103, 104, 105],
            [61, 62, 63, 103, 104, 105],
            [61, 62, 63, 106, 107, 108],
            [61, 62, 63, 109, 110, 111],
            [67, 68, 69, 109, 110, 111],
            [67, 68, 69, 112, 113, 114],
            [73, 74, 75, 112, 113, 114],
            [79, 80, 81, 112, 113, 114],
            # höger sida, överst
            [100, 101, 102, 103, 104, 105],
            [103, 104, 105, 103, 104, 105],
            [106, 107, 108, 109, 110, 111],
            [109, 110, 111, 112, 113, 114],
            # överst mellan vänster & höger
            [85, 86, 87, 100, 101, 102],
            [88, 89, 90, 103, 104, 105],
            [91, 92, 93, 106, 107, 108],
            [94, 95, 96, 109, 110, 111],
            [97, 98, 99, 112, 113, 114]
        ])

        E = 210000000
        v = 0.3
        G = E/(2*(1+v))
        A_beam = 11250*0.000001
        A_bar = np.pi*0.05*0.05
        Iy = 63.1*0.000001
        Iz = 182.6*0.000001
        Kv = 0.856*0.000001

        

        #ex_beams = np.zeros([17,2])
        #ey_beams = np.zeros([17,2])
        #ez_beams = np.zeros([17,2])

        #ex_bars = np.zeros([35,2])
        #ey_bars = np.zeros([35,2])
        #ez_bars = np.zeros([35,2])


        ex_beams = np.array([
            # vänster sida längst ner
            [0, 3],
            [3, 6],
            [6, 9],
            [9, 12],
            [12, 15],
            [15, 18],
            # höger sida längst ner
            [0, 3],
            [3, 6],
            [6, 9],
            [9, 12],
            [12, 15],
            [15, 18],
            # botten mellan vänster & höger
            [3, 3],
            [6, 6],
            [9, 9],
            [12, 12],
            [15, 15]
        ])
        ey_beams = np.zeros([17,2])
        ez_beams = np.array([
            # vänster sida längst ner
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            # höger sida längst ner
            [3, 3],
            [3, 3],
            [3, 3],
            [3, 3],
            [3, 3],
            [3, 3],
            # botten mellan vänster & höger
            [0, 3],
            [0, 3],
            [0, 3],
            [0, 3],
            [0, 3]
        ])

        eo_beams = np.array([
            # vänster sida längst ner
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            # höger sida längst ner
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            # botten mellan vänster & höger
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1]
        ])

        ex_bars = np.array([
            # vänster sida, vertikala/sneda
            [0, 3],
            [3, 3],
            [6, 3],
            [6, 6],
            [9, 6],
            [9, 9],
            [9, 12],
            [12, 12],
            [12, 15],
            [15, 15],
            [18, 15],
            # vänster sida, överst
            [3, 6],
            [6, 9],
            [9, 12],
            [12, 15],
            # höger sida, vertikala/sneda
            [0, 3],
            [3, 3],
            [6, 3],
            [6, 6],
            [9, 6],
            [9, 9],
            [9, 12],
            [12, 12],
            [12, 15],
            [15, 15],
            [18, 15],
            # höger sida, överst
            [3, 6],
            [6, 9],
            [9, 12],
            [12, 15],
            # överst mellan vänster & höger
            [3, 3],
            [6, 6],
            [9, 9],
            [12, 12],
            [15, 15]
        ])

        ey_bars = np.array([
            # vänster sida, vertikala/sneda
            [0, 3],
            [0, 3],
            [0, 3],
            [0, 3],
            [0, 3],
            [0, 3],
            [0, 3],
            [0, 3],
            [0, 3],
            [0, 3],
            [0, 3],
            # vänster sida, överst
            [3, 3],
            [3, 3],
            [3, 3],
            [3, 3],
            # höger sida, vertikala/sneda
            [0, 3],
            [0, 3],
            [0, 3],
            [0, 3],
            [0, 3],
            [0, 3],
            [0, 3],
            [0, 3],
            [0, 3],
            [0, 3],
            [0, 3],
            # höger sida, överst
            [3, 3],
            [3, 3],
            [3, 3],
            [3, 3],
            # överst mellan vänster & höger
            [3, 3],
            [3, 3],
            [3, 3],
            [3, 3],
            [3, 3]
        ])

        ez_bars = np.array([
            # vänster sida, vertikala/sneda
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            # vänster sida, överst
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            # höger sida, vertikala/sneda
            [3, 3],
            [3, 3],
            [3, 3],
            [3, 3],
            [3, 3],
            [3, 3],
            [3, 3],
            [3, 3],
            [3, 3],
            [3, 3],
            [3, 3],
            # höger sida, överst
            [3, 3],
            [3, 3],
            [3, 3],
            [3, 3],
            # överst mellan vänster & höger
            [0, 3],
            [0, 3],
            [0, 3],
            [0, 3],
            [0, 3]
        ])

        ep_beams = [E, G, A_beam, Iy, Iz, Kv]
        ep_bars = [E, A_bar]

        #for i in range(8):
        #    Ke = cfc.flw2qe(Ex[i],Ey[i],ep,D)
        #    K = cfc.assem(Edof[i],K,Ke)
        #Ke = np.zeros([12,12])
        for i in range(17):
            Ke = cfc.beam3e(ex_beams[i], ey_beams[i], ez_beams[i], eo_beams[i], ep_beams)
            K = cfc.assem(edof_beams[i],K,Ke)

        #Ke = np.zeros([6,6])
        for i in range(35):
            Ke = cfc.bar3e(ex_bars[i], ey_bars[i], ez_bars[i], ep_bars)
            K = cfc.assem(edof_bars[i],K,Ke)

        # ----- Solve equation system -----

        bcPrescr = np.array([1,2,3,4,5,6,37,38,39,40,41,42,43,44,45,46,47,48,79,80,81,82,83,84])
        #bcVal = np.array([0,0,0,0,0,0,0.5e-3,1e-3,1e-3])
        a = cfc.solveq(K,f,bcPrescr)

        print(a)

        # ----- Compute element flux vector -----

        #Ed = cfc.extractEldisp(Edof,a)
        #Es = np.zeros((8,2))
        #for i in range(8):
        #    Es[i],Et = cfc.flw2qs(Ex[i],Ey[i],ep,D,Ed[i])
"""


coord = np.array([
    [0, 0, 0],
    [3, 0, 0],
    [6, 0, 0],
    [9, 0, 0],
    [3, 3, 0],
    [6, 3, 0],
    [9, 3, 0],
    [0, 0, 3],
    [3, 0, 3],
    [6, 0, 3],
    [9, 0, 3],
    [3, 3, 3],
    [6, 3, 3],
    [9, 3, 3]
])

dof = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12],
    [13, 14, 15],
    [16, 17, 18],
    [19, 20, 21],
    [22, 23, 24],
    [25, 26, 27],
    [28, 29, 30],
    [31, 32, 33],
    [34, 35, 36],
    [37, 38, 39],
    [40, 41, 42]
])

edof = np.array([
    [1, 2, 3, 4, 5, 6],
    [4, 5, 6, 7, 8, 9],
    [7, 8, 9, 10, 11, 12],
    
    [1, 2, 3, 13, 14, 15],
    [4, 5, 6, 13, 14, 15],
    [7, 8, 9, 13, 14, 15],
    [7, 8, 9, 16, 17, 18],
    [10, 11, 12, 16, 17, 18],
    [10, 11, 12, 19, 20, 21],
    
    [13, 14, 15, 16, 17, 18],
    [16, 17, 18, 19, 20, 21],
    
    [22, 23, 24, 25, 26, 27],
    [25, 26, 27, 28, 29, 30],
    [28, 29, 30, 31, 32, 33],
    
    [22, 23, 24, 34, 35, 36],
    [25, 26, 27, 34, 35, 36],
    [28, 29, 30, 34, 35, 36],
    [28, 29, 30, 37, 38, 39],
    [31, 32, 33, 37, 38, 39],
    [31, 32, 33, 40, 41, 42],
     
    [34, 35, 36, 37, 38, 39],
    [37, 38, 39, 40, 41, 42],
    
    [1, 2, 3, 22, 23, 24],
    [4, 5, 6, 25, 26, 27],
    [7, 8, 9, 29, 30, 31],
    [10, 11, 12, 31, 32, 33],
    
    [13, 14, 15, 34, 35, 36],
    [16, 17, 18, 37, 38, 39],
    [19, 20, 21, 40, 41, 42]
])

ex,ey,ez = cfc.coordxtr(edof,coord,dof)





if __name__ == "__main__":
    #app = QtGui.QApplication(sys.argv)
    app = QApplication(sys.argv)
    ex = MainWindow()
    ex.show()
    
    
    
    truss.main()
    

    #window = MainWindow()

    sys.exit(app.exec_())