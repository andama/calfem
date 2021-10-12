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
sys.path.append('../')
import vis_vtk as cfvv
import core_beam_extensions as cfcb

import model


class MainWindow(QMainWindow):
    
    def __init__(self, parent=None):
    #def __init__(beam):
        #self.model = beam
        
        super().__init__()
        coord,dof,edof = model.model()

        
        print(coord)
        #self.init_gui()
        cfvv.vtk_initialize(self)
        linesPolyData,spheres = self.vtk_actor(coord,dof,edof)
        self.vtk_render(linesPolyData,spheres)

    #def init_gui(self):



    def vtk_actor(self,coord,dof,edof):
        linesPolyData = vtk.vtkPolyData()
        pts = vtk.vtkPoints()
        lines = vtk.vtkCellArray()

        #source = vtk.vtkSphereSource()
        spheres = []
        ncoord = np.size(coord, axis = 0)
        for i in range(ncoord):
            pts.InsertNextPoint(coord[i])
            spheres.append(vtk.vtkSphereSource())
            spheres[i].SetCenter(coord[i])
            spheres[i].SetRadius(0.05)
        linesPolyData.SetPoints(pts)

        print(pts)

        print(spheres)

        nel = np.size(edof, axis = 0)
        line = [] 
        for i in range(nel):
            line.append(vtk.vtkLine())
            line[i].GetPointIds().SetId(0, i)
            line[i].GetPointIds().SetId(0, i+1)
            lines.InsertNextCell(line[i])
        linesPolyData.SetLines(lines)

        print(lines)

        return linesPolyData,spheres

        #mapper1 = cfvv.vtk_mapper1(ball)
        #mapper2 = cfvv.vtk_mapper2(linesPolyData)
        #mapper.SetInputData(linesPolyData)
        
        #actor1 = cfvv.vtk_actor(mapper1)
        #actor2 = cfvv.vtk_actor(mapper2)
        
        #self.ren.AddActor(actor1)
        #self.ren.AddActor(actor2)
        
        
    def vtk_render(self,linesPolyData,spheres):
        #mapper = cfvv.vtk_mapper_lines(linesPolyData)
        actor = cfvv.vtk_actor_lines(linesPolyData)

        

        colors = vtk.vtkNamedColors()

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
        textActor.GetProperty().SetColor(colors.GetColor3d('White'))



        cfvv.MakeAxesActor(self)
        
        

        self.ren.AddActor(actor)
        self.ren.AddActor(axesActor)
        self.ren.AddActor(textActor)

        #sphere_actors = []
        nsph = np.size(spheres, axis = 0)
        for i in range(nsph):
            sphere_actor = cfvv.vtk_actor_objects(spheres[i])
            sphere_actor.GetProperty().SetColor(colors.GetColor3d('Red'))
            self.ren.AddActor(sphere_actor)

        

        self.frame.setLayout(self.vl)
        #self.show()
        #self.iren.Initialize()

        self.ren.ResetCamera()

        #self.renwin.Render()
        #self.renwin.Render()
        self.iren.Start()
        

if __name__ == "__main__":
    #beam = beam
    app = QApplication(sys.argv)
    #ex = MainWindow().__init__(beam)
    #MainWindow.vtk_actor(beam.coord)
    ex = MainWindow()
    ex.show()
    sys.exit(app.exec_())