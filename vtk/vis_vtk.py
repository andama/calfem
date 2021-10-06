# -*- coding: utf-8 -*-
"""
CALFEM Core vtk

Contains all the functions for 3D visualization in CALFEM

@author: Andreas Åmand
"""

import numpy as np
import vtk

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