#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import calfem.core as cfc
#import vtk

# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonCore import (
    vtkPoints,
    vtkUnsignedCharArray
)
from vtkmodules.vtkCommonDataModel import (
    vtkCellArray,
    vtkLine,
    vtkPolyData
)
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)


def main():
    
    K = np.zeros([18,18])
    f = np.zeros([18,1])
    f[7,0] = -3000
    
    edof = np.array([
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    ])
    
    ex1 = np.array([3, 1.5])
    ey1 = np.array([1, 1])
    ez1 = np.array([0, 2])
    
    ex2 = np.array([1.5, 0])
    ey2 = np.array([1, 1])
    ez2 = np.array([2, 4])
    
    
    
    
    
    
    eo = np.array([-3, 4, 0])
    
    E = 210000000
    v = 0.3
    G = E/(2*(1+v))
    A = 11250*0.000001
    Iy = 63.1*0.000001
    Iz = 182.6*0.000001
    Kv = 0.856*0.000001
    
    ep = [E, G, A, Iy, Iz, Kv]
    
    
    
    Ke1 = cfc.beam3e(ex1, ey1, ez1, eo, ep)
    Ke2 = cfc.beam3e(ex2, ey2, ez2, eo, ep)
    
    cfc.assem(edof, K, Ke1)
    cfc.assem(edof, K, Ke2)
    
    bcPrescr = np.array([1, 2, 3, 4, 7, 9, 13, 14, 15, 16])
    a, r = cfc.solveq(K, f, bcPrescr)
    
    print(a)
    print(f)
    
    ed1 = cfc.extractEldisp(edof[0,:],a)
    ed2 = cfc.extractEldisp(edof[1,:],a)
    
    es1 = cfc.beam3s(ex1,ey1,ez1,eo,ep,ed1)
    es2 = cfc.beam3s(ex2,ey2,ez2,eo,ep,ed2)
    
    print(ed1)
    print(ed2)
    
    
    
    
    
    
    

    # Create the polydata where we will store all the geometric data
    linesPolyData = vtkPolyData()

    # Create three points
    origin = [0.0, 0.0, 0.0]
    p0 = [0.5, 0.0, 0.0]
    p1 = [0.0, 0.5, 0.0]
    p2 = [0.0, 0.0, 0.5]
    
    #pts.InsertNextPoint([ex1[0], ey1[0], ey1[0]])
    #pts.InsertNextPoint([ex1[1], ey1[1], ey1[1]])
    #pts.InsertNextPoint([ex2[0], ey2[0], ey2[0]])
    #pts.InsertNextPoint([ex2[1], ey2[1], ey2[1]])
    
    p3 = [ex1[0], ey1[0], ey1[0]]
    p4 = [ex1[1], ey1[1], ey1[1]]
    #p5 = [ex2[0], ey2[0], ey2[0]]
    p5 = [ex1[1]+a[6], ey1[1]+a[7], ey1[1]+a[8]]
    p6 = [ex2[1], ey2[1], ey2[1]]
    
    

    # Create a vtkPoints container and store the points in it
    pts = vtkPoints()
    pts.InsertNextPoint(origin)
    pts.InsertNextPoint(p0)
    pts.InsertNextPoint(p1)
    pts.InsertNextPoint(p2)
    pts.InsertNextPoint(p3)
    pts.InsertNextPoint(p4)
    pts.InsertNextPoint(p5)
    pts.InsertNextPoint(p6)

    # Add the points to the polydata container
    linesPolyData.SetPoints(pts)

    # Create the first line (between Origin and P0)
    line0 = vtkLine()
    line0.GetPointIds().SetId(0, 0)  # the second 0 is the index of the Origin in linesPolyData's points
    line0.GetPointIds().SetId(1, 1)  # the second 1 is the index of P0 in linesPolyData's points

    # Create the second line (between Origin and P1)
    line1 = vtkLine()
    line1.GetPointIds().SetId(0, 0)  # the second 0 is the index of the Origin in linesPolyData's points
    line1.GetPointIds().SetId(1, 2)  # 2 is the index of P1 in linesPolyData's points
    
    # Create the third line (between Origin and P2)
    line2 = vtkLine()
    line2.GetPointIds().SetId(0, 0)  # the second 0 is the index of the Origin in linesPolyData's points
    line2.GetPointIds().SetId(1, 3)  # 2 is the index of P1 in linesPolyData's points
    
    line3 = vtkLine()
    line3.GetPointIds().SetId(0, 4)  
    line3.GetPointIds().SetId(1, 5)  
    
    line4 = vtkLine()
    line4.GetPointIds().SetId(0, 5)  
    line4.GetPointIds().SetId(1, 7)  
    
    line5 = vtkLine()
    line5.GetPointIds().SetId(0, 4)  
    line5.GetPointIds().SetId(1, 6)  
    
    line6 = vtkLine()
    line6.GetPointIds().SetId(0, 6)  
    line6.GetPointIds().SetId(1, 7)  


    # Create a vtkCellArray container and store the lines in it
    lines = vtkCellArray()
    lines.InsertNextCell(line0)
    lines.InsertNextCell(line1)
    lines.InsertNextCell(line2)
    lines.InsertNextCell(line3)
    lines.InsertNextCell(line4)
    lines.InsertNextCell(line5)
    lines.InsertNextCell(line6)

    # Add the lines to the polydata container
    linesPolyData.SetLines(lines)

    namedColors = vtkNamedColors()

    # Create a vtkUnsignedCharArray container and store the colors in it
    colors = vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    try:
        colors.InsertNextTupleValue(namedColors.GetColor3ub("Red"))
        colors.InsertNextTupleValue(namedColors.GetColor3ub("Blue"))
        colors.InsertNextTupleValue(namedColors.GetColor3ub("Green"))
        colors.InsertNextTupleValue(namedColors.GetColor3ub("Tomato"))
        colors.InsertNextTupleValue(namedColors.GetColor3ub("Mint"))
        colors.InsertNextTupleValue(namedColors.GetColor3ub("Black"))
        colors.InsertNextTupleValue(namedColors.GetColor3ub("Orange"))
    except AttributeError:
        colors.InsertNextTypedTuple(namedColors.GetColor3ub("Red"))
        colors.InsertNextTypedTuple(namedColors.GetColor3ub("Blue"))
        colors.InsertNextTypedTuple(namedColors.GetColor3ub("Green"))
        colors.InsertNextTypedTuple(namedColors.GetColor3ub("Tomato"))
        colors.InsertNextTypedTuple(namedColors.GetColor3ub("Mint"))
        colors.InsertNextTypedTuple(namedColors.GetColor3ub("Black"))
        colors.InsertNextTypedTuple(namedColors.GetColor3ub("Orange"))


    linesPolyData.GetCellData().SetScalars(colors)

    mapper = vtkPolyDataMapper()
    mapper.SetInputData(linesPolyData)

    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetLineWidth(4)

    renderer = vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(namedColors.GetColor3d("SlateGray"))

    window = vtkRenderWindow()
    window.SetWindowName("ColoredLines")
    window.AddRenderer(renderer)

    interactor = vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)

    window.Render()
    interactor.Start()


if __name__ == '__main__':
    main()