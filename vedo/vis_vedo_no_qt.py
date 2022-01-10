# -*- coding: utf-8 -*-

"""
CALFEM Vedo

Module for 3D visualization in CALFEM using Vedo (https://vedo.embl.es/)

@author: Andreas Åmand
"""

import numpy as np
import vedo as v
import sys
import webbrowser
from scipy.io import loadmat

# Examples using this module:
    # exv1: Spring
    # exv2: 3D Truss (beams & bars)
    # exv3: 3D Flow
    # exv4a: 3D Solid (using multiple renderers)
    # exv4b: 3D Solid w/ animation
    # exv5: 2D Plate visualized in 3D

### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###

# --- Tillfälliga anteckningar ---


# 2D mode: self.plotter.parallelProjection(value=True, at=0) + annat 'mode' för kameran
# Balkdiagram: vedo.pyplot.plot()
# Klicka på element, se nedan
        
"""
# Click a sphere to highlight it
from vedo import Text2D, Sphere, Plotter
import numpy as np

spheres = []
for i in range(25):
    p = np.random.rand(2)
    s = Sphere(r=0.05).pos(p).color('k5')
    s.name = f"sphere nr.{i} at {p}"
    spheres.append(s)

def func(evt):
    if not evt.actor: return
    sil = evt.actor.silhouette().lineWidth(6).c('red5')
    msg.text("You clicked: "+evt.actor.name)
    plt.remove(silcont.pop()).add(sil)
    silcont.append(sil)

silcont = [None]
msg = Text2D("", pos="bottom-center", c='k', bg='r9', alpha=0.8)

plt = Plotter(axes=1, bg='black')
plt.addCallback('mouse click', func)
plt.show(spheres, msg, __doc__, zoom=1.2).close()
        
"""

# Siluetter + mått, se nedan

"""
# Generate the silhouette of a mesh
# as seen along a specified direction

from vedo import *

s = Hyperboloid().rotateX(20)

sx = s.clone().projectOnPlane('x').c('r').x(-3) # sx is 2d
sy = s.clone().projectOnPlane('y').c('g').y(-3)
sz = s.clone().projectOnPlane('z').c('b').z(-3)

show(s,
     sx, sx.silhouette('2d'), # 2d objects dont need a direction
     sy, sy.silhouette('2d'),
     sz, sz.silhouette('2d'),
     __doc__,
     axes={'zxGrid':True, 'yzGrid':True},
     viewup='z',
).close()
"""

# self.plotter.show(mode=?), 0 -> trackball, 1 -> actor, 10 -> terrain
# self.plotter.show(axes=?), 4 -> koordinataxlar, 4 -> koordinataxlar i hörnet, 7 -> mått, 12 -> gradskiva






### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
# Initialization

def init_app():
    global vedo_app
    vedo_app = VedoPlotWindow.instance()
    if vedo_app is None: vedo_app = VedoMainWindow()
    return vedo_app

class VedoPlotWindow:
    __instance = None
    @staticmethod
    def instance():
        """ Static access method. """
        if VedoPlotWindow.__instance == None: VedoPlotWindow()
        return VedoPlotWindow.__instance

    def __init__(self):
        """ Virtually private constructor. """
        if VedoPlotWindow.__instance is None:
            VedoPlotWindow.__instance = self
            self.plot_window = VedoMainWindow()






### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
# Main window

class VedoMainWindow():
    
    def __init__(self):
        v.settings.immediateRendering = False
        self.plotter = v.Plotter(title='CALFEM vedo visualization tool')
        self.plotters = 0
        self.plotter.addHoverLegend(useInfo=True,s=1.25,maxlength=96)
        self.plotter.addCallback('mouse click', self.click)
        self.silcont = [None]
        self.click_msg = v.Text2D("", pos="bottom-center", bg='auto', alpha=0.1, font='Calco')
        self.plotter.add(self.click_msg)

    def click(self,evt):
        if evt.isAssembly: # endast för testning ifall en assembly skapas
            print('assembly')
            self.click_msg.text(evt.actor.info)
        elif evt.actor:
            sil = evt.actor.silhouette().lineWidth(6).c('red5')
            self.click_msg.text(evt.actor.info)
            self.plotter.remove(self.silcont.pop()).add(sil)
            self.silcont.append(sil)
        else: return

    def render(self,bg):
        if self.plotters > 0:
            for i in range(self.plotters):
                self.plotter.background(c1=bg, at=i)
            self.plotter.render(resetcam=True)
            v.interactive().close()
        else:
            self.plotter.show(resetcam=True,axes=4,bg=bg).close()
        
    def render_geometry(self,meshes,nodes=None,merge=False,window=0):

        # Mesh/elements plotted, possibly merged for correct numbering
        if merge == True:
            mesh = v.merge(meshes,flag=True)
            #mesh = v.Assembly(meshes)
            #self.plotter += mesh
            self.plotter.add(mesh,at=window)

            #mesh.clean()
            #mesh.computeNormals().clean().lw(0.1)
            #pids = mesh.boundaries(returnPointIds=True)
            #bpts = mesh.points()[pids]

            #pts = v.Points(bpts, r=1, c='red')
            #labels = mesh.labels('id', scale=0.02).c('w')

            #self.plotter += pts
            #self.plotter += labels
        else:
            nel = np.size(meshes, axis = 0)
            for i in range(nel):
                #self.plotter += meshes[i]
                self.plotter.add(meshes[i],at=window)

                #pids = meshes[i].boundaries(returnPointIds=True)
                #bpts = meshes[i].points()[pids]

                #pts = v.Points(bpts, r=1, c='red')
                #labels = meshes[i].labels('id', scale=0.02).c('w')

                #self.plotter += pts
                #self.plotter += labels

        # Optional nodes
        if nodes is not None:
            nnode = np.size(nodes, axis = 0)
            for i in range(nnode):
                #self.plotter += nodes[i]
                self.plotter.add(nodes[i],at=window)






### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
# Plotting functions, for adding things to a rendering window
    
# Add scalar bar to a renderer
def add_scalar_bar(
    meshes,
    label,
    pos=[0.75,0.05],
    font_size=16,
    color='white',
    window=0
    ):

    app = init_app()
    plot_window = VedoPlotWindow.instance().plot_window
    #mesh = v.merge(meshes,flag=True)
    #scalar_bar = v.addons.addScalarBar(mesh,title=label,pos=pos)
    nel = np.size(meshes, axis = 0)
    for i in range(nel):
        #scalar_bar = v.addons.addScalarBar(meshes[i],title=label,pos=pos,horizontal=True,titleFontSize=font_size,c=color)
        scalar_bar = v.addons.addScalarBar(meshes[i],title=label,pos=pos,horizontal=True,titleFontSize=font_size,useAlpha=False,c=color)
    plot_window.plotter.add(scalar_bar,at=window)

# Add text to a renderer
def add_text(
    text,
    color='white',
    pos='top-middle',
    window=0
    ):

    app = init_app()
    plot_window = VedoPlotWindow.instance().plot_window
    #msg = v.Text2D(text, pos=pos, c=color)
    msg = v.Text2D(text, pos=pos, alpha=1, c=color)
    plot_window.plotter.add(msg,at=window)

"""
# Add legend to a renderer
def add_legend(meshes,window=0):
    app = init_app()
    plot_window = VedoPlotWindow.instance().plot_window

    #mesh = v.merge(meshes,flag=True)

    legend_box = v.addons.LegendBox(meshes,pos='top-left')

    #nel = np.size(meshes, axis = 0)
    #for i in range(nel):
    #    scalar_bar = v.addons.LegendBox(meshes[i],pos='top-left')
    #plot_window.render_addon(legend_box,window=window)
    plot_window.plotter.add(legend_box,at=window)
"""



"""
# Add silhouette with measurements to a renderer
def add_silouette(text,color,pos=[0,1],window=0):
    app = init_app()
    plot_window = VedoPlotWindow.instance().plot_window

    #mesh = v.merge(meshes,flag=True)

    #legend_box = v.addons.LegendBox(meshes,pos='top-left')
    #msg = v.Text2D(text, pos=pos, c=color, bg='r9', alpha=0.8)
    msg = v.Text2D(text, pos=pos, c=color)

    #nel = np.size(meshes, axis = 0)
    #for i in range(nel):
    #    scalar_bar = v.addons.LegendBox(meshes[i],pos='top-left')
    #plot_window.render_addon(msg,window=window)
    plot_window.plotter.add(msg,at=window)
"""






### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
# Functions for plotting results from CALFEM calculations

# Element types: 1: Spring, 2: Bar, 3: Flow, 4: Solid, 5: Beam, 6: Plate

    # 2 node: 1,2,5 (1-3D)
    # 3 node: 3,4 (Triangular 2D)
    # 4 node: 3,4,6 (Quadratic/rectangular/isoparametric 2D)
    # 8 node: 3,4 (Isoparametric 2D or 3D)

# Creates an undeformed mesh for rendering, see element types above
def draw_geometry(
    edof,
    coord,
    dof,
    element_type,
    el_values=None,
    label=None,
    colormap='jet',
    scale=0.02,
    alpha=1,
    nseg=2,
    render_nodes=True,
    color=None,
    window=0,
    merge=False,
    t=None
    ):

    app = init_app()
    plot_window = VedoPlotWindow.instance().plot_window





    if element_type == 1:           

        nel = np.size(edof, axis = 0)
        elements = []

        for i in range(nel):
            coord1,coord2 = tools.get_coord_from_edof(edof[i,:],dof,element_type)

            #print(coord[coord1,0])
            #print(coord[coord2,0])
            #spring = v.Spring([coord[coord1,0],0,0],[coord[coord2,0],0,0])
            spring = v.Spring([coord[coord1,0],0,0],[coord[coord2,0],0,0],r=1.5*scale).alpha(alpha)
            spring.info = f"Spring nr. {i}"
            elements.append(spring)

        #print(elements,nodes)
        if render_nodes == True:
            nodes = tools.get_node_elements(coord,scale,alpha)
            plot_window.render_geometry(elements,nodes)
        else:
            plot_window.render_geometry(elements)

        return elements








    elif element_type == 2:

        nel = np.size(edof, axis = 0)
        elements = []

        res = 4

        vmin, vmax = np.min(el_values), np.max(el_values)

        el_values_array = np.zeros((1,4*res))[0,:]

        for i in range(nel):
            coord1,coord2 = tools.get_coord_from_edof(edof[i,:],dof,element_type)

            bar = v.Cylinder([[coord[coord1,0],coord[coord1,1],coord[coord1,2]],[coord[coord2,0],coord[coord2,1],coord[coord2,2]]],r=scale,res=res,c=color).alpha(alpha)
            elements.append(bar)


            if el_values is not None:
                bar.info = f"Bar nr. {i}, max el. value {el_values[i]}"

                el_values_array[1] = el_values[i]
                el_values_array[3] = el_values[i]
                el_values_array[5] = el_values[i]
                el_values_array[7] = el_values[i]
                el_values_array[12] = el_values[i]
                el_values_array[13] = el_values[i]
                el_values_array[14] = el_values[i]
                el_values_array[15] = el_values[i]

                el_values_array[0] = el_values[i]
                el_values_array[2] = el_values[i]
                el_values_array[4] = el_values[i]
                el_values_array[6] = el_values[i]
                el_values_array[8] = el_values[i]
                el_values_array[9] = el_values[i]
                el_values_array[10] = el_values[i]
                el_values_array[11] = el_values[i]

                bar.cmap(colormap, el_values_array, on="points", vmin=vmin, vmax=vmax)
            else:
                bar.info = f"Bar nr. {i}"
        if render_nodes == True:
            nodes = tools.get_node_elements(coord,scale,alpha)
            plot_window.render_geometry(elements,nodes)
        else:
            plot_window.render_geometry(elements)

        return elements











    elif element_type == 3 or element_type == 4:

        meshes = []
        nel = np.size(edof, axis = 0)

        vmin, vmax = np.min(el_values), np.max(el_values)
        #print(coord)
        for i in range(nel):
            coords = tools.get_coord_from_edof(edof[i,:],dof,4)


            mesh = v.Mesh([coord[coords,:],[[0,1,2,3],[4,5,6,7],[0,3,7,4],[1,2,6,5],[0,1,5,4],[2,3,7,6]]],alpha=alpha)
            mesh.info = f"Mesh nr. {i}"
            meshes.append(mesh)

            if el_values is not None and np.size(el_values, axis = 1) == 1:
                el_values_array = np.zeros((1,6))[0,:]
                el_values_array[0] = el_values[i]
                el_values_array[1] = el_values[i]
                el_values_array[2] = el_values[i]
                el_values_array[3] = el_values[i]
                el_values_array[4] = el_values[i]
                el_values_array[5] = el_values[i]

                mesh.cmap(colormap, el_values_array, on="cells", vmin=vmin, vmax=vmax)
            elif el_values is not None and np.size(el_values, axis = 1) > 1:
                el_values_array = np.zeros((1,8))[0,:]
                el_values_array[0] = el_values[i,0]
                el_values_array[1] = el_values[i,1]
                el_values_array[2] = el_values[i,2]
                el_values_array[3] = el_values[i,3]
                el_values_array[4] = el_values[i,4]
                el_values_array[5] = el_values[i,5]
                el_values_array[6] = el_values[i,6]
                el_values_array[7] = el_values[i,7]

                mesh.cmap(colormap, el_values_array, on="points", vmin=vmin, vmax=vmax)
        #if export is not None:
        #    v.io.write(mesh, export+".vtk")

        if render_nodes == True:
            nodes = tools.get_node_elements(coord,scale,alpha)
            plot_window.render_geometry(meshes,nodes)
        else:
            plot_window.render_geometry(meshes)

        return meshes











    elif element_type == 5:
        ncoord = np.size(coord, axis = 0)
        nel = np.size(edof, axis = 0)
        elements = []
        
        res = 4

        vmin, vmax = np.min(el_values), np.max(el_values)

        el_values_array = np.zeros((1,4*res))[0,:]

        for i in range(nel):
            coord1,coord2 = tools.get_coord_from_edof(edof[i,:],dof,5)

            if nseg > 2:
                steps = np.float32(1/(nseg-1))

                dx = (coord[coord2,0]-coord[coord1,0])*steps
                dy = (coord[coord2,1]-coord[coord1,1])*steps
                dz = (coord[coord2,2]-coord[coord1,2])*steps

                for j in range(nseg-1):
                    x1 = coord[coord1,0]+dx*j
                    y1 = coord[coord1,1]+dy*j
                    z1 = coord[coord1,2]+dz*j

                    x2 = coord[coord1,0]+dx*(j+1)
                    y2 = coord[coord1,1]+dy*(j+1)
                    z2 = coord[coord1,2]+dz*(j+1)

                    beam = v.Cylinder([[x1,y1,z1],[x2,y2,z2]],r=scale,res=res,c=color).alpha(alpha)
                    beam.info = f"Beam nr. {i}, seg. {j}"
                    elements.append(beam)

                    if el_values is not None:
                        el_value1 = el_values[nseg*i]
                        el_value2 = el_values[nseg*i+1]

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

                        beam.cmap(colormap, el_values_array, on="points", vmin=vmin, vmax=vmax)

            else:
                beam = v.Cylinder([[coord[coord1,0],coord[coord1,1],coord[coord1,2]],[coord[coord2,0],coord[coord2,1],coord[coord2,2]]],r=scale,res=res,c=color).alpha(alpha)
                beam.info = f"Beam nr. {i}"
                elements.append(beam)

                if el_values is not None:
                    el_value1 = el_values[2*i]
                    el_value2 = el_values[2*i+1]

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

                    beam.cmap(colormap, el_values_array, on="points", vmin=vmin, vmax=vmax)


        #if export is not None:
        #    v.io.write(mesh, export+".vtk")

        if render_nodes == True:
            nodes = tools.get_node_elements(coord,scale,alpha)
            plot_window.render_geometry(elements,nodes)
        else:
            plot_window.render_geometry(elements)

        return elements
    











    elif element_type == 6:
        meshes = []
        nel = np.size(edof, axis = 0)

        vmin, vmax = np.min(el_values), np.max(el_values)
        #print(coord)
        for i in range(nel):
            coords = tools.get_coord_from_edof(edof[i,:],dof,6)
            #print(coords)
            #print(coord[coords,:])

            new_coord = np.zeros([8,3])
            new_coord[0,0] = coord[coords[0],0]
            new_coord[1,0] = coord[coords[1],0]
            new_coord[2,0] = coord[coords[2],0]
            new_coord[3,0] = coord[coords[3],0]
            new_coord[4,0] = coord[coords[0],0]
            new_coord[5,0] = coord[coords[1],0]
            new_coord[6,0] = coord[coords[2],0]
            new_coord[7,0] = coord[coords[3],0]

            new_coord[0,1] = coord[coords[0],1]
            new_coord[1,1] = coord[coords[1],1]
            new_coord[2,1] = coord[coords[2],1]
            new_coord[3,1] = coord[coords[3],1]
            new_coord[4,1] = coord[coords[0],1]
            new_coord[5,1] = coord[coords[1],1]
            new_coord[6,1] = coord[coords[2],1]
            new_coord[7,1] = coord[coords[3],1]

            new_coord[0,2] = -0.5*t
            new_coord[1,2] = -0.5*t
            new_coord[2,2] = -0.5*t
            new_coord[3,2] = -0.5*t
            new_coord[4,2] =  0.5*t
            new_coord[5,2] =  0.5*t
            new_coord[6,2] =  0.5*t
            new_coord[7,2] =  0.5*t

            #print(new_coord)

            #coord.append(np.array([-;]))

            plate = v.Mesh([new_coord,[[0,1,2,3],[4,5,6,7],[0,3,7,4],[1,2,6,5],[0,1,5,4],[2,3,7,6]]],alpha=alpha)
            plate.info = f"Mesh nr. {i}"
            meshes.append(plate)

            if el_values is not None and np.size(el_values, axis = 1) == 1:
                el_values_array = np.zeros((1,6))[0,:]
                el_values_array[0] = el_values[i]
                el_values_array[1] = el_values[i]
                el_values_array[2] = el_values[i]
                el_values_array[3] = el_values[i]
                el_values_array[4] = el_values[i]
                el_values_array[5] = el_values[i]

                mesh.cmap(colormap, el_values_array, on="cells", vmin=vmin, vmax=vmax)
            elif el_values is not None and np.size(el_values, axis = 1) > 1:
                el_values_array = np.zeros((1,8))[0,:]
                el_values_array[0] = el_values[i,0]
                el_values_array[1] = el_values[i,1]
                el_values_array[2] = el_values[i,2]
                el_values_array[3] = el_values[i,3]
                el_values_array[4] = el_values[i,4]
                el_values_array[5] = el_values[i,5]
                el_values_array[6] = el_values[i,6]
                el_values_array[7] = el_values[i,7]

                plate.cmap(colormap, el_values_array, on="points", vmin=vmin, vmax=vmax)
        #if export is not None:
        #    v.io.write(mesh, export+".vtk")

        if render_nodes == True:
            nodes = tools.get_node_elements(coord,scale,alpha)
            plot_window.render_geometry(meshes,nodes)
        else:
            plot_window.render_geometry(meshes)

        return meshes

    else:
        print("Invalid element type, please declare 'element_type'. The element types are:\n    1 - Spring\n    2 - Bar\n    3 - Flow\n    4 - Solid\n    5 - Beam\n    6 - Plate")
        sys.exit()
        #try:
        #    raise ValueError("Invalid element type, please declare 'element_type'. The element types are; 1: Spring, 2: Bar, 3: Flow, 4: Solid, 5: Beam, 6: Plate")
        #except ValueError:
            #return ValueError("Invalid element type, please declare 'element_type'. The element types are; 1: Spring, 2: Bar, 3: Flow, 4: Solid, 5: Beam, 6: Plate")
        #    return
            #print('test')













# Creates a deformed mesh for rendering, see element types above
def draw_displaced_geometry(
    edof,
    coord,
    dof,
    element_type,
    a,
    el_values=None,
    label=None,
    colormap='jet',
    scale=0.02,
    alpha=1,
    def_scale=1,
    nseg=2,
    render_nodes=True,
    color='white',
    offset = [0, 0, 0],
    window=0,
    merge=False,
    t=None
    ):
 

    app = init_app()
    plot_window = VedoPlotWindow.instance().plot_window
    #animations = animations().instance()

    if element_type == 1:
        ndof = np.size(dof, axis = 0)*np.size(dof, axis = 1)
        print(coord[:,1])
        coord[:,0] = coord[:,0] + offset[0]
        coord[:,1] = coord[:,1] + offset[1]
        print(coord[:,1])
        coord[:,2] = coord[:,2] + offset[2]

        nel = np.size(edof, axis = 0)
        elements = []

        for i in range(nel):
            coord1,coord2 = tools.get_coord_from_edof(edof[i,:],dof,element_type)

            spring = v.Spring([coord[coord1,0],0,0],[coord[coord2,0],0,0],r=scale*1.5).alpha(alpha)
            spring.info = f"Spring nr. {i}"
            elements.append(spring)

        if render_nodes == True:
            nodes = tools.get_node_elements(coord,scale,alpha)
            plot_window.render_geometry(elements,nodes)
        else:
            plot_window.render_geometry(elements)

        return elements









    elif element_type == 2:
        ndof = np.size(dof, axis = 0)*np.size(dof, axis = 1)
        ncoord = np.size(coord, axis = 0)
        def_nodes = []
        def_coord = np.zeros([ncoord,3])

        for i in range(0, ncoord):
            #if a.any() == None:
            #    x = coord[i,0]
            #    y = coord[i,1]
            #    z = coord[i,2]
            #else:
            a_dx, a_dy, a_dz = tools.get_a_from_coord(i,6,a,def_scale)

            x = coord[i,0]+a_dx
            y = coord[i,1]+a_dy
            z = coord[i,2]+a_dz

            def_coord[i] = [x,y,z]

            def_nodes.append(v.Sphere(c='white').scale(1.5*scale).pos([x,y,z]).alpha(alpha))


        nel = np.size(edof, axis = 0)
        def_elements = []
        res = 4

        vmin, vmax = np.min(el_values), np.max(el_values)

        el_values_array = np.zeros((1,4*res))[0,:]

        for i in range(nel):
            coord1,coord2 = tools.get_coord_from_edof(edof[i,:],dof,element_type)


            bar = v.Cylinder([[def_coord[coord1,0],def_coord[coord1,1],def_coord[coord1,2]],[def_coord[coord2,0],def_coord[coord2,1],def_coord[coord2,2]]],r=scale,res=res,c=color).alpha(alpha)
            bar.info = f"Bar nr. {i}, at [{def_coord[coord1,0]+0.5*(def_coord[coord2,0]-def_coord[coord1,0])},{def_coord[coord1,1]+0.5*(def_coord[coord2,1]-def_coord[coord1,1])},{def_coord[coord1,2]+0.5*(def_coord[coord2,2]-def_coord[coord1,2])}]"
            def_elements.append(bar)
            if el_values is not None:
                bar.info = bar.info + f", max el. value {el_values[i]}"
                el_values_array[1] = el_values[i]
                el_values_array[3] = el_values[i]
                el_values_array[5] = el_values[i]
                el_values_array[7] = el_values[i]
                el_values_array[12] = el_values[i]
                el_values_array[13] = el_values[i]
                el_values_array[14] = el_values[i]
                el_values_array[15] = el_values[i]

                el_values_array[0] = el_values[i]
                el_values_array[2] = el_values[i]
                el_values_array[4] = el_values[i]
                el_values_array[6] = el_values[i]
                el_values_array[8] = el_values[i]
                el_values_array[9] = el_values[i]
                el_values_array[10] = el_values[i]
                el_values_array[11] = el_values[i]

                bar.cmap(colormap, el_values_array, on="points", vmin=vmin, vmax=vmax)

        if render_nodes == True:
            plot_window.render_geometry(def_elements,def_nodes)
        else:
            plot_window.render_geometry(def_elements)

        return def_elements












    elif element_type == 3:
        print("Displaced mesh for flow elements is not supported")
        sys.exit()












    elif element_type == 4:
        ncoord = np.size(coord, axis = 0)

        def_nodes = []
        def_coord = np.zeros([ncoord,3])

        for i in range(0, ncoord):
            #if a.any() == None:
            #    x = coord[i,0]
            #    y = coord[i,1]
            #    z = coord[i,2]
            #else:
            a_dx, a_dy, a_dz = tools.get_a_from_coord(i,3,a,def_scale)

            x = coord[i,0]+a_dx
            y = coord[i,1]+a_dy
            z = coord[i,2]+a_dz

            def_coord[i] = [x,y,z]

            def_nodes.append(v.Sphere(c='white').scale(1.5*scale).pos([x,y,z]).alpha(alpha))

        meshes = []
        nel = np.size(edof, axis = 0)
        
        vmin, vmax = np.min(el_values), np.max(el_values)
        for i in range(nel):
            coords = tools.get_coord_from_edof(edof[i,:],dof,4)

            mesh = v.Mesh([def_coord[coords,:],[[0,1,2,3],[4,5,6,7],[0,3,7,4],[1,2,6,5],[0,1,5,4],[2,3,7,6]]],alpha=alpha)
            mesh.info = f"Mesh nr. {i}"
            meshes.append(mesh)

            
            
            if el_values is not None and np.size(el_values, axis = 1) == 1:
                el_values_array = np.zeros((1,6))[0,:]
                el_values_array[0] = el_values[i]
                el_values_array[1] = el_values[i]
                el_values_array[2] = el_values[i]
                el_values_array[3] = el_values[i]
                el_values_array[4] = el_values[i]
                el_values_array[5] = el_values[i]

                mesh.cmap(colormap, el_values_array, on="cells", vmin=vmin, vmax=vmax)
            elif el_values is not None and np.size(el_values, axis = 1) > 1:
                el_values_array = np.zeros((1,8))[0,:]
                el_values_array[0] = el_values[i,0]
                el_values_array[1] = el_values[i,1]
                el_values_array[2] = el_values[i,2]
                el_values_array[3] = el_values[i,3]
                el_values_array[4] = el_values[i,4]
                el_values_array[5] = el_values[i,5]
                el_values_array[6] = el_values[i,6]
                el_values_array[7] = el_values[i,7]

                mesh.cmap(colormap, el_values_array, on="points", vmin=vmin, vmax=vmax)

        #if export is not None:
        #    v.io.write(mesh, export+".vtk")

        if render_nodes == True:
            nodes = tools.get_node_elements(coord,scale,alpha)
            plot_window.render_geometry(meshes,nodes,window=window)
        else:
            plot_window.render_geometry(meshes,window=window)

        return meshes













    elif element_type == 5:
        ndof = np.size(dof, axis = 0)*np.size(dof, axis = 1)
        ncoord = np.size(coord, axis = 0)
        def_nodes = []
        def_coord = np.zeros([ncoord,3])

        for i in range(0, ncoord):
            #if a.any() == None:
            #    x = coord[i,0]
            #    y = coord[i,1]
            #    z = coord[i,2]
            #else:
            a_dx, a_dy, a_dz = tools.get_a_from_coord(i,6,a,def_scale)

            x = coord[i,0]+a_dx
            y = coord[i,1]+a_dy
            z = coord[i,2]+a_dz

            def_coord[i] = [x,y,z]

            def_nodes.append(v.Sphere(c='white').scale(1.5*scale).pos([x,y,z]).alpha(alpha))

        nel = np.size(edof, axis = 0)
        def_elements = []
        res = 4

        vmin, vmax = np.min(el_values), np.max(el_values)

        el_values_array = np.zeros((1,4*res))[0,:]

        for i in range(nel):
            coord1,coord2 = tools.get_coord_from_edof(edof[i,:],dof,5)

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

                    beam = v.Cylinder([[x1,y1,z1],[x2,y2,z2]],r=scale,res=res,c=color).alpha(alpha)
                    beam.info = f"Beam nr. {i}, seg. {j}"
                    def_elements.append(beam)

                    if el_values is not None:
                        el_value1 = el_values[nseg*i]
                        el_value2 = el_values[nseg*i+1]

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

                        beam.cmap(colormap, el_values_array, on="points", vmin=vmin, vmax=vmax)

            else:
                beam = v.Cylinder([[def_coord[coord1,0],def_coord[coord1,1],def_coord[coord1,2]],[def_coord[coord2,0],def_coord[coord2,1],def_coord[coord2,2]]],r=scale,res=res,c=color).alpha(alpha)
                beam.info = f"Beam nr. {i}"
                def_elements.append(beam)

                if el_values is not None:
                    el_value1 = el_values[2*i]
                    el_value2 = el_values[2*i+1]

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

                    beam.cmap(colormap, el_values_array, on="points", vmin=vmin, vmax=vmax)

        if render_nodes == True:
            plot_window.render_geometry(def_elements,def_nodes,merge=merge)
        else:
            plot_window.render_geometry(def_elements,merge=merge)

        return def_elements








    elif element_type == 6:
        print("Displaced mesh for plate elements is not supported")
        sys.exit()

    else:
        print("Invalid element type, please declare 'element_type'. The element types are:\n    1 - Spring\n    2 - Bar\n    3 - Flow (unsupported in this function)\n    4 - Solid\n    5 - Beam\n    6 - Plate")
        sys.exit()
        #try:
        #    raise ValueError("Invalid element type, please declare 'element_type'. The element types are; 1: Spring, 2: Bar, 3: Flow, 4: Solid, 5: Beam, 6: Plate")
        #except ValueError:
            #return ValueError("Invalid element type, please declare 'element_type'. The element types are; 1: Spring, 2: Bar, 3: Flow, 4: Solid, 5: Beam, 6: Plate")
        #    return
            #print('test')
    #plot_window.anim_def_coord = coord






### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
# Functions regarding animations

def animate(
    edof,
    coord,
    dof,
    element_type,
    a,
    steps,
    def_scale=1
    ):
    app = init_app()
    plot_window = VedoPlotWindow.instance().plot_window

    

    t = np.arange(0, 1+1/(steps-1), 1/(steps-1))
    print(t)
    #print(start)
    #print(end)
    #print(steps)
    
    #if self.elements != None:
    #    element_type = 1
    #elif self.mesh != None:
    #    element_type = 2

    #dt = 0.1

    #t = np.arange(0.0, 10.0, dt)

    pb = v.ProgressBar(0, len(t), c="b")


    if element_type == 5:
        ncoord = np.size(coord, axis = 0)

        def_coord = np.zeros([ncoord,3])
        def_coord = np.zeros((ncoord,3,steps))




        for i in range(0, ncoord):
            a_dx, a_dy, a_dz = tools.get_a_from_coord(i,3,a,def_scale)

            x_step = a_dx/(steps-1)
            y_step = a_dy/(steps-1)
            z_step = a_dz/(steps-1)

            for j in range(0, steps):

                x = coord[i,0]+x_step*j
                y = coord[i,1]+y_step*j
                z = coord[i,2]+z_step*j

                def_coord[i,:,j] = [x,y,z]

        meshes = np.empty(nel,steps)
        nel = np.size(edof, axis = 0)
        
        vmin, vmax = np.min(el_values), np.max(el_values)
        for i in range(nel):
            coords = tools.get_coord_from_edof(edof[i,:],dof,4)
            for j in range(steps):
                mesh = v.Mesh([def_coord[coords,:,j],[[0,1,2,3],[4,5,6,7],[0,3,7,4],[1,2,6,5],[0,1,5,4],[2,3,7,6]]],alpha=alpha)
                meshes[i,j] = mesh
            #mesh = v.Mesh([def_coord[coords,:],[[0,1,2,3],[4,5,6,7],[0,3,7,4],[1,2,6,5],[0,1,5,4],[2,3,7,6]]],alpha=alpha)
            #mesh.info = f"Mesh nr. {i}"
            #meshes.append(mesh)
            

        #mesh = v.merge(meshes,flag=True)

        #if export is not None:
        #    v.io.write(mesh, export+".vtk")

        #plot_window.render_geometry(meshes,window=window)

        for j in pb.range():
            for i in range(nel):
                #plot_window.plotter.show(meshes[i,j])
                #v.interactive()
                plot_window.render_geometry(meshes[:,j])




    
    #elif element_type == 5:
    #    ncoord = np.size(coord, axis = 0)
    #    def_coord = np.zeros([ncoord,3])





"""
class animations():
    
    #def __init__(self):
    #    #Qt.QMainWindow.__init__(self)
    #    #self.initialize()
    #    #self.type = None

    #    # Type 1
    #    self.elements = None
    #   self.nodes = None
    #    self.def_elements = None
    #    self.def_nodes = None

    #    self.mesh = None
    #    self.def_mesh = None
    

    def __init__(self):

        self.start = 0
        self.end = 1
        self.steps = 101

    def __call__(self):
        #print('test')
        #self.elements = None
        #self.nodes = None
        #self.def_elements = None
        #self.def_nodes = None

        #self.mesh = None
        #self.def_mesh = None

        self.type = None
        self.coords = None
        self.def_coords = None

    def edit(self):
        print('edit_parameters')


    #def animate(coord_start,coord_end,el_values,element_type):
    def animate(self):
        app = init_app()
        plot_window = VedoPlotWindow.instance().plot_window

        t = np.arange(self.start, self.end, self.steps)
        print(t)
        
        #if self.elements != None:
        #    element_type = 1
        #elif self.mesh != None:
        #    element_type = 2
        
        if element_type == 1:
            ncoord = np.size(self.coord, axis = 0)
            def_coord = np.zeros([ncoord,3])
"""











### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
# Functions for importing/exporting

#def import(file,list=None):
    #data = loadmat(file+'.mat')

#def export(mesh,file):
    #v.io.write(mesh, file+".vtk")









### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
# Tools, used in this file but can be accessed by a user as well (see exv4a.py)

class tools:
    def get_coord_from_edof(edof_row,dof,element_type):
        if element_type == 1 or element_type == 2 or element_type == 5:
            edof_row1,edof_row2 = np.split(edof_row,2)
            coord1 = int(np.where(np.all(edof_row1==dof,axis=1))[0])
            coord2 = int(np.where(np.all(edof_row2==dof,axis=1))[0])
            return coord1, coord2
        elif element_type == 3 or element_type == 4:
            edof_row1,edof_row2,edof_row3,edof_row4,edof_row5,edof_row6,edof_row7,edof_row8 = np.split(edof_row,8)
            coord1 = int(np.where(np.all(edof_row1==dof,axis=1))[0])
            coord2 = int(np.where(np.all(edof_row2==dof,axis=1))[0])
            coord3 = int(np.where(np.all(edof_row3==dof,axis=1))[0])
            coord4 = int(np.where(np.all(edof_row4==dof,axis=1))[0])
            coord5 = int(np.where(np.all(edof_row5==dof,axis=1))[0])
            coord6 = int(np.where(np.all(edof_row6==dof,axis=1))[0])
            coord7 = int(np.where(np.all(edof_row7==dof,axis=1))[0])
            coord8 = int(np.where(np.all(edof_row8==dof,axis=1))[0])
            coords = np.array([coord1, coord2, coord3, coord4, coord5, coord6, coord7, coord8])
            return coords
        elif element_type == 6:
            edof_row1,edof_row2,edof_row3,edof_row4 = np.split(edof_row,4)
            coord1 = int(np.where(np.all(edof_row1==dof,axis=1))[0])
            coord2 = int(np.where(np.all(edof_row2==dof,axis=1))[0])
            coord3 = int(np.where(np.all(edof_row3==dof,axis=1))[0])
            coord4 = int(np.where(np.all(edof_row4==dof,axis=1))[0])
            coords = np.array([coord1, coord2, coord3, coord4])
            return coords

    def get_a_from_coord(coord_row_num,num_of_deformations,a,scale=1):
        dx = a[coord_row_num*num_of_deformations]*scale
        dy = a[coord_row_num*num_of_deformations+1]*scale
        dz = a[coord_row_num*num_of_deformations+2]*scale
        return dx, dy, dz

    def get_node_elements(coord,scale,alpha,t=None):
        nnode = np.size(coord, axis = 0)
        ncoord = np.size(coord, axis = 1)
        nodes = []
        for i in range(nnode):
            if ncoord == 3:
                node = v.Sphere(c='white').scale(1.5*scale).pos([coord[i,0],coord[i,1],coord[i,2]]).alpha(alpha)
                node.info = f"Node nr. {i}"
                nodes.append(node)
            elif ncoord == 2:
                node = v.Sphere(c='white').scale(1.5*scale).pos([coord[i,0],coord[i,1],0]).alpha(alpha)
                node.info = f"Node nr. {i}"
                nodes.append(node)
            #elif ncoord == 2 and t is not None:
            #    node = v.Sphere(c='white').scale(1.5*scale).pos([coord[i,0],coord[i,1],0]).alpha(alpha)
            #    node.info = f"Node nr. {i}"
            #    nodes.append(node)
            elif ncoord == 1:
                node = v.Sphere(c='white').scale(1.5*scale).pos([coord[i,0],0,0]).alpha(alpha)
                node.info = f"Node nr. {i}"
                nodes.append(node)
        return nodes






### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
# Functions for rendering

# If multiple renderers are used
def set_windows(n):
    app = init_app()
    plot_window = VedoPlotWindow.instance().plot_window

    del(plot_window.plotter)

    plot_window.plotter = v.Plotter(title='CALFEM vedo visualization tool',N=n,axes=4)
    plot_window.plotters = n
    plot_window.click_msg = []
    for i in range(n):
        #plot_window.plotter.addGlobalAxes(4,at=i)
        plot_window.plotter.addHoverLegend(useInfo=True,s=1.25,maxlength=96,at=i)
        #plot_window.plotter.addCallback('mouse click', plot_window.click)
        #plot_window.click_msg.append(v.Text2D("", pos="bottom-center", bg='auto', alpha=0.1, font='Calco'))
        #plot_window.plotter.add(plot_window.click_msg[i],at=i)

# Start Calfem-vedo visualization
def render(bg='black'):
    app = init_app()
    plot_window = VedoPlotWindow.instance().plot_window
    plot_window.render(bg)
