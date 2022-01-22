# -*- coding: utf-8 -*-

"""
CALFEM Vedo

Module for 3D visualization in CALFEM using Vedo (https://vedo.embl.es/)

@author: Andreas Åmand
"""

import numpy as np
import vedo as v
import pyvtk
import vtk
import sys
#import webbrowser
from scipy.io import loadmat
import calfem.core as cfc

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
        # Variables to keep track of no. plotters
        #self.n = 0
        self.fig = 0
        self.meshes = [[]]
        self.nodes = [[]]
        self.msg = [[]]
        self.proj = [[]]
        self.plt = []
        self.rulers = [[]]

        # Mouse click callback
        #self.plotter[self.n].addCallback('mouse click', self.click)
        self.silcont = [None]
        self.click_msg = v.Text2D("", pos="bottom-center", bg='auto', alpha=0.1, font='Calco',c='black')
        #self.plotter[self.n].add(self.click_msg)

        # Global settings
        v.settings.immediateRendering = False
        #v.settings.renderLinesAsTubes = True
        v.settings.allowInteraction = True
        v.settings.useFXAA = True
        #v.settings.useSSAO         = True
        v.settings.visibleGridEdges = True
        #v.settings.useParallelProjection = True

    def click(self,evt):
        #if evt.isAssembly: # endast för testning ifall en assembly skapas
        #    print('assembly')
        #    self.click_msg.text(evt.actor.info)
        if evt.actor:
            sil = evt.actor.silhouette().lineWidth(6).c('red5')
            self.click_msg.text(evt.actor.name)
            self.plt[evt.title].remove(self.silcont.pop()).add(sil)
            #evt.interactor.add(sil)
            #evt.interactor.pop()
            #.remove(self.silcont.pop())
            #.add(sil)
            self.plotter.remove(self.silcont.pop()).add(sil)
            #self.silcont.append(sil)
        else:
            self.click_msg.text('')
            return

    def render(self):

        for i in range(self.fig+1):
            opts = dict(axes=4, interactive=False, new=True, title=f'Figure {i+1} - CALFEM vedo visualization tool')
            plt = v.show(self.meshes[i], self.nodes[i], self.click_msg, **opts)
            #plt.addGlobalAxes(11)#
            #plt.addShadows()
            #plt.addScaleIndicator(pos=(0.7, 0.05), s=0.02, length=2, lw=4, c='k1', alpha=1, units='', gap=0.05)
            plt.addCallback('mouse click', self.click)
            #plt += self.click_msg#.addHoverLegend(useInfo=True,s=1.25,maxlength=96)
            print('Figure text: ',self.msg[i])
            print('Projections: ',self.proj[i])
            self.plt.append(plt)

            #if self.msg[i]:
            for j in range(len(self.msg[i])):
                plt.add(self.msg[i][j])
                    #plt += self.msg[i][j]
                    #plt.add(self.click_msg)

            #if self.proj[i]:
            for j in range(len(self.proj[i])):
                plt.add(self.proj[i][j])

            #if self.rulers[i]:
            for j in range(len(self.rulers[i])):
                plt.add(self.rulers[i][j])

        v.interactive()



### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
# Plotting functions, for adding things to a rendering window
    
# Add scalar bar to a renderer
def add_scalar_bar(
    label,
    pos=[0.75,0.05],
    font_size=24,
    color='black',
    size=(3000, 50)
    ):
    app = init_app()
    plot_window = VedoPlotWindow.instance().plot_window

    fig = plot_window.fig
    plot_window.meshes[fig][0].addScalarBar(title=label, size=size, pos=pos, titleFontSize=font_size, horizontal=True, useAlpha=False)

# Add text to a renderer
def add_text(
    text,
    color='black',
    pos='top-middle'
    ):
    app = init_app()
    plot_window = VedoPlotWindow.instance().plot_window

    #while len(plot_window.msg) < plot_window.fig + 1:
        

    #plot_window.msg.append([])

    msg = v.Text2D(text, pos=pos, alpha=1, c=color)
    plot_window.msg[plot_window.fig] += [msg]

# Add silhouette with or without measurements to a renderer
def add_projection(color='black',plane='xy',offset=0,rulers=False):
    app = init_app()
    plot_window = VedoPlotWindow.instance().plot_window

    

    print("Size of projections: ",len(plot_window.proj))
    #while len(plot_window.proj) < plot_window.fig + 1:
        
    #if rulers == True:
        #while len(plot_window.rulers) < plot_window.fig + 1:
            
    print("Size of projections after loop: ",len(plot_window.proj))

    if plane == 'xy':
        assem = v.merge(plot_window.meshes[plot_window.fig])
        proj = assem.projectOnPlane('z').z(offset).silhouette('2d').c(color)
        plot_window.proj[plot_window.fig] += [proj]
        if rulers == True:
            ruler = v.addons.RulerAxes(proj, xtitle='', ytitle='', ztitle='', xlabel='', ylabel='', zlabel='', xpad=0.1, ypad=0.1, zpad=0, font='Normografo', s=None, italic=0, units='m', c=color, alpha=1, lw=1, precision=3, labelRotation=0, axisRotation=0, xycross=True)
            plot_window.rulers[plot_window.fig] += [ruler]
        """
        for i in range(len(plot_window.meshes[plot_window.fig])):
            proj = plot_window.meshes[plot_window.fig][i].clone().projectOnPlane('z').c(color).z(offset)
            plot_window.proj[plot_window.fig] += [proj]
        """
        #plot_window.meshes[plot_window.fig][0].addShadow(plane='x')
    elif plane == 'xz':
        assem = v.merge(plot_window.meshes[plot_window.fig])
        proj = assem.projectOnPlane('y').y(offset).silhouette('2d').c(color)
        plot_window.proj[plot_window.fig] += [proj]
        if rulers == True:
            ruler = v.addons.RulerAxes(proj, xtitle='', ytitle='', ztitle='', xlabel='', ylabel='', zlabel='', xpad=0.1, ypad=0, zpad=0.1, font='Normografo', s=None, italic=0, units='m', c=color, alpha=1, lw=1, precision=3, labelRotation=0, axisRotation=0, xycross=True)
            plot_window.rulers[plot_window.fig] += [ruler]
    elif plane == 'yz':
        assem = v.merge(plot_window.meshes[plot_window.fig])
        proj = assem.projectOnPlane('x').x(offset).silhouette('2d').c(color)
        plot_window.proj[plot_window.fig] += [proj]
        if rulers == True:
            ruler = v.addons.RulerAxes(proj, xtitle='', ytitle='', ztitle='', xlabel='', ylabel='', zlabel='', xpad=0, ypad=0.1, zpad=0.1, font='Normografo', s=None, italic=0, units='m', c=color, alpha=1, lw=1, precision=3, labelRotation=0, axisRotation=0, xycross=True)
            plot_window.rulers[plot_window.fig] += [ruler]
    else:
        print("Please choose a plane to project to. Set plane to 'xy', 'xz' or 'yz'")
        sys.exit()

    

    #plot_window.proj.append([])

    #plot_window.proj[plot_window.fig] += [proj]

#Add measurements to a renderer
def add_rulers(xtitle='', ytitle='', ztitle='', xlabel='', ylabel='', zlabel='', alpha=1):
    app = init_app()
    plot_window = VedoPlotWindow.instance().plot_window

    #while len(plot_window.rulers) < plot_window.fig + 1:
        #plot_window.rulers.append([])

    assem = v.merge(plot_window.meshes[plot_window.fig])

    ruler = v.addons.RulerAxes(assem, xtitle=xtitle, ytitle=ytitle, ztitle=ztitle, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, xpad=0.1, ypad=0.1, zpad=0.1, font='Normografo', s=None, italic=0, units='m', c=(1,1,1), alpha=alpha, lw=1, precision=3, labelRotation=0, axisRotation=0, xycross=False)

    plot_window.rulers[plot_window.fig] += [ruler]



### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
# Functions for plotting results from CALFEM calculations

# Element types: 1: Spring, 2: Bar, 3: Flow, 4: Solid, 5: Beam, 6: Plate

    # 2 node: 1,2,5 (1-3D)
    # 3 node: 3,4 (Triangular 2D)
    # 4 node: 3,4,6 (Quadratic/rectangular/isoparametric 2D)
    # 8 node: 3,4 (Isoparametric 2D or 3D)

# Creates an undeformed mesh for rendering, see element types above
def draw_mesh(
    edof,
    coord,
    dof,
    element_type,
    el_values=None,
    colormap='jet',
    scale=0.02,
    alpha=1,
    nseg=2,
    render_nodes=True,
    color='yellow',
    offset = [0, 0, 0],
    merge=False,
    t=None,
    ):

    app = init_app()
    plot_window = VedoPlotWindow.instance().plot_window





    if element_type == 1:           

        nel = np.size(edof, axis = 0)
        elements = []

        for i in range(nel):
            coord1,coord2 = get_coord_from_edof(edof[i,:],dof,element_type)

            #print(coord[coord1,0])
            #print(coord[coord2,0])
            #spring = v.Spring([coord[coord1,0],0,0],[coord[coord2,0],0,0])
            spring = v.Spring([coord[coord1,0],0,0],[coord[coord2,0],0,0],r=1.5*scale).alpha(alpha)
            spring.info = f"Spring nr. {i}"
            elements.append(spring)

        #print(elements,nodes)
        if render_nodes == True:
            nodes = get_node_elements(coord,scale,alpha)
            #plot_window.add_geometry(elements,nodes)
            plot_window.meshes[plot_window.fig].extend(elements)
            plot_window.nodes[plot_window.fig].extend(nodes)
        else:
            #plot_window.add_geometry(elements)
            plot_window.meshes[plot_window.fig].extend(elements)

        return elements

    elif element_type == 2:

        nel = np.size(edof, axis = 0)
        elements = []

        res = 4

        vmin, vmax = np.min(el_values), np.max(el_values)

        el_values_array = np.zeros((1,4*res))[0,:]

        for i in range(nel):
            coord1,coord2 = get_coord_from_edof(edof[i,:],dof,element_type)

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
            nodes = get_node_elements(coord,scale,alpha)
            #plot_window.add_geometry(elements,nodes)
            plot_window.meshes[plot_window.fig].extend(elements)
            plot_window.nodes[plot_window.fig].extend(nodes)
        else:
            #plot_window.add_geometry(elements)
            plot_window.meshes[plot_window.fig].extend(elements)

        return elements

    elif element_type == 3 or element_type == 4:

        meshes = []
        nel = np.size(edof, axis = 0)

        vmin, vmax = np.min(el_values), np.max(el_values)
        #print(coord)
        for i in range(nel):
            coords = get_coord_from_edof(edof[i,:],dof,4)


            mesh = v.Mesh([coord[coords,:],[[0,1,2,3],[4,5,6,7],[0,3,7,4],[1,2,6,5],[0,1,5,4],[2,3,7,6]]],alpha=alpha,c=color).lw(1)
            #mesh.info = f"Mesh nr. {i}"
            mesh.name = f"Mesh nr. {i+1}"
            meshes.append(mesh)

            if el_values is not None and np.size(el_values, axis = 1) == 1:
                el_values_array = np.zeros((1,6))[0,:]
                el_values_array[0] = el_values[i]
                el_values_array[1] = el_values[i]
                el_values_array[2] = el_values[i]
                el_values_array[3] = el_values[i]
                el_values_array[4] = el_values[i]
                el_values_array[5] = el_values[i]
                #if title is not None:
                #    mesh.cmap(colormap, el_values_array, on="cells", vmin=vmin, vmax=vmax).addScalarBar(title=title,horizontal=True,useAlpha=False,titleFontSize=16)
                #else:
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
                #if title is not None:
                #    mesh.cmap(colormap, el_values_array, on="points", vmin=vmin, vmax=vmax).addScalarBar(title=title,horizontal=True,useAlpha=False,titleFontSize=16)
                #else:
                mesh.cmap(colormap, el_values_array, on="points", vmin=vmin, vmax=vmax)
        #if export is not None:
        #    v.io.write(mesh, export+".vtk")

        if render_nodes == True:
            nodes = get_node_elements(coord,scale,alpha)
            #plot_window.add_geometry(meshes,nodes)
            
            #plot_window.meshes = np.append(meshes, nodes, axis=0)
            #plot_window.meshes.extend(meshes)
            plot_window.meshes[plot_window.fig].extend(meshes)
            plot_window.nodes[plot_window.fig].extend(nodes)
            #plot_window.meshes.extend(nodes)
            #print("Meshes are ",np.size(plot_window.meshes, axis=0),"X",np.size(plot_window.meshes, axis=1))
            print("Adding mesh to figure ",plot_window.fig+1)
        else:
            #plot_window.add_geometry(meshes)
            #plot_window.meshes.extend(meshes)
            plot_window.meshes[plot_window.fig].extend(meshes)
            #print("Meshes are ",np.size(plot_window.meshes, axis=0),"X",np.size(plot_window.meshes, axis=1))
            print("Adding mesh to figure ",plot_window.fig+1)
        return meshes

    elif element_type == 5:
        ncoord = np.size(coord, axis = 0)
        nel = np.size(edof, axis = 0)
        elements = []
        
        res = 4

        vmin, vmax = np.min(el_values), np.max(el_values)

        el_values_array = np.zeros((1,4*res))[0,:]

        for i in range(nel):
            coord1,coord2 = get_coord_from_edof(edof[i,:],dof,5)

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
            nodes = get_node_elements(coord,scale,alpha)
            #plot_window.add_geometry(elements,nodes)
            plot_window.meshes[plot_window.fig].extend(elements)
            plot_window.nodes[plot_window.fig].extend(nodes)
        else:
            #plot_window.add_geometry(elements)
            plot_window.meshes[plot_window.fig].extend(elements)

        return elements

    elif element_type == 6:
        meshes = []
        nel = np.size(edof, axis = 0)

        vmin, vmax = np.min(el_values), np.max(el_values)
        #print(coord)
        for i in range(nel):
            coords = get_coord_from_edof(edof[i,:],dof,6)
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
            nodes = get_node_elements(coord,scale,alpha)
            #plot_window.add_geometry(meshes,nodes)
            plot_window.meshes[plot_window.fig].extend(meshes)
            plot_window.nodes[plot_window.fig].extend(nodes)
        else:
            #plot_window.add_geometry(meshes)
            plot_window.meshes[plot_window.fig].extend(meshes)

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
def draw_displaced_mesh(
    edof,
    coord,
    dof,
    element_type,
    a,
    el_values=None,
    colormap='jet',
    scale=0.02,
    alpha=1,
    def_scale=1,
    nseg=2,
    render_nodes=False,
    color='white',
    offset = [0, 0, 0],
    merge=False,
    t=None,
    ):

    app = init_app()
    plot_window = VedoPlotWindow.instance().plot_window

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
            coord1,coord2 = get_coord_from_edof(edof[i,:],dof,element_type)

            spring = v.Spring([coord[coord1,0],0,0],[coord[coord2,0],0,0],r=scale*1.5).alpha(alpha)
            spring.info = f"Spring nr. {i}"
            elements.append(spring)

        if render_nodes == True:
            nodes = get_node_elements(coord,scale,alpha)
            #plot_window.add_geometry(elements,nodes)
            plot_window.meshes[plot_window.fig].extend(elements)
            plot_window.nodes[plot_window.fig].extend(nodes)
        else:
            #plot_window.add_geometry(elements)
            plot_window.meshes[plot_window.fig].extend(elements)

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
            a_dx, a_dy, a_dz = get_a_from_coord(i,6,a,def_scale)

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
            coord1,coord2 = get_coord_from_edof(edof[i,:],dof,element_type)


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
            #plot_window.add_geometry(def_elements,def_nodes)
            plot_window.meshes[plot_window.fig].extend(def_elements)
            plot_window.nodes[plot_window.fig].extend(def_nodes)
        else:
            #plot_window.add_geometry(def_elements)
            plot_window.meshes[plot_window.fig].extend(def_elements)

        return def_elements

    elif element_type == 3:
        print("Displaced mesh for flow elements is not supported")
        sys.exit()

    elif element_type == 4:
        ncoord = np.size(coord, axis = 0)
        nnode = np.size(coord, axis = 0)

        ex,ey,ez = cfc.coordxtr(edof,coord,dof)

        coord, topo, node_dofs = convert_to_node_topo(edof,ex,ey,ez,ignore_first=False)

        def_nodes = []
        def_coord = np.zeros([ncoord,3])
        """
        for i in range(nnode):
        #a_dx, a_dy, a_dz = get_a_from_coord(i,3,a,def_scale)
        #x = coord[i][0]+a_dx
        #y = coord[i][1]+a_dy
        #z = coord[i][2]+a_dz
        #def_coord[i] = [x,y,z]
            def_coord[i,0] = a[i*3]
            def_coord[i,1] = a[i*3+1]
            def_coord[i,2] = a[i*3+2]
        """
        for i in range(0, ncoord):
            #if a.any() == None:
            #    x = coord[i,0]
            #    y = coord[i,1]
            #    z = coord[i,2]
            #else:
            a_dx, a_dy, a_dz = get_a_from_coord(i,3,a,def_scale)

            x = coord[i][0]+a_dx
            y = coord[i][1]+a_dy
            z = coord[i][2]+a_dz

            def_coord[i] = [x,y,z]

            #def_nodes.append(v.Sphere(c='white').scale(1.5*scale).pos([x,y,z]).alpha(alpha))

        #meshes = []
        nel = np.size(edof, axis = 0)
        
        vmin, vmax = np.min(el_values), np.max(el_values)

        #print(topo)

        mesh = v.Mesh([coord, topo]).lw(1)
        """
        for i in range(nel):
            coords = get_coord_from_edof(edof[i,:],dof,4)

            mesh = v.Mesh([def_coord[coords,:],[[0,1,2,3],[4,5,6,7],[0,3,7,4],[1,2,6,5],[0,1,5,4],[2,3,7,6]]],alpha=alpha).lw(1)
            #mesh.flag().color('m')
            #mesh.name = f"Mesh nr. {i+1}"
            meshes.append(mesh)

            
            
            if el_values is not None and np.size(el_values, axis = 1) == 1:
                el_values_array = np.zeros((1,6))[0,:]
                el_values_array[0] = el_values[i]
                el_values_array[1] = el_values[i]
                el_values_array[2] = el_values[i]
                el_values_array[3] = el_values[i]
                el_values_array[4] = el_values[i]
                el_values_array[5] = el_values[i]
                #if title is not None:
                #    mesh.cmap(colormap, el_values_array, on="cells", vmin=vmin, vmax=vmax).addScalarBar(title=title,horizontal=True,useAlpha=False,titleFontSize=16)
                #else:
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
                #if title is not None:
                #    mesh.cmap(colormap, el_values_array, on="points", vmin=vmin, vmax=vmax).addScalarBar(title=title,horizontal=True,useAlpha=False,titleFontSize=16)
                #else:
                mesh.cmap(colormap, el_values_array, on="points", vmin=vmin, vmax=vmax)
        """
        #if export is not None:
        #    v.io.write(mesh, export+".vtk")

        #if merge == True:
        #    meshes = v.merge(meshes)

        #mesh = v.merge(meshes)
        print('Number of Vedo mesh coordinates: ',mesh.N())

        if render_nodes == True:
            nodes = get_node_elements(coord,scale,alpha)
            #plot_window.add_geometry(meshes,nodes)
            #plot_window.meshes.extend(meshes)
            #plot_window.meshes[plot_window.fig].extend(meshes)
            #plot_window.nodes[plot_window.fig].extend(nodes)
            plot_window.meshes[plot_window.fig] += mesh
            plot_window.nodes[plot_window.fig] += nodes
            #plot_window.meshes.extend(nodes)
            #print("Meshes are ",np.size(plot_window.meshes, axis=0),"X",np.size(plot_window.meshes, axis=1))
            print("Adding mesh to figure ",plot_window.fig+1)
        else:
            #plot_window.add_geometry(meshes)
            plot_window.meshes[plot_window.fig].append(mesh)
            #plot_window.meshes[plot_window.fig].extend(mesh)
            #plot_window.meshes[plot_window.fig] += mesh
            #plot_window.meshes.extend(meshes)
            #print("Meshes are ",np.size(plot_window.meshes, axis=0),"X",np.size(plot_window.meshes, axis=1))
            print("Adding mesh to figure ",plot_window.fig+1)
        


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
            a_dx, a_dy, a_dz = get_a_from_coord(i,6,a,def_scale)

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
            coord1,coord2 = get_coord_from_edof(edof[i,:],dof,5)

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
            plot_window.meshes[plot_window.fig].extend(def_elements)
            plot_window.nodes[plot_window.fig].extend(def_nodes)
            #plot_window.add_geometry(def_elements,def_nodes,merge=merge)
        else:
            plot_window.meshes[plot_window.fig].extend(def_elements)
            #plot_window.add_geometry(def_elements,merge=merge)

        return def_elements

    elif element_type == 6:
        print("Displaced mesh for plate elements is not supported")
        sys.exit()

def test(edof,
    ex,
    ey,
    ez,
    a=None,
    el_values=None,
    colormap='jet',
    scale=0.02,
    alpha=1,
    def_scale=1,
    nseg=2,
    color='white',
    offset = [0, 0, 0],
    merge=False,
    t=None):

    app = init_app()
    plot_window = VedoPlotWindow.instance().plot_window


    coord, topo, node_dofs = convert_to_node_topo(edof,ex,ey,ez,ignore_first=False)
    #return coords, topo, node_dofs

    if a is None:
        a = np.zeros([np.size(coord, axis = 0)*np.size(coord, axis = 1),1])

    nnode = np.size(coord, axis = 0)
    nel = np.size(topo, axis = 0)
    ndof = np.size(topo, axis = 1)
    print(np.size(topo, axis = 0))
    print(np.size(topo, axis = 1))
    print(np.size(a, axis = 0))
    #print(np.size(a, axis = 1))
    #print(coord[0][0])
    #print(coord[0][1])
    #print(coord[0][2])

    """
    if ct == vtk.VTK_HEXAHEDRON:
                    cell = vtk.vtkHexahedron()
                elif ct == vtk.VTK_TETRA:
                    cell = vtk.vtkTetra()
                elif ct == vtk.VTK_VOXEL:
                    cell = vtk.vtkVoxel()
                elif ct == vtk.VTK_WEDGE:
                    cell = vtk.vtkWedge()
                elif ct == vtk.VTK_PYRAMID:
                    cell = vtk.vtkPyramid()
                elif ct == vtk.VTK_HEXAGONAL_PRISM:
                    cell = vtk.vtkHexagonalPrism()
                elif ct == vtk.VTK_PENTAGONAL_PRISM:
                    cell = vtk.vtkPentagonalPrism()
    """

    #def_nodes = []
    def_coord = np.zeros([nnode,3])
    #celltype = [vtk.VTK_HEXAHEDRON] * nel

    #pdata = np.zeros((nnode), dtype=float)

    for i in range(nnode):
        #a_dx, a_dy, a_dz = get_a_from_coord(i,3,a,def_scale)
        #x = coord[i][0]+a_dx
        #y = coord[i][1]+a_dy
        #z = coord[i][2]+a_dz
        #def_coord[i] = [x,y,z]
        def_coord[i,0] = a[i*3]
        def_coord[i,1] = a[i*3+1]
        def_coord[i,2] = a[i*3+2]


    #meshes = []
    #nel = np.size(edof, axis = 0)

    #for i, dofs in enumerate(node_dofs):
        #v = u0[dofs-1]
        #pdata[i] = np.linalg.norm(v)

    #ugrid = v.UGrid([coord, topo, celltype])
    print(coord[0])
    print(topo)
    mesh = v.Mesh([coord, topo[0]]).lw(1)
    #ugrid.pointdata["mag"] = pdata

    #mesh = ugrid.tomesh()
    
    #for i in range(nel):
    #coords = get_coord_from_edof(edof[i,:],dof,4)

    #print(topo)

    #mesh = v.UGrid([def_coord, topo, [10]]).lw(10)
    #[[0,1,2,3],[4,5,6,7],[0,3,7,4],[1,2,6,5],[0,1,5,4],[2,3,7,6]],
    mesh.color(c='red')
    #mesh.name = f"Mesh nr. {i+1}"
    #meshes.append(mesh)

            


    #plot_window.meshes[plot_window.fig].extend(mesh)
    plot_window.meshes[plot_window.fig] += [mesh]
    print("Adding mesh to figure ",plot_window.fig+1)





# Creates a deformed mesh for rendering, see element types above
#def draw_vectors(edof,coord,dof,element_type,vect):




### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
# Functions regarding animations

def animate(
    edof,
    coord,
    dof,
    element_type,
    a,
    steps,
    def_scale=1,
    alpha=1
    ):
    app = init_app()
    plot_window = VedoPlotWindow.instance().plot_window

    v.settings.immediateRendering = True

    camera = dict(viewAngle=30)

    timesteps = np.arange(0, 1+1/(steps), 1/(steps))
    print(timesteps)
    #print(start)
    #print(end)
    #print(steps)
    
    #if self.elements != None:
    #    element_type = 1
    #elif self.mesh != None:
    #    element_type = 2

    #dt = 0.1

    #t = np.arange(0.0, 10.0, dt)

    #pb = v.ProgressBar(0, len(t), c="b")


    if element_type == 4:
        ncoord = np.size(coord, axis = 0)
        nel = np.size(edof, axis = 0)

        def_coord = np.zeros([ncoord,3])
        #def_coord = np.zeros((ncoord,3,steps))


        meshes = []


        for i in range(nel):
            coords = get_coord_from_edof(edof[i,:],dof,4)


            mesh = v.Mesh([def_coord[coords,:],[[0,1,2,3],[4,5,6,7],[0,3,7,4],[1,2,6,5],[0,1,5,4],[2,3,7,6]]],alpha=alpha).lw(1)
            #mesh.info = f"Mesh nr. {i}"
            #mesh.name = f"Mesh nr. {i+1}"
            meshes.append(mesh)

        mesh = v.merge(meshes)

        #plt = v.show(mesh, axes=4, interactive=0)

        plt = v.Plotter(axes=4, interactive=0)

        plt.show(mesh)





        for t in timesteps:

            for i in range(0, ncoord):
                a_dx, a_dy, a_dz = get_a_from_coord(i,3,a,def_scale)

                a_dx = t*a_dx
                a_dy = t*a_dy
                a_dz = t*a_dz

                x_step = a_dx/(steps)
                y_step = a_dy/(steps)
                z_step = a_dz/(steps)

                for j in range(0, steps):

                    x = coord[i,0]+x_step*j
                    y = coord[i,1]+y_step*j
                    z = coord[i,2]+z_step*j

                    def_coord[i,:] = [x,y,z]

            
            meshes = []
            
            
            #vmin, vmax = np.min(el_values), np.max(el_values)
            for i in range(nel):
                coords = get_coord_from_edof(edof[i,:],dof,4)

                mesh = v.Mesh([def_coord[coords,:],[[0,1,2,3],[4,5,6,7],[0,3,7,4],[1,2,6,5],[0,1,5,4],[2,3,7,6]]],alpha=alpha).lw(1)
                meshes.append(mesh)
                #for j in range(steps):
                    #mesh = v.Mesh([def_coord[coords,:,j],[[0,1,2,3],[4,5,6,7],[0,3,7,4],[1,2,6,5],[0,1,5,4],[2,3,7,6]]],alpha=alpha)
                    #meshes[i,j] = mesh
                #mesh = v.Mesh([def_coord[coords,:],[[0,1,2,3],[4,5,6,7],[0,3,7,4],[1,2,6,5],[0,1,5,4],[2,3,7,6]]],alpha=alpha)
                #mesh.info = f"Mesh nr. {i}"
                #meshes.append(mesh)
            mesh = v.merge(meshes)
            plt.clear()
            plt += mesh
            plt.render(resetcam=True)
            #plt.show(mesh)
            


        v.interactive().close()

        v.settings.immediateRendering = False

### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
# Functions for importing/exporting

def import_mat(file,list=None):
    data = {} # dict to be returned by loadmat
    loadmat(file, data, variable_names=list)

    if list == None: # remove unnecessary entries
        keys = ['__header__', '__version__', '__globals__']
        for key in keys:
            data.pop(key)
        return data # returns the data, random ordering
    else: # supplying a 'list' is recommended
        ret = dict.fromkeys(list,None) # returns the data, ordering by user
        for key,val in ret.items():
            x = data[key] 
            if key == 'edof' or key == 'Edof' or key == 'EDOF':
                x = np.delete(x,0,1) # auto convert Edof from Matlab to Python
            yield x # returns the data one-by-one, ordering by user

def export_vtk(file, meshes):
    mesh = v.merge(meshes)
    #for i in range(len(meshes)):
    v.io.write(mesh, file+".vtk")

### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
# Tools, used in this file but can be accessed by a user as well (see exv4a.py/exv4b.py)

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

def convert_to_node_topo(edof, ex, ey, ez, n_dofs_per_node=3, ignore_first=False):
    """
    Routine to convert dof based topology and element coordinates to node based
    topology required for visualisation with VTK and other visualisation frameworks

    :param array edof: element topology [nel x (n_dofs_per_node)|(n_dofs_per_node+1)*n_nodes ]
    :param array ex: element x coordinates [nel x n_nodes]
    :param array ey: element y coordinates [nel x n_nodes]
    :param array ez: element z coordinates [nel x n_nodes]
    :param array n_dofs_per_node: number of dofs per node. (default = 3)
    :param boolean ignore_first: ignore first column of edof. (default = True)
    :return array coords: Array of node coordinates. [n_nodes x 3]
    :return array topo: Node topology. [nel x n_nodes]
    :return array node_dofs: Dofs for each node. [n_nodes x n_dofs_per_node]
    """

    node_hash_coords = {}
    node_hash_numbers = {}
    node_hash_dofs = {}
    el_hash_dofs = []

    nel, cols = edof.shape

    if ignore_first:
        tot_dofs = cols-1
    else:
        tot_dofs = cols

    n_nodes = int(tot_dofs / n_dofs_per_node)

    print("cols    =", tot_dofs)
    print("nel     =", nel)
    print("n_nodes =", n_nodes)

    for elx, ely, elz, dofs in zip(ex, ey, ez, edof):

        if ignore_first:
            el_dofs = dofs[1:]
        else:
            el_dofs = dofs

        # 0 1 2  3 4 5  6 7 8  9 12 11 

        el_dof = np.zeros((n_nodes, n_dofs_per_node), dtype=int)
        el_hash_topo = []

        for i in range(n_nodes):
            el_dof[i] = el_dofs[ (i*n_dofs_per_node):((i+1)*n_dofs_per_node) ]
            node_hash_coords[hash(tuple(el_dof[i]))] = [elx[i], ely[i], elz[i]]
            node_hash_numbers[hash(tuple(el_dof[i]))] = -1
            node_hash_dofs[hash(tuple(el_dof[i]))] = el_dof[i]
            el_hash_topo.append(hash(tuple(el_dof[i])))

        el_hash_dofs.append(el_hash_topo)

    coord_count = 0

    coords = []
    node_dofs = []

    for node_hash in node_hash_numbers.keys():
        node_hash_numbers[node_hash] = coord_count
        node_dofs.append(node_hash_dofs[node_hash])
        coord_count +=1

        coords.append(node_hash_coords[node_hash])

    topo = []

    for el_hashes in el_hash_dofs:
        """
        topo.append([
            node_hash_numbers[el_hashes[0]], 
            node_hash_numbers[el_hashes[1]], 
            node_hash_numbers[el_hashes[2]], 
            node_hash_numbers[el_hashes[3]]
            ]
        )
        """
        topo.append(
            
            [node_hash_numbers[el_hashes[0]], 
            node_hash_numbers[el_hashes[1]], 
            node_hash_numbers[el_hashes[2]], 
            node_hash_numbers[el_hashes[3]]])
        topo.append(
            [node_hash_numbers[el_hashes[4]],
            node_hash_numbers[el_hashes[5]],
            node_hash_numbers[el_hashes[6]],
            node_hash_numbers[el_hashes[7]]])
        topo.append(
            [node_hash_numbers[el_hashes[0]],
            node_hash_numbers[el_hashes[3]],
            node_hash_numbers[el_hashes[7]],
            node_hash_numbers[el_hashes[4]]])
        topo.append(
            [node_hash_numbers[el_hashes[1]],
            node_hash_numbers[el_hashes[2]],
            node_hash_numbers[el_hashes[6]],
            node_hash_numbers[el_hashes[5]]])
        topo.append(
            [node_hash_numbers[el_hashes[0]],
            node_hash_numbers[el_hashes[1]],
            node_hash_numbers[el_hashes[5]],
            node_hash_numbers[el_hashes[4]]])
        topo.append(
            [node_hash_numbers[el_hashes[2]],
            node_hash_numbers[el_hashes[3]],
            node_hash_numbers[el_hashes[7]],
            node_hash_numbers[el_hashes[6]]])
        
            
        
        
        
    #mesh = v.Mesh([def_coord[coords,:],[[0,1,2,3],[4,5,6,7],[0,3,7,4],[1,2,6,5],[0,1,5,4],[2,3,7,6]]],alpha=alpha).lw(1)

    return coords, topo, node_dofs




### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
# Functions for handling rendering


def figure(fig):
    app = init_app()
    plot_window = VedoPlotWindow.instance().plot_window

    if fig < 1:
        print("Please give a positive integer (> 0)")
        sys.exit()
    else:
        plot_window.fig = fig - 1

    print("Selecting figure ",fig)
    if fig > 1:
        while len(plot_window.meshes) < plot_window.fig + 1:
            #plot_window.proj.append([])
            plot_window.meshes.append([])
            plot_window.nodes.append([])

            plot_window.msg.append([])
            plot_window.proj.append([])
            plot_window.rulers.append([])



# Start Calfem-vedo visualization
def show_and_wait():
    app = init_app()
    plot_window = VedoPlotWindow.instance().plot_window
    plot_window.render()
