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
import vedo_utils as vdu
import vtk

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
        self.plt = {}
        self.rulers = [[]]
        self.vectors = [[]]

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
        v.settings.useSSAO         = True
        v.settings.visibleGridEdges = True
        #v.settings.useParallelProjection = True

    def click(self,evt):
        #if evt.isAssembly: # endast för testning ifall en assembly skapas
        #    print('assembly')
        #    self.click_msg.text(evt.actor.info)
        if evt.actor:
            #print(evt.actor.mapper())
            #if self.silcont != [] or self.silcont != [None]:
            # Silouette
            sil = evt.actor.silhouette().lineWidth(5).c('red5')
            self.plt[evt.title].remove(self.silcont.pop()).add(sil)
            self.silcont.append(sil)

            #self.silcont[0].c('black')

            # Color
            #evt.actor.c('red5')

            #sil = evt.actor.silhouette().lineWidth(5).c('red5')
            #self.plt[evt.title].remove(self.silcont.pop()).add(evt.actor)
            #self.silcont.append(evt.actor)
            #print('Title: ',evt.title,', Plotter: ',self.plt[evt.title])
            self.click_msg.text(evt.actor.name)
            
            

            #self.plt[evt.title].remove(self.silcont.pop()).add(sil)
            #evt.interactor.add(sil)
            #evt.interactor.pop()
            #.remove(self.silcont.pop())
            #.add(sil)
            #self.plotter.remove(self.silcont.pop()).add(sil)
            #self.silcont.append(sil)
        else:
            self.click_msg.text('')
            if self.silcont != [None]:
                sil = None
                self.plt[evt.title].remove(self.silcont.pop()).add(sil)
                self.silcont.append(sil)
                
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
            #print('Figure text: ',self.msg[i])
            #print('Projections: ',self.proj[i])
            #self.plt.append(plt)
            self.plt[f'Figure {i+1} - CALFEM vedo visualization tool'] = plt

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
    pos=[0.8,0.05],
    #pos="bottom-right",
    font_size=24,
    color='black',
    #sx=1000,
    #sy=1000
    #size=(100, 50)
    ):
    app = init_app()
    plot_window = VedoPlotWindow.instance().plot_window

    fig = plot_window.fig
    plot_window.meshes[fig][0].addScalarBar(pos=pos, titleFontSize=font_size)

    msg = v.Text2D(label, pos='bottom-right', alpha=1, c=color)
    plot_window.msg[plot_window.fig] += [msg]

# Add text to a renderer
def add_text(
    text,
    color='black',
    pos='top-middle',
    size=1
    ):
    app = init_app()
    plot_window = VedoPlotWindow.instance().plot_window

    #while len(plot_window.msg) < plot_window.fig + 1:
        

    #plot_window.msg.append([])

    msg = v.Text2D(text, pos=pos, alpha=1, c=color)
    plot_window.msg[plot_window.fig] += [msg]

def add_text_3D(text,pos=(0,0,0),color='black',size=1,alpha=1):
    app = init_app()
    plot_window = VedoPlotWindow.instance().plot_window

    msg = v.Text3D(text, pos=pos, s=size, font='', hspacing=1.15, vspacing=2.15, depth=0, italic=False, justify='bottom-left', c=color, alpha=alpha, literal=False)
    plot_window.msg[plot_window.fig] += [msg]

# Add silhouette with or without measurements to a renderer
def add_projection(color='black',plane='xy',offset=-1,rulers=False):
    app = init_app()
    plot_window = VedoPlotWindow.instance().plot_window

    

    #print("Size of projections: ",len(plot_window.proj))
    #while len(plot_window.proj) < plot_window.fig + 1:
        
    #if rulers == True:
        #while len(plot_window.rulers) < plot_window.fig + 1:
            
    #print("Size of projections after loop: ",len(plot_window.proj))

    if plane == 'xy':
        #assem = v.merge(plot_window.meshes[plot_window.fig])
        assem = plot_window.meshes[plot_window.fig]
        plot_window.meshes[plot_window.fig]
        proj = assem.projectOnPlane('z').z(offset).silhouette('2d').c(color)
        plot_window.proj[plot_window.fig] += [proj]
        if rulers == True:
            ruler = v.addons.RulerAxes(proj, xtitle='', ytitle='', ztitle='', xlabel='', ylabel='', zlabel='', xpad=1, ypad=1, zpad=1, font='Normografo', s=None, italic=0, units='m', c=color, alpha=1, lw=1, precision=3, labelRotation=0, axisRotation=0, xycross=True)
            plot_window.rulers[plot_window.fig] += [ruler]
        """
        for i in range(len(plot_window.meshes[plot_window.fig])):
            proj = plot_window.meshes[plot_window.fig][i].clone().projectOnPlane('z').c(color).z(offset)
            plot_window.proj[plot_window.fig] += [proj]
        """
        #plot_window.meshes[plot_window.fig][0].addShadow(plane='x')
    elif plane == 'xz':
        meshes = []
        for i in range(len(plot_window.meshes[plot_window.fig])):
            #print(plot_window.meshes[plot_window.fig,i])
            meshes.append(plot_window.meshes[plot_window.fig][i].clone())
        #assem = v.merge(plot_window.meshes[plot_window.fig])
        assem = v.merge(meshes)
        #assem = v.Assembly(plot_window.meshes[plot_window.fig])
        #assem = plot_window.meshes[plot_window.fig][0]
        proj = assem.projectOnPlane('y').y(offset).silhouette('2d').c(color)
        plot_window.proj[plot_window.fig] += [proj]
        if rulers == True:
            ruler = v.addons.RulerAxes(proj, xtitle='', ytitle='', ztitle='', xlabel='', ylabel='', zlabel='', xpad=1, ypad=1, zpad=1, font='Normografo', s=None, italic=0, units='m', c=color, alpha=1, lw=1, precision=3, labelRotation=0, axisRotation=0, xycross=True)
            plot_window.rulers[plot_window.fig] += [ruler]
    elif plane == 'yz':
        #assem = v.merge(plot_window.meshes[plot_window.fig])
        assem = v.Assembly(plot_window.meshes[plot_window.fig])
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

    ruler = v.addons.RulerAxes(assem, xtitle=xtitle, ytitle=ytitle, ztitle=ztitle, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, xpad=0.1, ypad=0.1, zpad=0.1, font='Normografo', s=None, italic=0, units='m', c=(0,0,0), alpha=alpha, lw=1, precision=3, labelRotation=0, axisRotation=0, xycross=False)

    plot_window.rulers[plot_window.fig] += [ruler]

#Add vector field
def add_vectors(edof,coord,dof,v,element_type):
    app = init_app()
    plot_window = VedoPlotWindow.instance().plot_window

    nel = np.size(edof, axis = 0)
    x = np.zeros((nel,1))
    y = np.zeros((nel,1))
    z = np.zeros((nel,1))
    for i in range(nel):
        coords = vdu.get_coord_from_edof(edof[i],dof,element_type)
        x[i] = np.average(coord[coords,0])
        y[i] = np.average(coord[coords,1])
        z[i] = np.average(coord[coords,2])

        mag = np.sqrt(v[i][0]**2 + v[i][1]**2 + v[i][2]**2)
        alpha = np.arccos(v[i][0]/mag)
        beta = np.arccos(v[i][1]/mag)
        gamma = np.arccos(v[i][2]/mag)
        #print('mag',mag)
        #print('alpha',alpha)
        #print('beta',beta)
        #print('gamma',gamma)
        #p1 = [-np.cos(alpha)]
        #for j in zip(coords):
        #    coord[j]
    #arrow = v.Arrow().scale(0.04)
    #field = v.Glyph([x,y,z],arrow,orientationArray=vectors)
    #field = v.Tensors([x,y,z],source='Arrow')
    plot_window.vectors[plot_window.fig] += [field]


### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
# Functions for plotting geometries, meshes & results from CALFEM calculations

def draw_geometry(points=None,lines=None,surfaces=None,scale=0.05):
    app = init_app()
    plot_window = VedoPlotWindow.instance().plot_window

    if surfaces == None and lines == None and points == None:
        print("Please input either points, lines or surfaces")
        sys.exit()
    else:
        if surfaces is not None:
            print('points',points)
            print('lines',lines)
            print('surfaces',surfaces)

            pts = []
            p_text = []
            for key,val in points.items():
                pts.append(v.Point(pos=val[0], r=12, c='black', alpha=1))
                p_text.append(v.Text3D(key, pos=val[0], s=scale, font='Normografo', hspacing=1.15, vspacing=2.15, depth=0, italic=False, justify='bottom-left', c='black', alpha=1, literal=False))

            plot_window.meshes[plot_window.fig].append(pts)
            plot_window.meshes[plot_window.fig].append(p_text)

            l = []
            l_text = []
            for key,val in lines.items():
                p1 = points[val[1][0]][0]
                p2 = points[val[1][1]][0]
                l.append(v.Lines([p1], [p2], c='k4', alpha=1, lw=1, res=2))
                l_text.append(v.Text3D(key, pos=[(p1[0]+0.5*(p2[0]-p1[0])), (p1[1]+0.5*(p2[1]-p1[1])), (p1[2]+0.5*(p2[2]-p1[2]))], s=scale, font='Normografo', hspacing=1.15, vspacing=2.15, depth=0, italic=False, justify='bottom-left', c='black', alpha=1, literal=False))

            plot_window.meshes[plot_window.fig].append(l)
            plot_window.meshes[plot_window.fig].append(l_text)

            surf = []
            s_text = []
            for key,val in surfaces.items():
                ### NOTE: only 4 point surfaces implemented
                l12 = lines[val[1][0]][1]
                l34 = lines[val[1][2]][1]

                p1=points[l12[0]][0]
                p2=points[l12[1]][0]
                p3=points[l34[0]][0]
                p4=points[l34[1]][0]

                x = np.average([p1[0],p2[0],p3[0],p4[0]])
                y = np.average([p1[1],p2[1],p3[1],p4[1]])
                z = np.average([p1[2],p2[2],p3[2],p4[2]])

                print(x)
                print(y)
                print(z)
                #print(p4)
                #sys.exit()
                
                surf.append(v.Plane(pos=(x, y, z), normal=(0, 0, 1), sx=1, sy=None, c='gray6', alpha=1))
                s_text.append(v.Text3D(key, pos=[x, y, z], s=scale, font='Normografo', hspacing=1.15, vspacing=2.15, depth=0, italic=False, justify='bottom-left', c='black', alpha=1, literal=False))

            plot_window.meshes[plot_window.fig].append(surf)
            plot_window.meshes[plot_window.fig].append(s_text)

            
        elif lines is not None and points is not None:
            pts = []
            p_text = []
            for key,val in points.items():
                pts.append(v.Point(pos=val[0], r=12, c='black', alpha=1))
                p_text.append(v.Text3D(key, pos=val[0], s=scale, font='Normografo', hspacing=1.15, vspacing=2.15, depth=0, italic=False, justify='bottom-left', c='black', alpha=1, literal=False))

            plot_window.meshes[plot_window.fig].append(pts)
            plot_window.meshes[plot_window.fig].append(p_text)

            l = []
            l_text = []
            for key,val in lines.items():
                p1 = points[val[1][0]][0]
                p2 = points[val[1][1]][0]
                l.append(v.Lines([p1], [p2], c='k4', alpha=1, lw=1, res=2))
                l_text.append(v.Text3D(key, pos=[(p1[0]+0.5*(p2[0]-p1[0])), (p1[1]+0.5*(p2[1]-p1[1])), (p1[2]+0.5*(p2[2]-p1[2]))], s=scale, font='Normografo', hspacing=1.15, vspacing=2.15, depth=0, italic=False, justify='bottom-left', c='black', alpha=1, literal=False))

            plot_window.meshes[plot_window.fig].append(l)
            plot_window.meshes[plot_window.fig].append(l_text)
        elif points is not None:
            pts = []
            text = []

            for key,val in points.items():
                pts.append(v.Point(pos=val[0], r=12, c='black', alpha=1))
                text.append(v.Text3D(key, pos=val[0], s=scale, font='Normografo', hspacing=1.15, vspacing=2.15, depth=0, italic=False, justify='bottom-left', c='black', alpha=1, literal=False))

            plot_window.meshes[plot_window.fig].append(pts)
            plot_window.meshes[plot_window.fig].append(text)
        elif lines is not None:
            print("Please provide point coordinates along with lines")
            sys.exit()

        

    #print(points)

    #lines = []

    #for i in range(len(points)):
        #print(i)
    #    for j in range(len(points)):
    #        lines.append([i,j])

    
    #geometry
    #pts = v.Points(points).ps(10)#.renderPointsAsSpheres()
    #geometry = v.utils.geometry(pts.tomesh())
    #dly = v.delaunay2D(pts, mode='fit').lw(1)

    #if vol == True:
        #volume = v.mesh2Volume(geometry).mode(4)
        #plot_window.meshes[plot_window.fig].append(volume)
    #    plot_window.meshes[plot_window.fig].append(geometry)
    #else:
    

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
    bc=None,
    eq=None
    ):

    app = init_app()
    plot_window = VedoPlotWindow.instance().plot_window





    if element_type == 1:           

        nel = np.size(edof, axis = 0)
        elements = []

        for i in range(nel):
            coord1,coord2 = vdu.get_coord_from_edof(edof[i,:],dof,element_type)

            #print(coord[coord1,0])
            #print(coord[coord2,0])
            #spring = v.Spring([coord[coord1,0],0,0],[coord[coord2,0],0,0])
            spring = v.Spring([coord[coord1,0],0,0],[coord[coord2,0],0,0],r=1.5*scale).alpha(alpha)
            spring.name = f"Element nr. {i}"
            elements.append(spring)

        #print(elements,nodes)
        if render_nodes == True:
            nodes = vdu.get_node_elements(coord,scale*0.5,alpha)
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

        #el_values_array = np.zeros((1,4*res))[0,:]
        #el_values_array = np.zeros((1,res))[0,:]

        for i in range(nel):
            coord1,coord2 = vdu.get_coord_from_edof(edof[i,:],dof,element_type)

            bar = v.Cylinder([[coord[coord1,0],coord[coord1,1],coord[coord1,2]],[coord[coord2,0],coord[coord2,1],coord[coord2,2]]],r=scale,res=res,c=color).alpha(alpha)
            elements.append(bar)


            if el_values is not None:
                bar.info = f"Bar nr. {i}, max el. value {el_values[i]}"
                '''
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
                '''
                #bar.cmap(colormap, el_values_array, on="points", vmin=vmin, vmax=vmax)
                #bar.cmap(colormap, [el], on="cells", vmin=vmin, vmax=vmax)
            else:
                bar.info = f"Bar nr. {i}"
        if render_nodes == True:
            nodes = vdu.get_node_elements(coord,scale,alpha,dof)
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
            coords = vdu.get_coord_from_edof(edof[i,:],dof,4)


            mesh = v.Mesh([coord[coords,:],[[0,1,2,3],[4,5,6,7],[0,3,7,4],[1,2,6,5],[0,1,5,4],[2,3,7,6]]],alpha=alpha).lw(1)
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
            if element_type == 3:
                nodes = vdu.get_node_elements(coord,scale,alpha,dof,bc,dofs_per_node=1)
            elif element_type == 4:
                nodes = vdu.get_node_elements(coord,scale,alpha,dof,bc,dofs_per_node=3)
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
            coord1,coord2 = vdu.get_coord_from_edof(edof[i,:],dof,5)

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
            #print(dof)
            nodes = vdu.get_node_elements(coord,scale,alpha,dof)
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
            coords = vdu.get_coord_from_edof(edof[i,:],dof,6)
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
            """
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
            """
            #print(new_coord)

            #coord.append(np.array([-;]))

            #plate = v.Mesh([new_coord,[[0,1,2,3],[4,5,6,7],[0,3,7,4],[1,2,6,5],[0,1,5,4],[2,3,7,6]]],alpha=alpha).lw(1)
            #print(f'Element {i}: {coord[i]}')
            plate = v.Mesh([new_coord,[[0,1,2,3]]],alpha=alpha).lw(1)
            plate.name = f"Element nr. {i}"
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
            nodes = vdu.get_node_elements(coord,scale,alpha)
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
    values=None,
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
    vmax=None,
    vmin=None
    ):

    app = init_app()
    plot_window = VedoPlotWindow.instance().plot_window

    if 1 <= element_type <= 6:
        if a is None and values is None:
            nel, ndof_per_el, nnode, ndim, ndof, ndof_per_n = vdu.check_input(edof,coord,dof,element_type,nseg=nseg)
        elif a is None:
            nel, ndof_per_el, nnode, ndim, ndof, ndof_per_n, val = vdu.check_input(edof,coord,dof,element_type,values=values,nseg=nseg)
        elif values is None:
            nel, ndof_per_el, nnode, ndim, ndof, ndof_per_n, ndisp = vdu.check_input(edof,coord,dof,element_type,a,nseg=nseg)
        else:
            nel, ndof_per_el, nnode, ndim, ndof, ndof_per_n, ndisp, val = vdu.check_input(edof,coord,dof,element_type,a,values,nseg=nseg)
    else:
        print("Invalid element type, please declare 'element_type'. The element types are:\n    1 - Spring\n    2 - Bar\n    3 - Flow\n    4 - Solid\n    5 - Beam\n    6 - Plate")
        sys.exit()

    #print(val)

    # Number of elements:                       nel
    # Number of degrees of freedom per element: ndof_per_el
    # Number of nodes:                          nnode
    # Number of dimensions:                     ndim
    # Number of degrees of freedom:             ndof
    # Number of degrees of freedom per node:    ndof_per_n
    # Number of displacements:                  ndisp
    # Element/nodal values:                     val


    if element_type == 1:
        ndof = np.size(dof, axis = 0)*np.size(dof, axis = 1)
        #print(coord[:,1])
        ncoord = np.size(coord, axis = 0)
        #coord[:,0] = coord[:,0] + offset[0]
        #coord[:,1] = coord[:,1] + offset[1]
        #print(coord[:,1])
        #coord[:,2] = coord[:,2] + offset[2]

        def_nodes = []

        nel = np.size(edof, axis = 0)
        elements = []

        def_coord = np.zeros([ncoord,3])

        #print(def_coord)

        for i in range(0, ncoord):
            #if a.any() == None:
            #    x = coord[i,0]
            #    y = coord[i,1]
            #    z = coord[i,2]
            #else:
            #def get_a_from_coord(coord_row_num,num_of_deformations,a,scale=1):
            #a_dx, a_dy, a_dz = vdu.get_a_from_coord(i,2,a,def_scale)
            a_dx = a[i]*def_scale
            #print(a)

            x = coord[i,0]+a_dx
            y = coord[i,1]
            z = coord[i,2]
            #print(def_coord[i])
            #print([x,y,z])
            #print(offset)
            def_coord[i] = [x,y,z]
            def_coord[i] += offset
            #print(def_coord[i])
            #def_nodes.append(v.Point(c='white').scale(scale*0.5).pos(def_coord[i]).alpha(alpha))

        for i in range(nel):
            coord1,coord2 = vdu.get_coord_from_edof(edof[i,:],dof,element_type)

            spring = v.Spring([def_coord[coord1,0],def_coord[coord1,1],def_coord[coord1,2]],[def_coord[coord2,0],def_coord[coord2,1],def_coord[coord2,2]],r=scale*1.5).alpha(alpha)
            spring.info = f"Spring nr. {i}"
            elements.append(spring)

        if render_nodes == True:
            print(def_coord)
            nodes = vdu.get_node_elements(def_coord,scale*0.5,alpha)
            print(nodes)
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
            a_dx, a_dy, a_dz = vdu.get_a_from_coord(i,6,a,def_scale)

            x = coord[i,0]+a_dx
            y = coord[i,1]+a_dy
            z = coord[i,2]+a_dz

            def_coord[i] = [x,y,z]

            def_nodes.append(v.Sphere(c='white').scale(1.5*scale).pos([x,y,z]).alpha(alpha))


        nel = np.size(edof, axis = 0)
        def_elements = []
        res = 4

        if vmin == None and vmax == None:
            vmin, vmax = np.min(values), np.max(values)

        el_values_array = np.zeros((1,4*res))[0,:]

        for i in range(nel):
            coord1,coord2 = vdu.get_coord_from_edof(edof[i,:],dof,element_type)


            bar = v.Cylinder([[def_coord[coord1,0],def_coord[coord1,1],def_coord[coord1,2]],[def_coord[coord2,0],def_coord[coord2,1],def_coord[coord2,2]]],r=scale,res=res,c=color).alpha(alpha)
            bar.info = f"Bar nr. {i}, at [{def_coord[coord1,0]+0.5*(def_coord[coord2,0]-def_coord[coord1,0])},{def_coord[coord1,1]+0.5*(def_coord[coord2,1]-def_coord[coord1,1])},{def_coord[coord1,2]+0.5*(def_coord[coord2,2]-def_coord[coord1,2])}]"
            def_elements.append(bar)
            if values is not None:
                bar.info = bar.info + f", max el. value {values[i]}"
                el_values_array[1] = values[i]
                el_values_array[3] = values[i]
                el_values_array[5] = values[i]
                el_values_array[7] = values[i]
                el_values_array[12] = values[i]
                el_values_array[13] = values[i]
                el_values_array[14] = values[i]
                el_values_array[15] = values[i]

                el_values_array[0] = values[i]
                el_values_array[2] = values[i]
                el_values_array[4] = values[i]
                el_values_array[6] = values[i]
                el_values_array[8] = values[i]
                el_values_array[9] = values[i]
                el_values_array[10] = values[i]
                el_values_array[11] = values[i]

                #bar.cmap(colormap, el_values_array, on="points", vmin=vmin, vmax=vmax)
                bar.cmap(colormap, [values[i],values[i],values[i],values[i],values[i],values[i]], on="cells", vmin=vmin, vmax=vmax)
        if render_nodes == True:
            #plot_window.add_geometry(def_elements,def_nodes)
            plot_window.meshes[plot_window.fig].extend(def_elements)
            plot_window.nodes[plot_window.fig].extend(def_nodes)
        else:
            #plot_window.add_geometry(def_elements)
            plot_window.meshes[plot_window.fig].extend(def_elements)

        return def_elements

    #elif element_type == 3:
    #    print("Displaced mesh for flow elements is not supported")
    #    sys.exit()

    elif element_type == 3 or element_type == 4:

        #print(a)

        #ncoord = np.size(coord, axis = 0)
        #nnode = np.size(coord, axis = 0)

        ex,ey,ez = cfc.coordxtr(edof,coord,dof)

        print(val)

        ed = cfc.extractEldisp(edof,a)
        if element_type == 3:
            if val != 'nodal_values_by_el':
                coord2, topo, node_dofs, a_node, node_scalars = vdu.convert_to_node_topo(edof,ex,ey,ez,ed,ignore_first=False,dofs_per_node=1)
            else:
                coord2, topo, node_dofs, a_node, node_scalars = vdu.convert_to_node_topo(edof,ex,ey,ez,ed,values,ignore_first=False,dofs_per_node=1)
        elif element_type == 4:
            if val != 'nodal_values_by_el':
                coord2, topo, node_dofs, a_node, node_scalars = vdu.convert_to_node_topo(edof,ex,ey,ez,ed,ignore_first=False)
            else:
                coord2, topo, node_dofs, a_node, node_scalars = vdu.convert_to_node_topo(edof,ex,ey,ez,ed,values,ignore_first=False)
        #a_node = vdu.convert_a(coord,coord2,a,3)

        #def_coord = np.zeros([nnode,3])

        def_coord = coord2 + a_node*def_scale

        #print(a_node)
        
        #for i in range(np.size(def_coord, axis = 0)):
            #def_coord[i] = coord2[i] + a_node[i]
        #    def_coord[i,0] = coord2[i,0] + a[i*3]
        #    def_coord[i,1] = coord2[i,1] + a[i*3+1]
        #    def_coord[i,2] = coord2[i,2] + a[i*3+2]

        


        #print(def_coord)
        
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

        #print(a)
        #print(np.size(a, axis = 0))
        #print(np.size(a, axis = 1))
        """
        for i in range(0, ncoord):
            #if a.any() == None:
            #    x = coord[i,0]
            #    y = coord[i,1]
            #    z = coord[i,2]
            #else:
            #a_dx, a_dy, a_dz = get_a_from_coord(i,3,a,def_scale)

            #x = coord[i,0]+a_dx
            #y = coord[i,1]+a_dy
            #z = coord[i,2]+a_dz

            x = coord[i][0]+a[i][0]*scale
            y = coord[i][1]+a[i][1]*scale
            z = coord[i][2]+a[i][2]*scale

            def_coord[i] = [x,y,z]

            #def_nodes.append(v.Sphere(c='white').scale(1.5*scale).pos([x,y,z]).alpha(alpha))
        """
        #meshes = []
        #nel = np.size(edof, axis = 0)
        
        

        #print(topo)

        #mesh = v.Mesh([def_coord, topo]).lw(1)


        
        #mesh = v.Mesh([def_coord, topo]).lw(1).alpha(alpha)


        
        ct = vtk.VTK_HEXAHEDRON

        celltypes = [ct] * nel

        #ug=v.UGrid([def_coord, topo, celltypes])
        ug=v.UGrid([def_coord, topo, celltypes])
        ug.points(def_coord)
        
        mesh = ug.tomesh().lw(1).alpha(alpha)
        

        #v.settings.useDepthPeeling = True

        #print(val)

        #print('Cell connectivity: ',mesh.faces())

        #elif val and val == 'nodal_values':
        if val and val == 'el_values':
            #print(val)
            #vmin, vmax = np.min(values), np.max(values)
            
            el_values = vdu.convert_el_values(edof,values)
            mesh.celldata["val"] = el_values

            mesh.cmap(colormap, "val", on="cells")
        
        elif val and val == 'nodal_values_by_el':
            #print(val)
            #vmin, vmax = np.min(values), np.max(values)
            #nodal_values = vdu.convert_nodal_values(edof,topo,dof,values)
            #nodal_values = vdu.convert_a(coord2,coord,nodal_values,1)
            #mesh.pointdata["val"] = nodal_values
            mesh.pointdata["val"] = node_scalars
            print(ug.celldata.keys())
            #nodal_values = vdu.convert_nodal_values(edof,dof,coord,coord2,values)
            mesh.cmap(colormap, 'val', on="points")


        elif val and val == 'nodal_values':
            print(val)
            vmin, vmax = np.min(values), np.max(values)
            mesh.pointdata["val"] = values
            mesh.cmap(colormap, 'val', on="points")
            #ug.pointdata["val"] = values
            #nodal_values = vdu.convert_nodal_values(edof,dof,coord,coord2,values)
            #mesh.cmap(colormap, values, on="points", vmin=vmin, vmax=vmax)



        
        

        print('Number of topo cells: ',np.size(topo, axis=0))
        print('Number of mesh cells: ',np.size(mesh.faces(), axis=0))

        print('Number of topo points: ',np.size(coord2, axis=0))
        print('Number of mesh points: ',np.size(mesh.points(), axis=0))







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
            nodes = vdu.get_node_elements(coord,scale,alpha)
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
        


        return mesh

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
            a_dx, a_dy, a_dz = vdu.get_a_from_coord(i,6,a,def_scale)
            print('def scale',def_scale,'dx',a_dx,'dy',a_dy,'dz',a_dz,)

            x = coord[i,0]+a_dx
            y = coord[i,1]+a_dy
            z = coord[i,2]+a_dz

            def_coord[i] = [x,y,z]

            def_nodes.append(v.Sphere(c='white').scale(1.5*scale).pos([x,y,z]).alpha(alpha))

        nel = np.size(edof, axis = 0)
        def_elements = []
        res = 4

        vmin, vmax = np.min(values), np.max(values)

        el_values_array = np.zeros((1,4*res))[0,:]

        for i in range(nel):
            coord1,coord2 = vdu.get_coord_from_edof(edof[i,:],dof,5)

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
                    beam.name = f"Beam nr. {i+1}, seg. {j+1}"
                    def_elements.append(beam)

                    if values is not None:
                        el_value1 = values[nseg*i+j]
                        #beam.name = f"Beam nr. {i}, seg. {j}, element value: {np.round(values[nseg*i],2)}"
                        #beam.celldata["val"] = [values[nseg*i],values[nseg*i],values[nseg*i],values[nseg*i],values[nseg*i],values[nseg*i]]
                        el_value2 = values[nseg*i+j+1]

                        print('beam',i+1,'segment',j+1,f'val {el_value1} and {el_value2}')
                        
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
                        #beam.cmap(colormap, 'val', on="cells", vmin=vmin, vmax=vmax)

            else:
                beam = v.Cylinder([[def_coord[coord1,0],def_coord[coord1,1],def_coord[coord1,2]],[def_coord[coord2,0],def_coord[coord2,1],def_coord[coord2,2]]],r=scale,res=res,c=color).alpha(alpha)
                beam.info = f"Beam nr. {i}"
                def_elements.append(beam)

                if values is not None:
                    el_value1 = values[2*i]
                    el_value2 = values[2*i+1]

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
        #print("Displaced mesh for plate elements is not supported")
        #sys.exit()

        ex,ey = cfc.coordxtr(edof,coord,dof)
        #print(ex)
        #print(ey)

        ez = np.zeros((nel,4))
        #ez_init = np.array([
        #        [-t, t, -t, t]
        #    ])

        #for i in range(ez.shape[0]):
        #    ez[i] = ez_init

        #print(ez)

        ed = cfc.extractEldisp(edof,a)
        coord2, topo, node_dofs, a_node, test = vdu.convert_to_node_topo(edof,ex,ey,ez,ed,ignore_first=False)
        #a_node = vdu.convert_a(coord,coord2,a,3)

        #def_coord = np.zeros([nnode,3])

        #print('a_node',a_node)

        def_coord = coord2
        def_coord[:,2] = a_node[:,0]*def_scale

        mesh = v.Mesh([def_coord, topo]).lw(1).alpha(alpha)


        #ct = vtk.VTK_HEXAHEDRON

        #celltypes = [ct] * nel

        #ug=v.UGrid([def_coord, topo, celltypes])
        #ug.points(def_coord)
        
        #mesh = ug.tomesh().lw(1).alpha(alpha)

        if val and val == 'el_values':
            #print(val)
            #vmin, vmax = np.min(values), np.max(values)
            
            #el_values = vdu.convert_el_values(edof,values)
            mesh.celldata["val"] = values

            mesh.cmap(colormap, "val", on="cells")

        if render_nodes == True:
            nodes = vdu.get_node_elements(coord,scale,alpha)
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
    values=None,
    def_scale=1,
    alpha=1,
    export=False,
    colormap='jet',
    file='anim/'
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
            coords = vdu.get_coord_from_edof(edof[i,:],dof,4)


            mesh = v.Mesh([def_coord[coords,:],[[0,1,2,3],[4,5,6,7],[0,3,7,4],[1,2,6,5],[0,1,5,4],[2,3,7,6]]],alpha=alpha).lw(1)
            #mesh.info = f"Mesh nr. {i}"
            #mesh.name = f"Mesh nr. {i+1}"
            meshes.append(mesh)

        mesh = v.merge(meshes)

        #plt = v.show(mesh, axes=4, interactive=0)

        plt = v.Plotter(axes=4, interactive=False)

        plt.show(mesh)
        #plt += mesh


        if values is not None:
            vmin, vmax = np.min(values), np.max(values)


        for t in timesteps:

            
            '''
            for i in range(0, ncoord):
                a_dx, a_dy, a_dz = vdu.get_a_from_coord(i,3,a,def_scale)

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
            '''


            ex,ey,ez = cfc.coordxtr(edof,coord,dof)
            ed = cfc.extractEldisp(edof,a)
            coord2, topo, node_dofs, a_node, node_scalars = vdu.convert_to_node_topo(edof,ex,ey,ez,ed,ignore_first=False,dofs_per_node=3)
            
            new_coord = coord2 + a_node*def_scale

            ct = vtk.VTK_HEXAHEDRON

            celltypes = [ct] * nel

            ug=v.UGrid([new_coord, topo, celltypes])
            ug.points(new_coord)
            
            mesh = ug.tomesh().lw(1).alpha(alpha)

            # Element values
            el_values = vdu.convert_el_values(edof,values)
            mesh.celldata["val"] = el_values

            mesh.cmap(colormap, "val", on="cells", vmin=vmin*t, vmax=vmax*t)
            #meshes = []
            
            '''
            #vmin, vmax = np.min(el_values), np.max(el_values)
            for i in range(nel):
                coords = vdu.get_coord_from_edof(edof[i,:],dof,4)

                mesh = v.Mesh([def_coord[coords,:],[[0,1,2,3],[4,5,6,7],[0,3,7,4],[1,2,6,5],[0,1,5,4],[2,3,7,6]]],alpha=alpha).lw(1)
                meshes.append(mesh)

                if values is not None and np.size(values, axis = 1) == 1:
                    el_values_array = np.zeros((1,6))[0,:]
                    el_values_array[0] = values[i]*t
                    el_values_array[1] = values[i]*t
                    el_values_array[2] = values[i]*t
                    el_values_array[3] = values[i]*t
                    el_values_array[4] = values[i]*t
                    el_values_array[5] = values[i]*t
                    #if title is not None:
                    #    mesh.cmap(colormap, el_values_array, on="cells", vmin=vmin, vmax=vmax).addScalarBar(title=title,horizontal=True,useAlpha=False,titleFontSize=16)
                    #else:
                    mesh.cmap(colormap, el_values_array, on="cells", vmin=vmin, vmax=vmax)
                #for j in range(steps):
                    #mesh = v.Mesh([def_coord[coords,:,j],[[0,1,2,3],[4,5,6,7],[0,3,7,4],[1,2,6,5],[0,1,5,4],[2,3,7,6]]],alpha=alpha)
                    #meshes[i,j] = mesh
                #mesh = v.Mesh([def_coord[coords,:],[[0,1,2,3],[4,5,6,7],[0,3,7,4],[1,2,6,5],[0,1,5,4],[2,3,7,6]]],alpha=alpha)
                #mesh.info = f"Mesh nr. {i}"
                #meshes.append(mesh)
            mesh = v.merge(meshes)
            '''
            plt.clear()
            plt += mesh
            if export == True:
                output = file+f'_{int(10*t)}'
                export_vtk(output,mesh)
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
# Functions for handling rendering


#def figure(fig=None):
def figure(fig):
    app = init_app()
    plot_window = VedoPlotWindow.instance().plot_window

    #if fig == None:
    #    fig = plot_window.fig + 1

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


# Lägg till figurnummer här???
# Start Calfem-vedo visualization
def show_and_wait():
    app = init_app()
    plot_window = VedoPlotWindow.instance().plot_window
    plot_window.render()
