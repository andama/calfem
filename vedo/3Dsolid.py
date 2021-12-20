#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D truss example using VTK to visualize

@author: Andreas Åmand
"""

import os

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import calfem.core as cfc
import numpy as np
import vis_vedo as cfvv
from PyQt5 import Qt
from scipy.io import loadmat

os.system('clear')

solid_data = loadmat('3Dsolid.mat')

edof = solid_data['edof']
coord = solid_data['coord']
dof = solid_data['dof']

#cfvv.solid3d.draw_geometry(edof,coord,dof,0.02,1)
cfvv.draw_geometry(edof,coord,dof,2)

cfvv.show_and_wait()