"""In this file we study the effects of varying the strength of a static magnetic
field onto the eigenenergies and eigenfunctions of the bulk luttinger kohn hamiltonian.
"""
# my imports
import sys
import os
current_path = os.path.dirname(os.path.realpath(__file__)) + "/Modules"
sys.path.append(current_path)
from HamiltonianBulk import preparing_terms, h_tot

# plotting imports
from PlottingUtils import IsoSurface, LinePlot

# other imports
from scipy.sparse.linalg import eigsh
import numpy as np
from tqdm import trange
#%% creating meshgrid for x,y,z
dimx = 50 # discretization nr in x
dimy = 50 # discretization nr in y
dimz = 50 # discretization nr in z

boundx_low = -1 # lower bound in x
boundx_upp = 1 # upper bound in x

boundy_low = -1 # lower bound in y
boundy_upp = 1 # upper bound in y

boundz_low = -1 # lower bound in z
boundz_upp = 1 # upper bound in z

coeff_x = 0.1 # coefficient determining half length of well in x direction
coeff_y = 1 # coefficient determining half length of well in y direction
coeff_z = 0.1 # coefficient determining half length of well in z direction

bmin = 0 # iniitial value of magnetic field strength
bmax = 2.5 # final value of magnetic field strength
points = 10 # number of integration points
bvals = np.linspace(bmin, bmax, points) # magnetic field values

k = 3 # number of eigenstates we'll solve for

eigenvalues = np.zeros((points, k)) # will store obtained eigenvalues
eigenvectors = np.zeros((points, k, dimx, dimy, dimz, 4), complex) # will store obtained eigenvetors

#%% varying magnetic the field
for i in trange(points): # varying magnetic field strength
    bz = bvals[i] # current magnetic field strength
    A = np.array([[0,0,0,0],[bz,0,0,0],[0,0,0,0]]) # vector potential

    needed_arrays = preparing_terms(boundx_low, boundx_upp, boundy_low, boundy_upp, boundz_low, boundz_upp, 
                                    dimx, dimy, dimz, 
                                    A = A,
                                    coeff_x = coeff_x, coeff_y = coeff_y, coeff_z = coeff_z, bc = 0)
    H, u = h_tot(needed_arrays, 
                 g1 = 13.35, g2 = 4.25, g3 = 5.69,
                 kappa = 1, B = [0,0,bz], infinity = 1e10,
                 conf = "wire") # obtaining hamiltonian
    
    eigvals, eigvects = eigsh(H, k = 3, which = "SM") # for CPU
    eigvects = eigvects / np.linalg.norm(eigvects, axis = 0)[None, :] # normalizing eigenvectors
    
    eigenvalues[i] = eigvals
    eigenvectors[i] = eigvects.T.reshape((k, dimx, dimy, dimz, 4))
#%% Plotting
LinePlot(bvals, eigenvalues.T, multiple_lines=True, label = ["e1", "e2", "e3"])

p_dist = np.sum(np.abs(eigenvectors[0,1])**2, axis = 3)

IsoSurface(p_dist, needed_arrays["grids"]["X"], needed_arrays["grids"]["Y"], needed_arrays["grids"]["Z"],
           iso_min = 1e-5, iso_max = None,
           iso_surf_n = 10,
           color_scale = 'RdBu_r', opacity = 0.6,
           x_show_caps = False, y_show_caps = False)