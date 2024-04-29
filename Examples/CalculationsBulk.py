"""In this file we study the eigenfunction profile of the bulk luttinger kohn hamiltonian.
"""
# my imports
import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__)) # current directory
prev_dir = os.path.abspath(os.path.join(dir_path, os.pardir)) # parent of current directory
sys.path.append(prev_dir+'/Modules') # appending modules folder
from HamiltonianBulk import preparing_terms, h_tot

# plotting imports
from PlottingUtils import IsoSurface

# other imports
from scipy.sparse.linalg import eigsh
import numpy as np
#%% creating meshgrid for x,y,z
dimx = 100 # discretization nr in x
dimy = 100 # discretization nr in y
dimz = 100 # discretization nr in z

boundx_low = -1 # lower bound in x
boundx_upp = 1 # upper bound in x

boundy_low = -1 # lower bound in y
boundy_upp = 1 # upper bound in y

boundz_low = -1 # lower bound in z
boundz_upp = 1 # upper bound in z

coeff_x = 0.1 # coefficient determining half length of well in x direction
coeff_y = 1 # coefficient determining half length of well in y direction
coeff_z = 0.1 # coefficient determining half length of well in z direction

A = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0]]) # vector potential

needed_arrays = preparing_terms(boundx_low, boundx_upp, boundy_low, boundy_upp, boundz_low, boundz_upp, 
                                dimx, dimy, dimz, 
                                A = A,
                                coeff_x = coeff_x, coeff_y = coeff_y, coeff_z = coeff_z, bc = 0)

#%% Obtainin full hamiltonian
H, u = h_tot(needed_arrays, 
             g1 = 13.35, g2 = 4.25, g3 = 5.69,
             kappa = 1, B = [0,0,0], infinity = 1e10,
             conf = "wire")
#%% Obtaining Eigvals and Eigvects
eigvals, eigvects = eigsh(H, k = 3, which = "SM") # for CPU
eigvects = eigvects / np.linalg.norm(eigvects, axis = 0)[None, :] # normalizing eigenvectors
tmpzip = zip(eigvects.T, eigvals) # zipping eigenvectors with eigenvalues
sort = sorted(tmpzip, key=lambda x: x[1]) # sorting vectors according to their eigenvalue in increasing order
eigvects = np.array([sort[i][0] for i in range(len(sort))]) # extracting sorted eigenvectors
eigvals = np.array([sort[i][1] for i in range(len(sort))]) # extracting sorted eigenvalues
#%% plotting eigenfunction isosurfaces
def get_v(n):
    return eigvects[n].reshape((dimx, dimy, dimz, 4))


p_dist = np.sum(np.abs(get_v(0))**2, axis = 3)

IsoSurface(p_dist, needed_arrays["grids"]["X"], needed_arrays["grids"]["Y"], needed_arrays["grids"]["Z"],
           iso_min = 1e-5, iso_max = None,
           iso_surf_n = 10,
           color_scale = 'RdBu_r', opacity = 0.6,
           x_show_caps = False, y_show_caps = False)


