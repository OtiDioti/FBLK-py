# my imports
import sys
import os
current_path = os.path.dirname(os.path.realpath(__file__)) + "/Modules"
sys.path.append(current_path)
from HamiltonianBulk import preparing_terms, h_tot

# plotting imports
from PlottingUtils import IsoSurface

# other imports
from scipy.sparse.linalg import eigsh
import numpy as np
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

coeff_x = 1
coeff_y = 1
coeff_z = 0.1

needed_arrays = preparing_terms(boundx_low, boundx_upp, boundy_low, boundy_upp, boundz_low, boundz_upp, 
                                dimx, dimy, dimz, 
                                A = np.array([[0,0,0],[0,0,0],[0,0,0]]), B = [0,0,0],
                                coeff_x = coeff_x, coeff_y = coeff_y, coeff_z = coeff_z, bc = 0)

#%% Obtainin full hamiltonian
H, u = h_tot(needed_arrays, 
             g1 = 13.35, g2 = 4.25, g3 = 5.69,
             kappa = 1, B = [0,0,0], infinity = 1e10,
             conf = "planar")
#%% Obtaining Eigvals and Eigvects
eigvals, eigvects = eigsh(H, k = 3, which = "SM") # for CPU
eigvects = eigvects / np.linalg.norm(eigvects, axis = 0)[None, :] # normalizing eigenvectors
#%% plotting eigenfunction isosurfaces
def get_v(n):
    return eigvects.T[n].reshape((dimx, dimy, dimz, 4))


p_dist = np.sum(np.abs(get_v(0))**2, axis = 3)

IsoSurface(p_dist, needed_arrays["grids"]["X"], needed_arrays["grids"]["Y"], needed_arrays["grids"]["Z"],
           iso_min = 1e-5, iso_max = None,
           iso_surf_n = 10,
           color_scale = 'RdBu_r', opacity = 0.6,
           x_show_caps = False, y_show_caps = False)


