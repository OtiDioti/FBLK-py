# my imports
import sys
import os
current_path = os.path.dirname(os.path.realpath(__file__)) + "/Modules"
sys.path.append(current_path)
from HamiltonianPlanar import preparing_terms, h_tot_spherical

# plotting imports
from PlottingUtils import HeatMap, LinePlot

# other imports
from scipy.sparse.linalg import eigsh
import numpy as np
from tqdm import trange
#%% constants
g1 = 13.35 # luttinger parameter for Ge
g2 = 4.25 # luttinger parameter for Ge
g3 = 5.69 # luttinger parameter for Ge

gs = 4.25 * np.sqrt(1 - 3/8 * (1 - (g3 / g2)**2 )) # spherical approximation term
gk = g1 + 5*gs/2 # spherical approximation term

Az = 0 # z-component of magnetic vector potential
bz = 0 # z-component of magnetic field
kappa = 1 # magnetic g-factor

Lz = 0.1 # width of well in z
le = 6 * Lz / (np.pi * g1**(1/3))  # characteristic length of e-field in triangular well (see paper Hole-spin qubits in Ge nanowire quantum dots: Interplay of orbital magnetic field, strain, and growth direction)
#%% creating meshgrid for x,y,z
dimx = 50 # discretization nr in x
dimy = 50 # discretization nr in y

boundx_low = -1 # lower bound in x
boundx_upp = 1 # upper bound in x

boundy_low = -1 # lower bound in y
boundy_upp = 1 # upper bound in y

needed_arrays = preparing_terms(boundx_low, boundx_upp, boundy_low, boundy_upp, dimx, dimy, Lz, le, bc = 0, g1 = g1, g2 = g2, g3 = g3, gs = gs) 
    
#%% Obtainin full hamiltonian
H, U = h_tot_spherical(needed_arrays, Az, bz, kappa, gs, gk, infinity = 1e10)
del needed_arrays # to save memory

#%% Obtaining Eigvals and Eigvects
eigvals, eigvects = eigsh(H, k = 3, which = "SM") # for CPU
eigvects = eigvects / np.linalg.norm(eigvects, axis = 0)[None, :] # normalizing eigenvectors

#%% plotting eigenfunctions
def get_v(n):
    return eigvects[:, n].reshape((dimx, dimy, 4))

p_dist = np.sum(np.abs(get_v(0))**2, axis = 2) # probability distribution

HeatMap(p_dist, boundx_low, boundx_upp, boundy_low, boundy_upp, 
        xlabel = r'$x$', ylabel = r'$y$', zlabel = r'$\left|\phi\right|^2$') # plotting prob_distribution

HeatMap(U, boundx_low, boundx_upp, boundy_low, boundy_upp, 
        xlabel = r'$x$', ylabel = r'$y$', zlabel = r'$V(x,y)$') # plotting potential



#%% plotting effect of varying magnetic field
"""
bmin = 0 # min magnetic field value
bmax = 10 # max magnetic field value
points = 10 # nr of integration points 
bvals = np.linspace(bmin, bmax, points) # values of magnetic field

needed_arrays = preparing_terms(boundx_low, boundx_upp, boundy_low, boundy_upp, dimx, dimy, Lz, le, bc = 0, g1 = g1, g2 = g2, g3 = g3, gs = gs) 
eigenvalues = np.zeros((points, 6))
for i in trange(points):
    H, U = h_tot_spherical(needed_arrays, Az, bvals[i], kappa, gs, gk, infinity = 1e10)
    eigvals, eigvects = eigsh(H, k = 6, which = "SM") # for CPU
    eigenvalues[i] = eigvals # appending eigenvalues

LinePlot(bvals, eigenvalues.T, multiple_lines = True, xlbl = r'$B_z$', ylbl = r'$E$', label = ["e1", "e2", "e3", "e4", "e5", "e6"])
"""
    
    

