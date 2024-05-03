"""In this file we study the eigenfunction profile of the bulk luttinger kohn hamiltonian
using the projection method.
"""
# my imports
import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__)) # current directory
prev_dir = os.path.abspath(os.path.join(dir_path, os.pardir)) # parent of current directory
sys.path.append(prev_dir+'/Modules') # appending modules folder
from HamiltonianBulkProjection import h_tot, k_ik_j, k2_i, psi_tot, h_tot_v
from UsefulFunctions import possible_states, converter

# plotting imports
from PlottingUtils import IsoSurface
import matplotlib.pyplot as plt

# other imports
from numpy import array, mgrid, diff, zeros, ndarray
from numpy.linalg import norm
from scipy.sparse.linalg import eigsh
from scipy.sparse import block_array
from tqdm import trange
from numpy import sum as Sum
from numpy import abs as Abs
from numba import jit
import time
#%% Constants
g1 = 13.35 # Ge Luttinger parameter gamma_1
g2 = 4.25 # Ge Luttinger parameter gamma_2
g3 = 5.69 # Ge Luttinger parameter gamma_3

B = [0, 0, 0] # magnetic field vector
A = array([[0,0,0,0],
           [0,0,0,0],
           [0,0,0,0]]) # magnetic vector potential

kappa = 1 # magnetic g-factor

boundx_low = -1 # lower bound in x
boundx_upp = 1 # upper bound in x

boundy_low = -1 # lower bound in y
boundy_upp = 1 # upper bound in y

boundz_low = -1 # lower bound in z
boundz_upp = 1 # upper bound in z

Lx = 2 # well between boundx_low < x < boundx_upp
Ly = 2 # well between boundy_low < y < boundy_upp
Lz = 2 # well between boundz_low < z < boundz_upp

dimx = 100 # discretization nr in x
dimy = 50 # discretization nr in y
dimz = 100 # discretization nr in z

X, Y, Z = mgrid[boundx_low : boundx_upp : dimx*1j, 
                boundy_low : boundy_upp : dimy*1j,
                boundz_low : boundz_upp : dimz*1j] # meshgrid

dx = diff(X[:,0,0])[0]
dy = diff(Y[0,:,0])[0]
dz = diff(Z[0,0,:])[0]

#%% Number of states
nx_max = 5 # highest state projected on x
ny_max = 5 # highest state projected on y
nz_max = 5 # highest state projected on z
Nx = nx_max + 1 # needed for the loops
Ny = ny_max + 1 # needed for the loops
Nz = nz_max + 1 # needed for the loops
dim = nx_max * ny_max * nz_max # dimensionality of the system
#%% obtaining indices
possible_statess = possible_states(nx_max, ny_max, nz_max) # permutation of all possible states
#%% Constructing Hamiltonian and diagonalizing (JIT)

# @jit(nopython=True)
def get_ks():
    kikj = [] # list to store kikj arrays (must be lsit for numba to work)
    k2 = [] # list to store k2 arrays (must be lsit for numba to work)
    idx = 0 # dummy index
    for nxket, nyket, nzket in possible_statess:
        for nxbra, nybra, nzbra in possible_statess:
            bra = array([nxbra, nybra, nzbra]) # defining bra state
            ket = array([nxket, nyket, nzket]) # defining ket state
            kikj.append([*k_ik_j(bra, ket, A, Lx, Ly, Lz)]) # obtaining k_ik_j expectation values 
            k2.append([*k2_i(bra, ket, A, Lx, Ly, Lz)]) # obtaining k_i^2 expectation values
            idx += 1
    kikj = array(kikj).reshape(dim, dim, 6)
    k2 = array(k2).reshape(dim, dim, 3)
    return kikj, k2

kikj, k2 = get_ks()

start = time.time()
H_v = h_tot_v(k2, kikj, dim)
end = time.time()
print(end - start)

start = time.time()
H = [[[] for idx in range(dim)]for idx in range(dim)] # empty lists will be replaced by blocks of hamiltonian
for nxket, nyket, nzket in possible_statess:
    for nxbra, nybra, nzbra in possible_statess:
        bra = array([nxbra, nybra, nzbra]) # defining bra state
        ket = array([nxket, nyket, nzket]) # defining ket state
        i = converter(ket, possible_statess) # obtaining block col index
        j = converter(bra, possible_statess) # obtaining block row index
        tmp = h_tot(k2[i,j][0], k2[i,j][1], k2[i,j][2],
                    kikj[i,j][0], kikj[i,j][1], kikj[i,j][2], kikj[i,j][3], kikj[i,j][4], kikj[i,j][5], 
                    g1 = g1, g2 = g2, g3 = g3,
                    kappa = kappa, B = B) # calculating block hamiltonian <nx ny nz|H|mx my mz>
        
        H[i][j] = tmp # appending block


H = block_array(H, format = "csr") # contructing hamiltonian from blocks
end = time.time()
print(end - start)
#%% Obtaining Eigvals and Eigvects
k = 10 # nr of solutions
eigvals, eigvects = eigsh(H, k = k, which = "SM") # for CPU
eigvects = eigvects / norm(eigvects, axis = 0)[None, :] # normalizing eigenvectors
tmpzip = zip(eigvects.T, eigvals) # zipping eigenvectors with eigenvalues
sort = sorted(tmpzip, key=lambda x: x[1]) # sorting vectors according to their eigenvalue in increasing order
eigvects = array([sort[i][0] for i in range(len(sort))]) # extracting sorted eigenvectors
eigvals = array([sort[i][1] for i in range(len(sort))]) # extracting sorted eigenvalues
#%% tracing out spin components to obtain plottable eigenvectors
spin_component = zeros((k, dim, 4), complex) # here we will store the eigenvectors spin components
traced_out_psi = zeros((k, dim), complex) # here we will store eigenvectors with spin component traced out

for i in trange(k, desc="Tracing out spin-components"): # iterating through eigenvectors
    tmp = eigvects[i] # current eigenvector
    for j in range(dim): # iterating through basis states
        spin_component[i, j, :] = eigvects[i][j*4:j*4 + 4] # for each basis state we append the spin components (eg: the first 4 element of tmp correspond to |1,1,1,3/2>, |1,1,1,-3/2>, |1,1,1,1/2>, |1,1,1,-1/2>)
    coeff =  Sum(Abs(spin_component[i])**2, axis = 1)
    traced_out_psi[i] = coeff / norm(coeff)

#%% plotting eigenfunctions
def eigfn(X, Y, Z,
          expanded_state,
          Lx, Ly, Lz):
    eign_fn = 0 # initializing eigenfunction
    for n, state in enumerate(possible_statess):
        nx, ny, nz = state
        eign_fn += expanded_state[n] * psi_tot(X, Y, Z, 
                                               nx, ny, nz, 
                                               Lx, Ly, Lz)
    eign_fn = eign_fn / norm(eign_fn)
    return eign_fn
        
n = 0
p_dist = Abs(eigfn(X, Y, Z,
                   traced_out_psi[n],
                   Lx, Ly, Lz))**2

IsoSurface(p_dist, X, Y, Z,
            iso_min = None, iso_max = None,
            iso_surf_n = 10,
            color_scale = 'RdBu_r', opacity = 0.6)