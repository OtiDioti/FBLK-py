"""In this file we explore the luttinger kohn hamiltonian projected onto orbital guess-states.
for the bulk as well as other confinement geometries.
Note that we here consider hbar = me = e = 1.
"""
# my imports
import sys
import os
current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_path)
prev_dir = os.path.abspath(os.path.join(current_path, os.pardir)) # parent of current directory
sys.path.append(prev_dir+'/Modules') # appending modules folder
from UsefulFunctions import J0, Jx, Jy, Jz, anti_comm, possible_states
from GetInputProjection_Vectorized import get_input
from HamiltonianBulkProjection import psi_tot
from PlottingUtils import IsoSurface, plotly_animate_3d

# other imports
from scipy.sparse import kron, eye
from scipy.sparse.linalg import eigsh
from numpy import pi, array, ones, zeros, diff, mgrid, transpose, linspace
from numpy import sum as Sum
from numpy.linalg import norm
#%%
def k_ik_j(inputt,
           soft = 1e-15):
    """Returns list of expectation value <bra| k_ik_j = hbar^2 p_ip_j |ket> for infinite well (xc=0 and n = 1,2,3,...)
    eigenstates: each element of list is expectation value for the different dimensions x,y,z.
    input is generated via get_input() fn.
    soft is a softening factor to avoid singularities from blowing up (purely numerical object).
    """
    
    Lx, Ly, Lz = inputt["well widths"] # well widths
    Ac = inputt["vector potential coefficients"] 
    A = inputt["vector potential"] # extracting vector potential
    kx, ky, kz = inputt["linear momentum op"] # <n_x|kx|m_x> for all permutations of all basis states (adding time axis at the front to match shape of Aij)
    xkx, kxx, yky, kyy, zkz, kzz = inputt["i ki and ki i op"] # expecation values <n_i|i ki|m_i> and <n_i|ki i|m_i> for all permutations of all basis states
    dx, dy, dz, dxdy, dxdz, dydz, dxdydz = inputt["dirac deltas"] # dirac delta functions for all permutations of all basis states
    x2, y2, z2 = inputt["squared position op"] # expect. val. of all squared position operators for all permutations of all basis states
    
    Axx, Axy, Axz, Axc = [A[:,0,0], A[:,0,1], A[:,0,2], A[:,0,3]]
    Ayx, Ayy, Ayz, Ayc = [A[:,1,0], A[:,1,1], A[:,1,2], A[:,1,3]]
    Azx, Azy, Azz, Azc = [A[:,2,0], A[:,2,1], A[:,2,2], A[:,2,3]]
    
    # cross momenta after min. coup. substitution
    kxky = sum([kx * ky * dz, kxx * Ac[:,1,0] * dydz, kx * Ayy * dz, kx * Ayz * dy, kx * Ayc * dydz,
                Axx * ky * dz, Ac[:,0,0]*Ac[:,1,0] * x2 * dydz, Axx*Ayy * dz, Axx*Ayz * dy, Axx*Ayc * dydz,
                Ac[:,0,1] * yky * dxdz, Axy*Ayx * dz, Ac[:,0,1]*Ac[:,1,1] * y2 * dxdz, Axy*Ayz * dx, Axy*Ayc * dxdz,
                Axz * ky * dx, Axz*Ayx * dy, Axz*Ayy * dx, Ac[:,0,2]*Ac[:,1,2] * z2 * dxdy, Axz*Ayc * dxdy,
                Axc * ky * dxdz, Axc*Ayx * dydz, Axc*Ayy * dxdz, Axc*Ayz * dxdy, Axc*Ayc * dxdydz]) # <bra|(kx + Ax)(ky + Ay)|bra>
               
    kykx = sum([ky * kx * dz, ky * Axx * dz, kyy * Ac[:,0,1] * dxdz, ky * Axz * dx, ky * Axc * dxdz,
                Ac[:,1,0] * xkx * dydz, Ac[:,1,0]*Ac[:,0,0] * x2 * dydz, Ayx*Axy * dz, Ayx*Axz * dy, Ayx*Axc * dydz,
                Ayy * kx * dz, Ayy*Axx * dz, Ac[:,1,1]*Ac[:,0,1] * y2 * dxdz, Ayy*Axz * dx, Ayy*Axc * dxdz,
                Ayz * kx * dy, Ayz*Axx * dy, Ayz*Axy * dx, Ac[:,1,2]*Ac[:,0,2] * z2 * dxdy, Ayz*Axc * dxdy,
                Ayc * kx * dydz, Ayc*Axx * dydz, Ayc*Axy * dxdz, Ayc*Axz * dxdy, Ayc*Axc * dxdydz]) # <bra|(ky + Ay)(kx + Ax)|bra>         

    kykz = sum([ky * kz * dx, ky * Azx * dz, kyy * Ac[:,2,1] * dxdz, ky * Azz * dx, ky * Azc * dxdz,
                Ayx * kz * dy, Ac[:,1,0]*Ac[:,2,0] * x2 * dydz, Ayx*Azy * dz, Ayx*Azz * dy, Ayx*Azc * dydz,
                Ayy * kz * dx, Ayy*Azx * dz, Ac[:,1,1]*Ac[:,2,1] * y2 * dxdz, Ayy*Azz * dx, Ayy*Azc * dxdz,
                Ac[:,1,2] * zkz * dxdy, Ayz*Azx * dy, Ayz*Azy * dx, Ac[:,1,2]*Ac[:,2,2] * z2 * dxdy, Ayz*Azc * dxdy,
                Ayc * kz * dxdy, Ayc*Azx * dydz, Ayc*Azy * dxdz, Ayc*Azz * dxdy, Ayc*Azc * dxdydz]) # <bra|(ky + Ay)(kz + Az)|bra>       
                
    kzky = sum([kz * ky * dx, kz * Ayx * dy, kz * Ayy * dx, kzz * Ac[:,1,2] * dxdy, kz * Ayc * dxdy,
                Azx * ky * dz, Ac[:,2,0]*Ac[:,1,0] * x2 * dydz, Azx*Ayy * dz, Azx*Ayz * dy, Azx*Ayc * dydz,
                Ac[:,2,1] * yky * dxdz, Azy*Ayx * dz, Ac[:,2,1]*Ac[:,1,1] * y2 * dxdz, Azy*Ayz * dx, Azy*Ayc * dxdz,
                Azz * ky * dx, Azz*Ayx * dy, Azz*Ayy * dx, Ac[:,2,2]*Ac[:,1,2] * z2 * dxdy, Azz*Ayc * dxdy,
                Azc * ky * dxdz, Azc*Ayx * dydz, Azc*Ayy * dxdz, Azc*Ayz * dxdy, Azc*Ayc * dxdydz]) # <bra|(kz + Az)(ky + Ay)|bra>
    
    kzkx = sum([kz * kx * dy, kz * Axx * dy, kz * Axy * dx, kzz * Ac[:,0,2] * dxdy, kz * Axc * dxdy,
                Ac[:,2,0] * xkx * dydz, Ac[:,2,0]*Ac[:,0,0] * x2 * dydz, Azx*Axy * dz, Azx*Axz * dy, Azx*Axc * dydz,
                Azy * kx * dz, Azy*Axx * dz, Ac[:,2,1]*Ac[:,0,1] * y2 * dxdz, Azy*Axz * dx, Azy*Axc * dxdz,
                Azz * kx * dy, Azz*Axx * dy, Azz*Axy * dx, Ac[:,2,2]*Ac[:,0,2] * z2 * dxdy, Azz*Axc * dxdy,
                Azc * kx * dydz, Azc*Axx * dydz, Azc*Axy * dxdz, Azc*Axz * dxdy, Azc*Axc * dxdydz]) # <bra|(kz + Az)(kx + Ax)|bra>
    
    kxkz = sum([kx * kz * dy, kxx * Ac[:,2,0] * dydz, kx * Azy * dz, kx * Azz * dy, kx * Azc * dydz,
                Axx * kz * dy, Ac[:,0,0]*Ac[:,2,0] * x2 * dydz, Axx*Azy * dz, Axx*Azz * dy, Axx*Azc * dydz,
                Axy * kz * dx, Axy*Azx * dz, Ac[:,0,1]*Ac[:,2,1] * y2 * dxdz, Axy*Azz * dx, Axy*Azc * dxdz,
                Ac[:,0,2] * zkz * dxdy, Axz*Azx * dy, Axz*Azy * dx, Ac[:,0,2]*Ac[:,2,2] * z2 * dxdy, Axz*Azc * dxdy,
                Axc * kz * dxdy, Axc*Azx * dydz, Axc*Azy * dxdz, Axc*Azz * dxdy, Axc*Azc * dxdydz]) # <bra|(kx + Ax)(kz + Az)|bra>  
               
    return kxky, kykx, kykz, kzky, kzkx, kxkz

def k2_i(inputt,
         soft = 1e-15):
    """Returns list of expectation value <bra| k^2 = hbar^2 p^2 |ket> for infinite well (xc=0 and n = 1,2,3,...)
    eigenstates: each element of list is expectation value for the different dimensions x,y,z.
    input is generated via get_input() fn.
    soft is a softening factor to avoid singularities from blowing up (purely numerical object).
    """

   
    k2 = lambda n_bra, n_ket, L: n_ket**2 * pi**2 / (L**2) * (n_bra==n_ket)
    
    nx_bra, ny_bra, nz_bra = inputt["n_bra"]
    nx_ket, ny_ket, nz_ket = inputt["n_ket"]
    
    Lx, Ly, Lz = inputt["well widths"] # well widths
    Ac = inputt["vector potential coefficients"] 
    A = inputt["vector potential"] # extracting vector potential
    kx, ky, kz = inputt["linear momentum op"] # <n_x|kx|m_x> for all permutations of all basis states (adding time axis at the front to match shape of Aij)
    xkx, kxx, yky, kyy, zkz, kzz = inputt["i ki and ki i op"] # expecation values <n_i|i ki|m_i> and <n_i|ki i|m_i> for all permutations of all basis states
    dx, dy, dz, dxdy, dxdz, dydz, dxdydz = inputt["dirac deltas"] # dirac delta functions for all permutations of all basis states
    x2, y2, z2 = inputt["squared position op"] # expect. val. of all squared position operators for all permutations of all basis states
    
    Axx, Axy, Axz, Axc = [A[:,0,0], A[:,0,1], A[:,0,2], A[:,0,3]]
    Ayx, Ayy, Ayz, Ayc = [A[:,1,0], A[:,1,1], A[:,1,2], A[:,1,3]]
    Azx, Azy, Azz, Azc = [A[:,2,0], A[:,2,1], A[:,2,2], A[:,2,3]]
    
    # calculating square momenta
    kx2 = sum([k2(nx_bra, nx_ket, Lx) * dydz, kxx * Ac[:,0,0] * dydz, kx * Axy * dz, kx * Axz * dy, kx * Axc * dydz,
               Ac[:,0,0] * xkx * dydz, Ac[:,0,0]*Ac[:,0,0] * x2 * dydz, Axx*Axy * dz, Axx*Axz * dy, Axx*Axc * dydz,
               Axy * kx * dz, Axy*Axx * dz, Ac[:,0,1]*Ac[:,0,1] * y2 * dxdz, Axy*Axz * dx, Axy*Axc * dxdz,
               Axz * kx * dy, Axz*Axx * dy, Axz*Axy * dx, Ac[:,0,2]*Ac[:,0,2] * z2 * dxdy, Axz*Axc * dxdy,
               Axc * kx * dydz, Axc*Axx * dydz, Axc*Axy * dxdz, Axc*Axz * dxdy, Axc*Axc * dxdydz]) # <bra|(kx + Ax)^2|ket>
    
    ky2 = sum([k2(ny_bra, ny_ket, Ly) * dxdz, ky * Ayx * dz, kyy * Ac[:,1,1] * dxdz, ky * Ayz * dx, ky * Axc * dxdz,
               Ayx * ky * dz, Ac[:,1,0]*Ac[:,1,0] * x2 * dxdz, Ayx*Ayy * dz, Ayx*Ayz * dy, Ayx*Ayc * dydz,
               Ac[:,1,1] * yky * dxdz, Ayy*Ayx * dz, Ac[:,1,1]*Ac[:,1,1] * y2 * dxdz, Ayy*Ayz * dx, Ayy*Ayc * dxdz,
               Ayz * ky * dx, Ayz*Ayx * dy, Ayz*Ayy * dx, Ac[:,1,2]*Ac[:,1,2] * z2 * dxdy, Ayz*Ayc * dxdy,
               Ayc * ky * dxdz, Ayc*Ayx * dydz, Ayc*Ayy * dxdz, Ayc*Ayz * dxdy, Ayc*Ayc * dxdydz]) # <bra|(ky + Ay)^2|ket>
    
    kz2 = sum([k2(nz_bra, nz_ket, Lz) * dxdy, kz * Azx * dy, kz * Azy * dx, kzz * Ac[:,2,2] * dxdy, kz * Axc * dxdy,
               Azx * kz * dy, Ac[:,2,0]*Ac[:,2,0] * x2 * dydz, Azx*Azy * dz, Azx*Azz * dy, Azx*Azc * dydz,
               Azy * kz * dx, Azy*Azx * dz, Ac[:,2,1]*Ac[:,2,1] * y2 * dxdz, Azy*Azz * dx, Azy*Azc * dxdz,
               Ac[:,2,2] * zkz * dxdy, Azz*Azx * dy, Azz*Azy * dx, Ac[:,2,2]*Ac[:,2,2] * z2 * dxdy, Azz*Azc * dxdy,
               Azc * kz * dxdy, Azc*Azx * dydz, Azc*Azy * dxdz, Azc*Azz * dxdy, Azc*Azc * dxdydz]) # <bra|(kz + Az)^2|ket>
    
    return kx2, ky2, kz2

def h_lk_v(kx2, ky2, kz2,
           kxky, kykx, kykz, kzky, kzkx, kxkz,
           g1 = 13.35, g2 = 4.25, g3 = 5.69):
    """Returns discritezied LK Hamiltonian, in sparse format.
    k_i is expectation value of k_i operator.
    ki_2 is expectation value of operator k_i^2.
    k_ik_j is expectation value of operator k_ik_j.
    momenta_arrays contains all discretized momentum operators needed.
    g1, g2, g3 are Luttinger poaramteters (Default: Luttinger params for Ge).
    """

    # magnetic field free HLK
    first_term = -0.5 * (g1 + 2.5 * g2) * kron(kx2 + ky2 + kz2, J0) # first term of LK Hamiltonian
    secon_term = g2 * (kron(kx2, Jx.__pow__(2)) + kron(ky2, Jy.__pow__(2)) + kron(kz2, Jz.__pow__(2))) # second term of LK Hamiltonian
    third_term = 0.5 * g3 * (kron(kxky + kykx, anti_comm(Jx, Jy)) + kron(kykz + kzky, anti_comm(Jy, Jz)) + kron(kzkx + kxkz, anti_comm(Jz, Jx))) # third term of LK Hamiltonian
    
    h_lk = first_term + secon_term + third_term # magnetic field free HLK
    
    return h_lk

def h_z_v(dim, kappa = 1, B = [0,0,0]):
    """Returns zeeman hamiltonian.
    k_i is expectation value of k_i operator.
    ki_2 is expectation value of operator k_i^2.
    k_ik_j is expectation value of operator k_ik_j.
    kappa is the magnetic g-factor.
    B is the magnetic field vector.
    """ 
    return kappa * kron(eye(dim), Jx * B[0] + Jy * B[1] + Jz * B[2])

def h_tot_v(kx2, ky2, kz2, 
            kxky, kykx, kykz, kzky, kzkx, kxkz,
            dim,
            g1 = 13.35, g2 = 4.25, g3 = 5.69,
            kappa = 1, B = [0,0,0]):
    """Returns total hamiltonian as well as plottable well potential.
    ki_2 is expectation value of operator k_i^2.
    k_ik_j is expectation value of operator k_ik_j.
    g1, g2, g3 are Luttinger poaramteters (Default: Luttinger params for Ge).
    B is magnetic field vector.
    kappa is magnetic g-factor.
    conf is the confinement type desired (can be "bulk", "planar", "wire").
    """
    
    lk = h_lk_v(kx2, ky2, kz2, kxky, kykx, kykz, kzky, kzkx, kxkz, g1, g2, g3) # lk hamiltonian
    z = h_z_v(dim, kappa, B)
    return - lk + z
#%% Constants
g1 = 13.35 # Ge Luttinger parameter gamma_1
g2 = 4.25 # Ge Luttinger parameter gamma_2
g3 = 5.69 # Ge Luttinger parameter gamma_3

gs = 4.84 # spherical approx
gk = g1 + 2.5 * gs # spherical approx

points = 10
zero = zeros(points)
one = ones(points)
bz = linspace(0, 1, points)
B = array([zero, zero, bz]).T # magnetic field vector

Ac = array([[zero,zero,zero,zero],
            [0.5 * bz,zero,zero,zero],
            [zero,zero,zero,zero]]) # magnetic vector potential coefficients
Ac = transpose(Ac, axes = (2,0,1))[:,:,:,None,None]


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

dimx = 50 # discretization nr in x
dimy = 50 # discretization nr in y
dimz = 50 # discretization nr in z

X, Y, Z = mgrid[boundx_low : boundx_upp : dimx*1j, 
                boundy_low : boundy_upp : dimy*1j,
                boundz_low : boundz_upp : dimz*1j] # meshgrid

dx = diff(X[:,0,0])[0]
dy = diff(Y[0,:,0])[0]
dz = diff(Z[0,0,:])[0]

#%% Number of states
nx_max = 2 # highest state projected on x
ny_max = 2 # highest state projected on y
nz_max = 2 # highest state projected on z
dim = nx_max * ny_max * nz_max # dimensionality of the system
#%% obtaining indices
possible_statess = possible_states(nx_max, ny_max, nz_max) # permutation of all possible states
inputt = get_input(possible_statess, Ac, Lx, Ly, Lz, dim)
#%% Constructing Hamiltonian and diagonalizing (JIT)

def get_ks():
    """Returns tuple of 2 ndarrays for the expectation values of k^2 and kikj opertators.
    """
    kikj = k_ik_j(inputt) # obtaining k_ik_j expectation values 
    k2 = k2_i(inputt) # obtaining k_i^2 expectation values
    return kikj, k2

kikj, k2 = get_ks()

H = [] # empty list of hamiltonians
for i in range(points):
    H.append(h_tot_v(k2[0][i], k2[1][i], k2[2][i], 
                   kikj[0][i], kikj[1][i], kikj[2][i], kikj[3][i], kikj[4][i], kikj[5][i],
                   dim,
                   g1 = 13.35, g2 = 4.25, g3 = 5.69,
                   kappa = kappa, B = B[i]))
#%% Obtaining Eigvals and Eigvects
k = 10 # nr of solutions
eigenvalues = []
eigenvector = []
for t in range(points):
    eigvals, eigvects = eigsh(H[t], k = k, which = "SM") # for CPU
    eigvects = eigvects / norm(eigvects, axis = 0)[None, :] # normalizing eigenvectors
    tmpzip = zip(eigvects.T, eigvals) # zipping eigenvectors with eigenvalues
    sort = sorted(tmpzip, key=lambda x: x[1]) # sorting vectors according to their eigenvalue in increasing order
    eigvects = array([sort[i][0] for i in range(len(sort))]) # extracting sorted eigenvectors
    eigvals = array([sort[i][1] for i in range(len(sort))]) # extracting sorted eigenvalues
    eigenvalues.append(eigvals)
    eigenvector.append(eigvects)
#%% tracing out spin components to obtain plottable eigenvectors
spin_component = zeros((points, k, dim, 4), complex) # here we will store the eigenvectors spin components
traced_out_psi = zeros((points, k, dim), complex) # here we will store eigenvectors with spin component traced out

for t in range(points):
    for i in range(k): # iterating through eigenvectors
        for j in range(dim): # iterating through basis states
            spin_component[t, i, j, :] = eigenvector[t][i][j*4:j*4 + 4] # for each basis state we append the spin components (eg: the first 4 element of tmp correspond to |1,1,1,3/2>, |1,1,1,-3/2>, |1,1,1,1/2>, |1,1,1,-1/2>)
        coeff =  Sum(spin_component[t, i], axis = 1)
        traced_out_psi[t, i] = coeff / norm(coeff)
#%% storing animation frames

def eigfn(X, Y, Z,
          basis_states_coeff,
          Lx, Ly, Lz):
    """Returns 3d plottable eigenfunction: this being the weighted sum of the 
    basis states in possible_statess.
    X,Y,Z are space meshgrid.
    basis_states_coeff are complex coefficients for the basis states in possible_statess.
    L_i are the well-depths in the three dimensions.
    """
    eign_fn = 0 # initializing eigenfunction
    for n, state in enumerate(possible_statess):
        nx, ny, nz = state
        eign_fn += basis_states_coeff[n] * psi_tot(X, Y, Z, 
                                               nx, ny, nz, 
                                               Lx, Ly, Lz)
    eign_fn = eign_fn / norm(eign_fn)
    return eign_fn

p_dist = zeros((points, dimx, dimy, dimz))
for t in range(points):
    p_dist[t] = abs(eigfn(X, Y, Z,
                       traced_out_psi[t, 0],
                       Lx, Ly, Lz))**2
fig = plotly_animate_3d(p_dist, X, Y, Z)
fig.show()
#%% plotting static solution
        
n = 0
p_dist = abs(eigfn(X, Y, Z,
                   traced_out_psi[n],
                   Lx, Ly, Lz))**2

IsoSurface(p_dist, X, Y, Z,
            iso_min = None, iso_max = None,
            iso_surf_n = 10,
            color_scale = 'RdBu_r', opacity = 0.6)