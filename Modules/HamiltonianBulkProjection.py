"""In this file we explore the luttinger kohn hamiltonian projected onto orbital guess-states.
for the bulk as well as other confinement geometries.
Note that we here consider hbar = me = e = 1.
"""
# my imports
import sys
import os
current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_path)
from UsefulFunctions import J0, Jx, Jy, Jz, anti_comm

# other imports
from scipy.sparse import kron, eye
from numpy import pi, sin, sqrt, where, array
from numpy import sum as Sum
from numpy.linalg import norm
#%% defining momentum and position operators for infinite well orbital states
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
#%%
def eigfn(X, Y, Z,
          basis_states_coeff, possible_statess,
          Lx, Ly, Lz):
    """Returns 3d plottable eigenfunction: this being the weighted sum of the 
    basis states in possible_statess. (Used in Projection method)
    X,Y,Z are space meshgrid.
    basis_states_coeff are complex coefficients for the basis states in possible_statess.
    possible_statess is array of all possible permutations of basis states.
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

def get_ks(inputt):
    """Returns tuple of 2 ndarrays for the expectation values of k^2 and kikj opertators.
    """
    kikj = k_ik_j(inputt) # obtaining k_ik_j expectation values 
    k2 = k2_i(inputt) # obtaining k_i^2 expectation values
    return kikj, k2

#%% eigenfunctions for infinite well (xc = 0 and n = 1,2,3,...)

def psi(x, n, L):
    """Returns eigenfunction of particle in infinite well (xc = 0 and n = 1,2,3,...).
    x is position.
    n is energy level.
    L is width of infinite well (xc = 0 and n = 1,2,3,...)."""
    half_L = L / 2 # getting half the well depth
    psi = sqrt(2 / L) * sin(n * pi / L * (x + L / 2)) # wavefunction 
    psi[where(x <= -half_L)] = 0 # setting values outside well to be 0
    psi[where(x >= half_L)] = 0 # setting values outside well to be 0
    return psi

def psi_tot(X, Y, Z, 
            nx, ny, nz,
            Lx, Ly, Lz):
    """Returns plottable total wavefunction.
    X,Y,Z are space meshgrids.
    nx,ny,nz are orbital quantum numbers for the three dimensions.
    Lx, Ly, Lz are well depths."""
    psi_x = psi(X, nx, Lx)
    psi_y = psi(Y, ny, Ly)
    psi_z = psi(Z, nz, Lz)
    psi_tot = psi_x * psi_y * psi_z  
    return psi_tot / norm(psi_tot)