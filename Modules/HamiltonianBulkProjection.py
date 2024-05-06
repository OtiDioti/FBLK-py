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
def i_k_i(bra, ket, soft = 1e-15):
    """Returns the expecation values <n_i|i ki|m_i> and <n_i|ki i|m_i>.
    """
    nx_bra, ny_bra, nz_bra = bra # extracting orbital states for each direction
    nx_ket, ny_ket, nz_ket = ket # extracting orbital states for each direction
    
    iki = lambda n_bra, n_ket: -1j * (1 + (-1)**(n_ket + n_bra)) * n_ket * n_bra / (n_ket**2 - n_bra**2 + soft) * (n_bra!=n_ket) + 0.5j * (n_bra==n_ket)
    kii = lambda n_bra, n_ket: iki(n_bra, n_ket) * (n_bra!=n_ket)  - 0.5j * (n_bra==n_ket) 
    
    xkx = iki(nx_bra, nx_ket) # <n_x|x kx|m_x>
    kxx = kii(nx_bra, nx_ket) # <n_x|kx x|m_x>
    yky = iki(ny_bra, ny_ket) # <n_y|y k|m_y> 
    kyy = kii(ny_bra, ny_ket) # <n_y|ky y|m_y>
    zkz = iki(nz_bra, nz_ket) # <n_z|z kz|m_z>
    kzz = kii(nz_bra, nz_ket) # <n_z|kz z|m_z>
    return xkx, kxx, yky, kyy, zkz, kzz

def k_ik_j(bra, ket,
           A,
           Lx, Ly, Lz,
           soft = 1e-15):
    """Returns list of expectation value <bra| k_ik_j = hbar^2 p_ip_j |ket> for infinite well (xc=0 and n = 1,2,3,...)
    eigenstates: each element of list is expectation value for the different dimensions x,y,z.
    bra, ket are tuples representing infinite well states.
    A is the vector potential to be used in minimal coup. substitution. This is of the form 
                           A = [Ax_x * x, Ax_y * y, Ax_z * z, Ax_c, 
                                Ay_x * x, Ay_y * y, Ay_z * z, Ay_c, 
                                Az_x * x, Az_y * y, Az_z * z, Ax_c]
    where x,y,z are the position operators, and Ai_j is the ith component of A multipling the j operator in that component.
    L_i is length of well in ith direction.
    soft is a softening factor to avoid singularities from blowing up (purely numerical object).
    """
    
    pos_op = lambda n_bra, n_ket, L: 4 * (-1 + (-1)**(n_ket + n_bra)) * n_ket * n_bra * L / ((n_ket**2 - n_bra**2 + soft)**2 * pi**2)    
    pos_op2 = lambda n_bra, n_ket, L: 4 * (1 + (-1)**(n_ket + n_bra)) * n_ket * n_bra * L**2 / ((n_ket**2 - n_bra**2 + soft)**2 * pi**2) * (n_bra!=n_ket) + 1/12 * L**2 * (1 - 6/(n_ket**2 * pi**2)) * (n_bra==n_ket) 
    k = lambda n_bra, n_ket, L: - 2j * (-1 + (-1)**(n_ket + n_bra)) * n_ket * n_bra / (L * (n_ket**2 - n_bra**2 + soft))

    nx_bra, ny_bra, nz_bra = bra # extracting orbital states for each direction
    nx_ket, ny_ket, nz_ket = ket # extracting orbital states for each direction
    # calculating vector potential
    x, y, z, x2, y2, z2 = [pos_op(nx_bra, nx_ket, Lx),
                           pos_op(ny_bra, ny_ket, Ly),
                           pos_op(nz_bra, nz_ket, Lz),
                           pos_op2(nx_bra, nx_ket, Lx),
                           pos_op2(ny_bra, ny_ket, Ly),
                           pos_op2(nz_bra, nz_ket, Lz)] # position expectation values
    
    Axx, Axy, Axz, Axc = [A[0,0] * x, A[0,1] * y, A[0,2] * z, A[0,3]]
    Ayx, Ayy, Ayz, Ayc = [A[1,0] * x, A[1,1] * y, A[1,2] * z, A[1,3]]
    Azx, Azy, Azz, Azc = [A[2,0] * x, A[2,1] * y, A[2,2] * z, A[2,3]]
    
    # calculating momenta
    kx = k(nx_bra, nx_ket, Lx) # <n_x|kx|m_x> 
    ky = k(ny_bra, ny_ket, Ly) # <n_y|ky|m_y>
    kz = k(nz_bra, nz_ket, Lz) # <n_z|kz|m_z>
    xkx, kxx, yky, kyy, zkz, kzz = i_k_i(bra, ket) # expecation values <n_i|i ki|m_i> and <n_i|ki i|m_i>
    # dirac delta functions
    dx = nx_bra==nx_ket # d_{nx=mx}
    dy = ny_bra==ny_ket # d_{ny=my}
    dz = nz_bra==nz_ket # d_{nz=mz}
    dxdy = dx*dy # d_{nx=mx} d_{ny=my}
    dxdz = dx*dz # d_{nx=mx} d_{nz=mz}
    dydz = dy*dz # d_{ny=my} d_{nz=mz}
    dxdydz = dxdy*dz # d_{nx=mx} d_{ny=my} d_{nz=mz} 
    
    # cross momenta after min. coup. substitution
    kxky = Sum(array([kx * ky * dz, kxx * A[1,0] * dydz, kx * Ayy * dz, kx * Ayz * dy, kx * Ayc * dydz,
                      Axx * ky * dz, A[0,0]*A[1,0] * x2 * dydz, Axx*Ayy * dz, Axx*Ayz * dy, Axx*Ayc * dydz,
                      A[0,1] * yky * dxdz, Axy*Ayx * dz, A[0,1]*A[1,1] * y2 * dxdz, Axy*Ayz * dx, Axy*Ayc * dxdz,
                      Axz * ky * dx, Axz*Ayx * dy, Axz*Ayy * dx, A[0,2]*A[1,2] * z2 * dxdy, Axz*Ayc * dxdy,
                      Axc * ky * dxdz, Axc*Ayx * dydz, Axc*Ayy * dxdz, Axc*Ayz * dxdy, Axc*Ayc * dxdydz])) # <bra|(kx + Ax)(ky + Ay)|bra>
               
    kykx = Sum(array([ky * kx * dz, ky * Axx * dz, kyy * A[0,1] * dxdz, ky * Axz * dx, ky * Axc * dxdz,
                      A[1,0] * xkx * dydz, A[1,0]*A[0,0] * x2 * dydz, Ayx*Axy * dz, Ayx*Axz * dy, Ayx*Axc * dydz,
                      Ayy * kx * dz, Ayy*Axx * dz, A[1,1]*A[0,1] * y2 * dxdz, Ayy*Axz * dx, Ayy*Axc * dxdz,
                      Ayz * kx * dy, Ayz*Axx * dy, Ayz*Axy * dx, A[1,2]*A[0,2] * z2 * dxdy, Ayz*Axc * dxdy,
                      Ayc * kx * dydz, Ayc*Axx * dydz, Ayc*Axy * dxdz, Ayc*Axz * dxdy, Ayc*Axc * dxdydz])) # <bra|(ky + Ay)(kx + Ax)|bra>         

    kykz = Sum(array([ky * kz * dx, ky * Azx * dz, kyy * A[2,1] * dxdz, ky * Azz * dx, ky * Azc * dxdz,
                      Ayx * kz * dy, A[1,0]*A[2,0] * x2 * dydz, Ayx*Azy * dz, Ayx*Azz * dy, Ayx*Azc * dydz,
                      Ayy * kz * dx, Ayy*Azx * dz, A[1,1]*A[2,1] * y2 * dxdz, Ayy*Azz * dx, Ayy*Azc * dxdz,
                      A[1,2] * zkz * dxdy, Ayz*Azx * dy, Ayz*Azy * dx, A[1,2]*A[2,2] * z2 * dxdy, Ayz*Azc * dxdy,
                      Ayc * kz * dxdy, Ayc*Azx * dydz, Ayc*Azy * dxdz, Ayc*Azz * dxdy, Ayc*Azc * dxdydz])) # <bra|(ky + Ay)(kz + Az)|bra>       
                
    kzky = Sum(array([kz * ky * dx, kz * Ayx * dy, kz * Ayy * dx, kzz * A[1,2] * dxdy, kz * Ayc * dxdy,
                      Azx * ky * dz, A[2,0]*A[1,0] * x2 * dydz, Azx*Ayy * dz, Azx*Ayz * dy, Azx*Ayc * dydz,
                      A[2,1] * yky * dxdz, Azy*Ayx * dz, A[2,1]*A[1,1] * y2 * dxdz, Azy*Ayz * dx, Azy*Ayc * dxdz,
                      Azz * ky * dx, Azz*Ayx * dy, Azz*Ayy * dx, A[2,2]*A[1,2] * z2 * dxdy, Azz*Ayc * dxdy,
                      Azc * ky * dxdz, Azc*Ayx * dydz, Azc*Ayy * dxdz, Azc*Ayz * dxdy, Azc*Ayc * dxdydz])) # <bra|(kz + Az)(ky + Ay)|bra>
    
    kzkx = Sum(array([kz * kx * dy, kz * Axx * dy, kz * Axy * dx, kzz * A[0,2] * dxdy, kz * Axc * dxdy,
                      A[2,0] * xkx * dydz, A[2,0]*A[0,0] * x2 * dydz, Azx*Axy * dz, Azx*Axz * dy, Azx*Axc * dydz,
                      Azy * kx * dz, Azy*Axx * dz, A[2,1]*A[0,1] * y2 * dxdz, Azy*Axz * dx, Azy*Axc * dxdz,
                      Azz * kx * dy, Azz*Axx * dy, Azz*Axy * dx, A[2,2]*A[0,2] * z2 * dxdy, Azz*Axc * dxdy,
                      Azc * kx * dydz, Azc*Axx * dydz, Azc*Axy * dxdz, Azc*Axz * dxdy, Azc*Axc * dxdydz])) # <bra|(kz + Az)(kx + Ax)|bra>
    
    kxkz = Sum(array([kx * kz * dy, kxx * A[2,0] * dydz, kx * Azy * dz, kx * Azz * dy, kx * Azc * dydz,
                      Axx * kz * dy, A[0,0]*A[2,0] * x2 * dydz, Axx*Azy * dz, Axx*Azz * dy, Axx*Azc * dydz,
                      Axy * kz * dx, Axy*Azx * dz, A[0,1]*A[2,1] * y2 * dxdz, Axy*Azz * dx, Axy*Azc * dxdz,
                      A[0,2] * zkz * dxdy, Axz*Azx * dy, Axz*Azy * dx, A[0,2]*A[2,2] * z2 * dxdy, Axz*Azc * dxdy,
                      Axc * kz * dxdy, Axc*Azx * dydz, Axc*Azy * dxdz, Axc*Azz * dxdy, Axc*Azc * dxdydz])) # <bra|(kx + Ax)(kz + Az)|bra>  
               
    return kxky, kykx, kykz, kzky, kzkx, kxkz

def k2_i(bra, ket,
         A,
         Lx, Ly, Lz,
         soft = 1e-15):
    """Returns list of expectation value <bra| k^2 = hbar^2 p^2 |ket> for infinite well (xc=0 and n = 1,2,3,...)
    eigenstates: each element of list is expectation value for the different dimensions x,y,z.
    bra, ket are tuples representing infinite well states.
    A is the vector potential to be used in minimal coup. substitution. This is of the form 
                           A = [Ax_x * x, Ax_y * y, Ax_z * z, Ax_c, 
                                Ay_x * x, Ay_y * y, Ay_z * z, Ay_c, 
                                Az_x * x, Az_y * y, Az_z * z, Ax_c]
    where x,y,z are the position operators, and Ai_j is the ith component of A multipling the j operator in that component.
    L_i is length of well in ith direction.
    soft is a softening factor to avoid singularities from blowing up (purely numerical object).
    """
    pos_op = lambda n_bra, n_ket, L: 4 * (-1 + (-1)**(n_ket + n_bra)) * n_ket * n_bra * L / ((n_ket**2 - n_bra**2 + soft)**2 * pi**2)    
    pos_op2 = lambda n_bra, n_ket, L: 4 * (1 + (-1)**(n_ket + n_bra)) * n_ket * n_bra * L**2 / ((n_ket**2 - n_bra**2 + soft)**2 * pi**2) * (n_bra!=n_ket) + 1/12 * L**2 * (1 - 6/(n_ket**2 * pi**2)) * (n_bra==n_ket) 
    k = lambda n_bra, n_ket, L: - 2j * (-1 + (-1)**(n_ket + n_bra)) * n_ket * n_bra / (L * (n_ket**2 - n_bra**2 + soft))
    k2 = lambda n_bra, n_ket, L: n_ket**2 * pi**2 / (L**2) * (n_bra==n_ket)
    
    nx_bra, ny_bra, nz_bra = bra # extracting orbital states for each direction
    nx_ket, ny_ket, nz_ket = ket # extracting orbital states for each direction
    
    # calculating vector potential
    x, y, z, x2, y2, z2 = [pos_op(nx_bra, nx_ket, Lx),
                           pos_op(ny_bra, ny_ket, Ly),
                           pos_op(nz_bra, nz_ket, Lz),
                           pos_op2(nx_bra, nx_ket, Lx),
                           pos_op2(ny_bra, ny_ket, Ly),
                           pos_op2(nz_bra, nz_ket, Lz)] # position expectation values
    
    Axx, Axy, Axz, Axc = [A[0,0] * x, A[0,1] * y, A[0,2] * z, A[0,3]]
    Ayx, Ayy, Ayz, Ayc = [A[1,0] * x, A[1,1] * y, A[1,2] * z, A[1,3]]
    Azx, Azy, Azz, Azc = [A[2,0] * x, A[2,1] * y, A[2,2] * z, A[2,3]]
    
    # calculating momenta
    kx = k(nx_bra, nx_ket, Lx) # <n_x|kx|m_x> 
    ky = k(ny_bra, ny_ket, Ly) # <n_y|ky|m_y>
    kz = k(nz_bra, nz_ket, Lz) # <n_z|kz|m_z>
    xkx, kxx, yky, kyy, zkz, kzz = i_k_i(bra, ket) # expecation values <n_i|i ki|m_i> and <n_i|ki i|m_i>
    # dirac delta functions
    dx = nx_bra==nx_ket # d_{nx=mx}
    dy = ny_bra==ny_ket # d_{ny=my}
    dz = nz_bra==nz_ket # d_{nz=mz}
    dxdy = dx*dy # d_{nx=mx} d_{ny=my}
    dxdz = dx*dz # d_{nx=mx} d_{nz=mz}
    dydz = dy*dz # d_{ny=my} d_{nz=mz}
    dxdydz = dxdy*dz # d_{nx=mx} d_{ny=my} d_{nz=mz}
    # calculating square momenta
    kx2 = Sum(array([k2(nx_bra, nx_ket, Lx) * dydz, kxx * A[0,0] * dydz, kx * Axy * dz, kx * Axz * dy, kx * Axc * dydz,
                     A[0,0] * xkx * dydz, A[0,0]*A[0,0] * x2 * dydz, Axx*Axy * dz, Axx*Axz * dy, Axx*Axc * dydz,
                     Axy * kx * dz, Axy*Axx * dz, A[0,1]*A[0,1] * y2 * dxdz, Axy*Axz * dx, Axy*Axc * dxdz,
                     Axz * kx * dy, Axz*Axx * dy, Axz*Axy * dx, A[0,2]*A[0,2] * z2 * dxdy, Axz*Axc * dxdy,
                     Axc * kx * dydz, Axc*Axx * dydz, Axc*Axy * dxdz, Axc*Axz * dxdy, Axc*Axc * dxdydz])) # <bra|(kx + Ax)^2|ket>
    
    ky2 = Sum(array([k2(ny_bra, ny_ket, Ly) * dxdz, ky * Ayx * dz, kyy * A[1,1] * dxdz, ky * Ayz * dx, ky * Axc * dxdz,
                     Ayx * ky * dz, A[1,0]*A[1,0] * x2 * dxdz, Ayx*Ayy * dz, Ayx*Ayz * dy, Ayx*Ayc * dydz,
                     A[1,1] * yky * dxdz, Ayy*Ayx * dz, A[1,1]*A[1,1] * y2 * dxdz, Ayy*Ayz * dx, Ayy*Ayc * dxdz,
                     Ayz * ky * dx, Ayz*Ayx * dy, Ayz*Ayy * dx, A[1,2]*A[1,2] * z2 * dxdy, Ayz*Ayc * dxdy,
                     Ayc * ky * dxdz, Ayc*Ayx * dydz, Ayc*Ayy * dxdz, Ayc*Ayz * dxdy, Ayc*Ayc * dxdydz])) # <bra|(ky + Ay)^2|ket>
    
    kz2 = Sum(array([k2(nz_bra, nz_ket, Lz) * dxdy, kz * Azx * dy, kz * Azy * dx, kzz * A[2,2] * dxdy, kz * Axc * dxdy,
                     Azx * kz * dy, A[2,0]*A[2,0] * x2 * dydz, Azx*Azy * dz, Azx*Azz * dy, Azx*Azc * dydz,
                     Azy * kz * dx, Azy*Azx * dz, A[2,1]*A[2,1] * y2 * dxdz, Azy*Azz * dx, Azy*Azc * dxdz,
                     A[2,2] * zkz * dxdy, Azz*Azx * dy, Azz*Azy * dx, A[2,2]*A[2,2] * z2 * dxdy, Azz*Azc * dxdy,
                     Azc * kz * dxdy, Azc*Azx * dydz, Azc*Azy * dxdz, Azc*Azz * dxdy, Azc*Azc * dxdydz])) # <bra|(kz + Az)^2|ket>
    
    return kx2, ky2, kz2
#%% defining hamiltonian terms
def h_lk(kx2, ky2, kz2,
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
    first_term = -0.5 * (g1 + 2.5 * g2) * (kx2 + ky2 + kz2) * J0 # first term of LK Hamiltonian
    secon_term = g2 * (kx2 * Jx.__pow__(2) + ky2 * Jy.__pow__(2) + kz2 * Jz.__pow__(2)) # second term of LK Hamiltonian
    third_term = 0.5 * g3 * ((kxky + kykx) * anti_comm(Jx, Jy) + (kykz + kzky) * anti_comm(Jy, Jz) + (kzkx + kxkz) * anti_comm(Jz, Jx)) # third term of LK Hamiltonian
    
    h_lk = first_term + secon_term + third_term # magnetic field free HLK
    
    return h_lk

def h_z(kappa = 1, B = [0,0,0]):
    """Returns zeeman hamiltonian.
    k_i is expectation value of k_i operator.
    ki_2 is expectation value of operator k_i^2.
    k_ik_j is expectation value of operator k_ik_j.
    kappa is the magnetic g-factor.
    B is the magnetic field vector.
    """ 
    return kappa * (Jx * B[0] + Jy * B[1] + Jz * B[2])

def h_tot(kx2, ky2, kz2,
          kxky, kykx, kykz, kzky, kzkx, kxkz, 
          g1 = 13.35, g2 = 4.25, g3 = 5.69,
          kappa = 1, B = [0,0,0]):
    """Returns total hamiltonian as well as plottable well potential.
    k_i is expectation value of k_i operator.
    ki_2 is expectation value of operator k_i^2.
    k_ik_j is expectation value of operator k_ik_j.
    g1, g2, g3 are Luttinger poaramteters (Default: Luttinger params for Ge).
    B is magnetic field vector.
    kappa is magnetic g-factor.
    conf is the confinement type desired (can be "bulk", "planar", "wire").
    """
    
    lk = h_lk(kx2, ky2, kz2, kxky, kykx, kykz, kzky, kzkx, kxkz, g1, g2, g3) # lk hamiltonian
    z = h_z(kappa, B)
    return - lk + z

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
#%% defining vectorized hamiltonian terms (roughly 6k times faster than functions above)
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

def h_tot_v(k2, kikj,
            dim,
            g1 = 13.35, g2 = 4.25, g3 = 5.69,
            kappa = 1, B = [0,0,0]):
    """Returns total hamiltonian as well as plottable well potential.
    k_i is expectation value of k_i operator.
    ki_2 is expectation value of operator k_i^2.
    k_ik_j is expectation value of operator k_ik_j.
    g1, g2, g3 are Luttinger poaramteters (Default: Luttinger params for Ge).
    B is magnetic field vector.
    kappa is magnetic g-factor.
    conf is the confinement type desired (can be "bulk", "planar", "wire").
    """
    kx2, ky2, kz2 = [k2[:,:,0], k2[:,:,1], k2[:,:,2]]
    kxky, kykx, kykz, kzky, kzkx, kxkz = [kikj[:,:,0], kikj[:,:,1],
                                          kikj[:,:,2], kikj[:,:,3],
                                          kikj[:,:,4], kikj[:,:,5]]
    lk = h_lk_v(kx2, ky2, kz2, kxky, kykx, kykz, kzky, kzkx, kxkz, g1, g2, g3) # lk hamiltonian
    z = h_z_v(dim, kappa, B)
    return - lk + z
#%%
def get_ks(possible_statess, 
           A, 
           Lx, Ly, Lz,
           dim):
    """Returns tuple of 2 ndarrays for the expectation values of k^2 and kikj opertators.
    (Used in Projection method).
    possible_statess is array of all possible permutations of basis states.
    A is magnetic vector potential.
    L_i is well depth in the three directions.
    dim is orbital dimension of the problem.
    """
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