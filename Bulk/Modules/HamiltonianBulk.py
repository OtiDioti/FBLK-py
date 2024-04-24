"""In this file we explore the luttinger kohn hamiltonian
for the bulk as well as other confinement geometries.
Note that we here consider hbar = me = e = 1.
"""
# my imports
import sys
import os
current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_path)
from UsefulFunctions import J0, Jx, Jy, Jz, k_i, k2_i, anti_comm

# other imports
from scipy.sparse import kron, diags, eye
from numpy import zeros, where, mgrid, diff, array
#%% Terms Precalculator
def preparing_terms(boundx_low, boundx_upp, boundy_low, boundy_upp, boundz_low, boundz_upp,
                    dimx, dimy, dimz,
                    A = array([[0,0,0],[0,0,0],[0,0,0]]), B = [0,0,0],
                    coeff_x = 1, coeff_y = 1, coeff_z = 1,
                    bc = 0):
    """Returns dictionary of discretized form of momentum operators & respective identities.
    boundi_low, boundi_upp are lower and upper bounds in x, y and z directions.
    dimx, dimy, dimz are discretization numbers along each direction.
    
    A is the vector potential to be used in minimal coup. substitution. This is of the form 
                            A = (Ax_x * x + Ax_y * y + Ax_z * z, 
                                 Ay_x * x + Ay_y * y + Ay_z * z, 
                                 Az_x * x + Az_y * y + Az_z * z) 
    where x,y,z are the position operators, and Ai_j is the ith component of A multipling the j operator in that component.
    
    B is magnetic field vector.
    coeff_x, coeff_y, coeff_z indicate at what fraction along x (or y, z) the well lifts to infty.
    bc is the selected boundary condition (bc=0 -> Dirichlect; bc=1 -> Neumann).
    """
    
    # creating mesh grid
    X, Y, Z = mgrid[boundx_low : boundx_upp : dimx*1j, 
                    boundy_low : boundy_upp : dimy*1j,
                    boundz_low : boundz_upp : dimz*1j] # meshgrid

    dx = diff(X[:,0,0])[0] # spacing in x
    dy = diff(Y[0,:,0])[0] # spacing in y
    dz = diff(Z[0,0,:])[0] # spacing in z

    # identities in space space
    Ix = eye(dimx) # identity in x-space
    Iy = eye(dimy) # identity in y-space
    Iz = eye(dimz) # identity in z-space
    
    # obtaining position operators
    x, y, z = [kron(kron(diags(X[:,0,0]), Iy), Iz), 
               kron(kron(Ix, diags(Y[0,:,0])), Iz),
               kron(kron(Ix, Iy), diags(Z[0,0,:]))] # position operators
    
    # creating vector potential from A coefficient list
    Ax = A[0][0] * x + A[0][1] * y + A[0][2] * z # x component of vector potential
    Ay = A[1][0] * x + A[1][1] * y + A[1][2] * z # y component of vector potential
    Az = A[2][0] * x + A[2][1] * y + A[2][2] * z # z component of vector potential
    
    """
    # obtaining momentum operators
    kx_1d, ky_1d, kz_1d = [k_i(dimx, dx, bc), 
                           k_i(dimy, dy, bc), 
                           k_i(dimz, dz, bc)] # momentum operators
    
    kx2, ky2, kz2 = [kron(kron(k2_i(dimx, dx, bc), Iy), Iz), 
                     kron(kron(Ix, k2_i(dimy, dy, bc)), Iz), 
                     kron(kron(Ix, Iy), k2_i(dimz, dz, bc))] # squaredmomentum operators (kx2 = kx2 x Iy x Iz, ky2 = Ix x ky2 x Iz, kz2 = Ix x Iy x kz2)
    
    kxky = kron(kron(kx_1d, ky_1d), Iz) # kxky (in theory these commute, but this may not be the case if we make a min. coup. substitution)
    kykx = kron(kron(ky_1d, kx_1d), Iz) # kykx
    
    kykz = kron(kron(Ix, ky_1d), kz_1d) # kykz (in theory these commute, but this may not be the case if we make a min. coup. substitution)
    kzky = kron(kron(Ix, kz_1d), ky_1d) # kzky
    
    kzkx = kron(kron(kz_1d, Iy), kx_1d) # kzkx (in theory these commute, but this may not be the case if we make a min. coup. substitution)
    kxkz = kron(kron(kx_1d, Iy), kz_1d) # kxkz
    """
    
    # obtaining momentum operators after minimal coupling substitution
    kx, ky, kz = [kron(kron(k_i(dimx, dx, bc), Iy), Iz) + Ax, 
                  kron(kron(Ix, k_i(dimy, dy, bc)), Iz) + Ay, 
                  kron(kron(Ix, Iy), k_i(dimz, dz, bc)) + Az] # momentum operators
    
    kx2, ky2, kz2 = [kron(kron(k2_i(dimx, dx, bc), Iy), Iz) + 2 * (kx @ Ax) + Ax.__pow__(2), 
                     kron(kron(Ix, k2_i(dimy, dy, bc)), Iz) + 2 * (ky @ Ay) + Ay.__pow__(2), 
                     kron(kron(Ix, Iy), k2_i(dimz, dz, bc)) + 2 * (kz @ Az) + Az.__pow__(2)] # squaredmomentum operators (kx2 = kx2 x Iy x Iz, ky2 = Ix x ky2 x Iz, kz2 = Ix x Iy x kz2)
    
    kxky = kx @ ky # kxky (in theory these commute, but this may not be the case if we make a min. coup. substitution)
    kykx = ky @ ky # kykx
    
    kykz = ky @ kz # kykz (in theory these commute, but this may not be the case if we make a min. coup. substitution)
    kzky = kz @ ky # kzky
    
    kzkx = kz @ kx # kzkx (in theory these commute, but this may not be the case if we make a min. coup. substitution)
    kxkz = kx @ kz # kxkz
    
    # setting bounds for infinite well in x and y
    half_Lx = coeff_x * (boundx_upp - boundx_low)/2 # half of the well width in x
    half_Ly = coeff_y * (boundy_upp - boundy_low)/2 # half of the well width in y
    half_Lz = coeff_z * (boundz_upp - boundz_low)/2 # half of the well width in y
    
    return {"grids" : {"X" : X,
                       "Y" : Y,
                       "Z" : Z},
            "spacings" : {"dx" : dx,
                          "dy" : dy,
                          "dz" : dz},
            "identities" : {"Ix" : Ix,
                            "Iy" : Iy,
                            "Iz" : Iz},
            "position" : {"x" : x,
                          "y" : y,
                          "z" : z},
            "momenta" : {"kx2" : kx2, "ky2" : ky2, "kz2" : kz2}, 
            "cross-momenta" : {"kxky" : kxky, "kykx" : kykx, 
                               "kykz" : kykz, "kzky" : kzky,
                               "kzkx" : kzkx, "kxkz" : kxkz},
            "well-walls" : {"half_Lx" : half_Lx,
                            "half_Ly" : half_Ly,
                            "half_Lz" : half_Lz}}


#%% Bulk LK Hamiltonian

def h_lk(kx2, ky2, kz2,
         kxky, kykx, kykz, kzky, kzkx, kxkz,
        g1 = 13.35, g2 = 4.25, g3 = 5.69):
    """Returns discritezied LK Hamiltonian (according to finite element method), in sparse format.
    momenta_arrays contains all discretized momentum operators needed.
    g1, g2, g3 are Luttinger poaramteters (Default: Luttinger params for Ge).
    """

    # magnetic field free HLK
    first_term = kron(-0.5 * (g1 + 2.5 * g2) * (kx2 + ky2 + kz2), J0) # first term of LK Hamiltonian
    secon_term = g2 * (kron(kx2, Jx.__pow__(2)) + kron(ky2, Jy.__pow__(2)) + kron(kz2, Jz.__pow__(2))) # second term of LK Hamiltonian
    third_term = 0.5 * g3 * (kron((kxky + kykx), anti_comm(Jx, Jy)) + kron((kykz + kzky), anti_comm(Jy, Jz)) + kron((kzkx + kxkz), anti_comm(Jz, Jx))) # third term of LK Hamiltonian
    
    h_lk = first_term + secon_term + third_term # magnetic field free HLK
    
    return h_lk

def h_z(Ix, Iy, Iz,
        kappa = 1, B = [0,0,0]):
    """Returns zeeman hamiltonian.
    Ix, Iy, Iz are the discretized identities in space for the three dimensions.
    kappa is the magnetic g-factor.
    B is the magnetic field vector.
    """ 
    return kron(kron(kron(Ix, Iy), Iz), kappa * (Jx * B[0] + Jy * B[1] + Jz * B[2]))

def h_tot(needed_arrays, 
          g1 = 13.35, g2 = 4.25, g3 = 5.69,
          kappa = 1, B = [0,0,0], infinity = 1e10,
          conf = "bulk"):
    """Returns total hamiltonian as well as plottable well potential.
    needed_arrays is obtained via preparing_terms().
    g1, g2, g3 are Luttinger poaramteters (Default: Luttinger params for Ge).
    B is magnetic field vector.
    kappa is magnetic g-factor.
    infinity is height of infinite well.
    conf is the confinement type desired (can be "bulk", "planar", "wire").
    """
    
    # Extracting needed terms
    X, Y, Z = [needed_arrays["grids"]["X"],
               needed_arrays["grids"]["Y"],
               needed_arrays["grids"]["Z"]] # grids for x, y, z position
    Ix, Iy, Iz = [needed_arrays["identities"]["Ix"],
                  needed_arrays["identities"]["Iy"],
                  needed_arrays["identities"]["Iz"]] # identity operators in space space
    kx2, ky2, kz2  = [needed_arrays["momenta"]["kx2"], 
                      needed_arrays["momenta"]["ky2"],
                      needed_arrays["momenta"]["kz2"]] # squared momentum operators
    kxky, kykx, kykz, kzky, kzkx, kxkz =  [needed_arrays["cross-momenta"]["kxky"], 
                                           needed_arrays["cross-momenta"]["kykx"],
                                           needed_arrays["cross-momenta"]["kykz"],
                                           needed_arrays["cross-momenta"]["kzky"],
                                           needed_arrays["cross-momenta"]["kzkx"],
                                           needed_arrays["cross-momenta"]["kxkz"]] # cross momentum operators
    half_Lx, half_Ly, half_Lz = [needed_arrays["well-walls"]["half_Lx"],
                                 needed_arrays["well-walls"]["half_Ly"],
                                 needed_arrays["well-walls"]["half_Lz"]] # half of well width in x, y, z
    
    lk = h_lk(kx2, ky2, kz2, kxky, kykx, kykz, kzky, kzkx, kxkz, g1, g2, g3) # lk hamiltonian
    z = h_z(Ix, Iy, Iz, kappa, B)
    
    if conf == "bulk": # if we are looking at bulk structure
        return - lk + z, None
    elif conf == "planar": # if we are looking at planar structure
        v, u = get_potential_planar(Z, half_Lz, infinity)
        return - lk + z + v, u
    elif conf == "wire": # if we are looking at wire structure
        v, u = get_potential_wire(X, Y, Z, half_Lx, half_Ly, half_Lz, infinity)
        return - lk + z + v, u

#%% Potentials
def get_potential_planar(z,
                         half_Lz,
                         infinity = 1e10): 
    """Returns tuple with infinite well potential in z in diagonal form (to be used for calcs) and in plottable format.
    z is the position grid z-direction.
    half_Lz, is half length of the well in z centered at the origin.
    infinity is the height of the well.
    """
    
    shape = z.shape
    U = zeros(shape) # initializing grid with same shape as z (or x and y, grid)
    U[where(z <= -half_Lz)] = infinity # raising walls outside well z
    U[where(z >= half_Lz)] = infinity # raising walls outside well in z
   
    V = kron(diags(U.reshape(shape[0] * shape[1] * shape[2]), (0)), J0) # This reshaping is needed in order to obtain a (N2,N2,N2,4) Hamiltonian to diagonalize
    return V, U

def get_potential_wire(X, Y, Z,
                       half_Lx, half_Ly, half_Lz,
                       infinity = 1e10): 
    """Returns tuple with infinite well potential in z, x, y in diagonal form (to be used for calcs) and in plottable format.
    X, Y, Z are the position grids in the three directions.
    half_Lx, half_Ly, half_Lz, is half length of the well in z centered at the origin.
    infinity is the height of the well.
    """
    
    shape = Z.shape
    U = zeros(shape) # initializing grid with same shape as z (or x and y, grid)
    U[where(X <= -half_Lx)] = infinity # raising walls outside well x
    U[where(X >= half_Lx)] = infinity # raising walls outside well in x
    
    U[where(Y <= -half_Ly)] = infinity # raising walls outside well y
    U[where(Y >= half_Ly)] = infinity # raising walls outside well in y
    
    U[where(Z <= -half_Lz)] = infinity # raising walls outside well z
    U[where(Z >= half_Lz)] = infinity # raising walls outside well in z
   
    V = kron(diags(U.reshape(shape[0] * shape[1] * shape[2]), (0)), J0) # This reshaping is needed in order to obtain a (N3,N3,N3,4) Hamiltonian to diagonalize
    return V, U

