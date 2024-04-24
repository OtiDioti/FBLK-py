"""In this file we explore the luttinger kohn hamiltonian.
for the planar confinement within the spherical approximation.
Note that we here consider hbar = me = e = 1.
"""
# my imports
import sys
import os
current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_path)
from UsefulFunctions import J0, Jx, Jy, Jz, k_i, k2_i

#scipy imports
from scipy.sparse import kron, block_array, diags, eye
from numpy import pi, sqrt, conjugate, zeros, where, mgrid, diff, exp
#%% Terms Precalculator

def preparing_terms(boundx_low, boundx_upp, boundy_low, boundy_upp,
                    dimx, dimy,
                    Lz, le,
                    coeff_x = 0.9, coeff_y = 0.9,
                    bc = 0,
                    g1 = 13.35, g2 = 4.25, g3 = 5.69, gs = 4.84):
    """Returns dictionary of discretized form of momentum operators & respective identities.
    boundx_low, boundx_upp, boundy_low, boundy_upp are lower and upper bounds in x and y directions.
    dimx, dimy are discretization numbers along each direction.
    Lz is width of well in z, centered at origo.
    le is characteristic length of electric field in z-well.
    coeff_x, coeff_y indicate at what fraction along x (or y) the well lifts to infty.
    bc is the selected boundary condition (bc=0 -> Dirichlect; bc=1 -> Neumann).
    g1, g2, g3 are Luttinger poaramteters (Default: Luttinger params for Ge).
    """
    # creating mesh grid
    X, Y = mgrid[boundx_low : boundx_upp : dimx*1j, 
                 boundy_low : boundy_upp : dimy*1j] # meshgrid

    dx = diff(X[:,0])[0] # spacing in x
    dy = diff(Y[0,:])[0] # spacing in y

    # identities in space space
    Ix = eye(dimx) # identity in x-space
    Iy = eye(dimy) # identity in y-space
    
    # obtaining position operators
    x, y = [kron(diags(X[:,0]), Iy), 
            kron(Ix, diags(Y[0,:]))] # position operators
    
    # obtaining momentum operators
    kx_1d, ky_1d = [k_i(dimx, dx, bc), k_i(dimy, dy, bc)] # momentum operators in 1d
    kx, ky = [kron(kx_1d, Iy), kron(Ix, ky_1d)] # moementum operators in 2d
    kx2, ky2 = [kron(k2_i(dimx, dx, bc), Iy), 
                kron(Ix, k2_i(dimy, dy, bc))] # squaredmomentum operators (kx2 = kx2 x Iy x Iz, ky2 = Ix x ky2 x Iz)
    
    kxky = kron(kx_1d, ky_1d) # kxky (in theory these commute, but this may not be the case if we make a min. coup. substitution)
    
    # obtaining energy for ground state of triangular well (low e-field)
    gz_h = g1 - 2 * gs
    gz_l = g1 + 2 * gs
    
    en_h = gz_h * pi**2 / (2 * Lz**2) - Lz**4 * (pi**2 - 6)**2 / (288 * pi**4 * gz_h * le**6) # heavy hole energy along z
    en_l = gz_l * pi**2 / (2 * Lz**2) - Lz**4 * (pi**2 - 6)**2 / (288 * pi**4 * gz_l * le**6) # light hole energy along z
    
    # obtaining expectation values and overlaps between z-states
    brah_ketl = 1 - (Lz / le)**6 * (pi**2 - 6)**3 * gs**2 / (216 * pi**6 * (g1**2 - 4 * gs**2)**2) # <phi_H|phi_L>
    brah_kz_ketl = -1j * Lz**2 * (pi**2 - 6) * gs / (6 * le**3 * pi**2 * (g1 ** 2 - 4 * gs**2)) # <phi_H|kz|phi_L>
    
    # setting bounds for infinite well in x and y
    half_Lx = coeff_x * (boundx_upp - boundx_low)/2 # half of the well width in x
    half_Ly = coeff_y * (boundy_upp - boundy_low)/2 # half of the well width in y
    
    return {"grids" : {"X" : X,
                       "Y" : Y},
            "spacings" : {"dx" : dx,
                          "dy" : dy},
            "identities" : {"Ix" : Ix,
                            "Iy" : Iy},
            "position" : {"x" : x,
                          "y" : y},
            "momenta" : {"kx" :  kx, "kx2" : kx2, 
                         "ky" :  ky, "ky2" : ky2}, 
            "cross-momenta" : {"kxky" : kxky},
            "energies_z" : {"e_hl" : kron(Ix, Iy) * en_h, "e_lh" : kron(Ix, Iy) * en_l},
            "expectations_z" : {"<phi_H|phi_L>" : brah_ketl,
                                "<phi_H|kz|phi_L>" : brah_kz_ketl},
            "well-walls" : {"half_Lx" : half_Lx,
                            "half_Ly" : half_Ly}
            }

#%% Spherical approximation
def f_spherical(kx2, ky2, en_hh,
                gs, gk):
    """Returns f term of h_lk.
    kx, ky are momentum operators in x and y directions.
    en_hh energy of hh in infinite triangular well centered at origin.
    gs, gk are dependendent on luttinger parameters (gk = g1 + gs * 5/2, gs = 4.84).
    """
    
    return 0.5 * (gk - 3 * 0.5 * gs) * (kx2 + ky2) + 0.5 * en_hh

def g_spherical(kx2, ky2, en_lh,
                gs, gk):
    """Returns g term of h_lk.
    kx, ky are momentum operators in x and y directions.
    en_lh energy of hh in infinite triangular well centered at origin.
    gs, gk are dependendent on luttinger parameters (gk = g1 + gs * 5/2, gs = 4.84).
    """
    
    return 0.5 * (gk - 7 * 0.5 * gs) * (kx2 + ky2) + 0.5 * en_lh

def m_spherical(kx2, ky2, over,
                gs, gk):
    """Returns l term of h_lk.
    kx2, ky2 are squared momentum operators in x and y directions.
    over is the overlap <phi_hh|phi_lh>.
    gs, gk are dependendent on luttinger parameters (gk = g1 + gs * 5/2, gs = 4.84).
    """

    return 0.5 * sqrt(3) * gs * (ky2 - kx2) * over

def h_lk_spherical(kx2, ky2,
                   over,
                   en_l, en_h,
                   gs, gk):
    """Returns discritezied LK Hamiltonian (according to finite element method), in sparse format.
    kx2, ky2 are squared momentum operators in x and y directions.
    over is overlap <phi_hh|phi_lh>.
    en_hh energy of hh in infinite triangular well centered at origin.
    gs, gk are dependendent on luttinger parameters (gk = g1 + gs * 5/2, gs = 4.84).
    """
    # calculating each term in h_lk
    ff = f_spherical(kx2, ky2, en_h, gs, gk)
    gg = g_spherical(kx2, ky2, en_l, gs, gk)
    mm = m_spherical(kx2, ky2, over, gs, gk)
        
    # magnetic field free HLK
    h_lk = block_array([[ff, None, None, mm],
                        [None, ff, mm, None],
                        [None, mm, gg, None],
                        [mm, None, None, gg]], format = "coo") 
    
    return h_lk
#%% orbital effects hamiltonian WARNING: this only supports effects generated by A = (0, x Bz, Az), where Az has no operator dependece
def f_orb_spherical(kx, ky,
                    x, 
                    Az, bz, gs, gk):
    """Return f term of orbital part of h_lk.
    kx, ky are momentum operators in x and y directions.
    x is position operator in x (i.e. diagonal with x values along diagonal x Iy x Iz).
    Az is the z-component of magnetic vector potential (WARNING: operator dependence not supported).
    bz is magnetic field along z-direction.
    gs, gk are dependendent on luttinger parameters (gk = g1 + g2 * 5/2, gs = 4.84).
    """
    first_term = 0 # kz * (Az * gk - (9 * Az * gs)/2) (note: <phi_HH|kz|phi_HH> = 0)
    secon_term = ky * (bz * gk * x - (3 * bz * gs * x)/2)
    third_term = 1/4 * gs * (-9 * Az**2 - 3 * bz**2 * x**2) 
    fourt_term = 1/4 * gk * (2 * Az**2 + 2 * bz**2 * x**2)
    return  first_term + secon_term + third_term + fourt_term

def g_orb_spherical(kx, ky,
                    x, 
                    Az, bz, gs, gk):
    """Return f term of orbital part of h_lk.
    kx, ky are momentum operators in x and y directions.
    x is position operator in x (i.e. diagonal with x values along diagonal x Iy).
    Az is the z-component of magnetic vector potential (WARNING: operator dependence not supported).
    bz is magnetic field along z-direction.
    gs, gk are dependendent on luttinger parameters (gk = g1 + g2 * 5/2, gs = 4.84).
    """
    first_term = 0 # kz * (Az * gk - (Az * gs)/2) (note: <phi_LH|kz|phi_LH> = 0)
    secon_term = ky * (bz * gk * x - (7 * bz * gs * x)/2)
    third_term = 1/4 * gs * (-Az**2 - 7 * bz**2 * x**2)
    fourt_term = 1/4 * gk * (2 * Az**2 + 2 * bz**2 * x**2)
    return  first_term + secon_term + third_term + fourt_term

def m_orb_spherical(kx, ky,
                    x, 
                    over,
                    Az, bz, gs, gk):
    """Return f term of orbital part of h_lk.
    kx, ky are momentum operators in x and y directions.
    x is position operator in x (i.e. diagonal with x values along diagonal x Iy).
    over is overlap <phi_H|phi_L>.
    Az is the z-component of magnetic vector potential (WARNING: operator dependence not supported).
    bz is magnetic field along z-direction.
    gs, gk are dependendent on luttinger parameters (gk = g1 + g2 * 5/2, gs = 4.84).
    """
    first_term = sqrt(3) * bz * gs * ky @ x * over
    secon_term = 0.5 * sqrt(3) * gs * (bz**2 * x**2 + 4j * kx @ (bz * x) + 4j * (bz * x) @ kx) * over
    return  first_term + secon_term 

def l_orb_spherical(kx, ky,
                    x, 
                    over, expec,
                    Az, bz, gs, gk):
    """Return f term of orbital part of h_lk.
    kx, ky are momentum operators in x and y directions.
    x is position operator in x (i.e. diagonal with x values along diagonal x Iy).
    over is overlap <phi_H|phi_L>.
    expec is expectation value <phi_H|kz|phi_L>.
    Az is the z-component of magnetic vector potential (WARNING: operator dependence not supported).
    bz is magnetic field along z-direction.
    gs, gk are dependendent on luttinger parameters (gk = g1 + g2 * 5/2, gs = 4.84).
    """
    
    first_term = -4 * sqrt(3) * Az * gs * kx * over
    secon_term = 4j * sqrt(3) * Az * gs * ky * over
    third_term = 4j * sqrt(3) * bz * gs * expec * x
    return  first_term + secon_term + third_term


def h_orb_sperical(kx, ky,
                   x,
                   over, expec,
                   Az, bz, gs, gk):
    """Returns discritezied part of LK Hamiltonian contining orbital information.
    kx, ky are momentum operators in x and y directions.
    x is position operator in x (i.e. diagonal with x values along diagonal x Iy).
    over is overlap <phi_H|phi_L>.
    expec is expectation value <phi_H|kz|phi_L>.
    Az is the z-component of magnetic vector potential (WARNING: operator dependence not supported).
    bz is magnetic field along z-direction.
    gs, gk are dependendent on luttinger parameters (gk = g1 + g2 * 5/2, gs = 4.84).
    """

    # calculating each term in h_lk
    ff = f_orb_spherical(kx, ky, x, Az, bz, gs, gk)
    gg = g_orb_spherical(kx, ky, x, Az, bz, gs, gk)
    ll = l_orb_spherical(kx, ky, x, over, expec, Az, bz, gs, gk)
    mm = m_orb_spherical(kx, ky, x, over, Az, bz, gs, gk)
    
    # orbital part of HLK
    h_orb = block_array([[ff, None, ll, mm],
                         [None, ff, conjugate(mm), -conjugate(ll)],
                         [conjugate(ll), mm, gg, None],
                         [conjugate(mm), -ll, None, gg]], format = "coo") 
    
    return h_orb

def h_tot_spherical(needed_arrays,
                    Az, bz, kappa, gs, gk, infinity = 1e10):
    """Returns total hamiltonian as well as plottable well potential.
    needed_arrays is obtained via preparing_terms().
    Az is z-component of magnetic vector potential (WARNING: operator dependence not supported).
    bz is z-component of magnetic field.
    kappa is magnetic g-factor.
    gs, gk are dependendent on luttinger parameters (gk = g1 + g2 * 5/2, gs = 4.84).
    infinity is height of infinite well.
    """
    # Extracting needed terms
    X, Y = [needed_arrays["grids"]["X"],
            needed_arrays["grids"]["Y"]] # grids for x and y position
    Ix, Iy = [needed_arrays["identities"]["Ix"],
              needed_arrays["identities"]["Iy"]] # identity operators in space space
    x = needed_arrays["position"]["x"] # x position operator
    kx, ky = [needed_arrays["momenta"]["kx"],
              needed_arrays["momenta"]["ky"]] # momentum operators
    kx2, ky2 = [needed_arrays["momenta"]["kx2"],
                needed_arrays["momenta"]["ky2"]] # squared momentum operators
    en_h, en_l = [needed_arrays["energies_z"]["e_hl"], 
                  needed_arrays["energies_z"]["e_lh"]] # ground state energies for triangular well centered at origo
    over, expec = [needed_arrays["expectations_z"]["<phi_H|phi_L>"],
                   needed_arrays["expectations_z"]["<phi_H|kz|phi_L>"]] # <phi_H|phi_L> and <phi_H|kz|phi_L> 
    half_Lx, half_Ly = [needed_arrays["well-walls"]["half_Lx"],
                        needed_arrays["well-walls"]["half_Ly"]] # half of well width in x and y
    
    # hamiltonian terms
    h_lk = h_lk_spherical(kx2, ky2, over, en_l, en_h, gs, gk)
    h_orb =h_orb_sperical(kx, ky, x, over, expec, Az, bz, gs, gk)
    h_ze = h_z(Ix, Iy, kappa, [0, 0, bz])
    v, u = get_potential_hard_wall(X, Y, half_Lx, half_Ly, infinity)
    
    return h_lk + h_orb + h_ze + v, u

#%% Potentials
def get_potential_hard_wall(X, Y,
                            half_Lx, half_Ly,
                            infinity = 1e10): 
    """Returns tuple with infinite well potential in diagonal form (to be used for calcs) and in plottable format.
    X, Y, are the position grids in x and y.
    half_Lx, half_Ly are half lengths of the wells in x and y centered at the origin.
    infinity is the height of the well.
    """
    
    shape = X.shape
    U = zeros(shape) # initializing grid with same shape as x (or y, grid)
    U[where(X <= -half_Lx)] = infinity # raising walls outside well x
    U[where(X >= half_Lx)] = infinity # raising walls outside well in x
    
    U[where(Y <= -half_Ly)] = infinity # raising walls outside well y
    U[where(Y >= half_Ly)] = infinity # raising walls outside well y
    V = kron(diags(U.reshape(shape[0] * shape[1]), (0)), J0) # This reshaping is needed in order to obtain a (N2,N2,4) Hamiltonian to diagonalize
    return V, U

#%% Planar LK Hamiltonian (projected onto ground state of triangular well in z centered at origo)

def f(kx2, ky2, en_hh,
             g1 = 13.35, g2 = 4.25, g3 = 5.69):
    """Return f term of h_lk.
    kx, ky are momentum operators in x and y directions.
    en_hh energy of hh in infinite triangular well centered at origin.
    g1, g2 are luttinger parameters (default: Ge).
    """

    return (g1 + g2) * (kx2 + ky2) + en_hh

def g(kx2, ky2, en_lh,
             g1 = 13.35, g2 = 4.25, g3 = 5.69):
    """Return g term of h_lk.
    kx, ky are momentum operators in x and y directions.
    en_lh energy of hh in infinite triangular well centered at origin.
    g1, g2 are luttinger parameters (default: Ge).
    """
    
    return (g1 - g2) * (kx2 + ky2) + en_lh

def l(kx, ky, brah_kz_ketl,
             g1 = 13.35, g2 = 4.25, g3 = 5.69):
    """Return l term of h_lk.
    kx, ky are momentum operators in x and y directions.
    brah_kz_ketl is the expectation value <phi_hh|kz|phi_lh>.
    g1, g2, g3 are luttinger parameters (default: Ge).
    """
    
    return -2 * sqrt(3) * g3 * (kx - 1j * ky) * brah_kz_ketl

def m(kx2, ky2, kxky, brah_ketl,
             g1 = 13.35, g2 = 4.25, g3 = 5.69):
    """Return l term of h_lk.
    kx, ky are momentum operators in x and y directions.
    brah_ketl is the overlap <phi_hh|phi_lh>.
    g1, g2, g3 are luttinger parameters (default: Ge).
    """
    g = (g2 + g3) / 2
    z = (g2 - g3) / 2
    
    return sqrt(3) * (g * (kx2 - 2j * kxky - ky2)**2 - z * (kx2 + 2j * kxky - ky2)**2 ) * brah_ketl


def h_lk(needed_arrays,
        g1 = 13.35, g2 = 4.25, g3 = 5.69):
    """Returns discritezied LK Hamiltonian (according to finite element method), in sparse format.
    momenta_arrays contains all discretized momentum operators needed.
    g1, g2, g3 are Luttinger poaramteters (Default: Luttinger params for Ge).
    """
    # extracting needed arrays from momenta_arrays
    kx, kx2, ky, ky2 = [needed_arrays["momenta"]["kx"], 
                        needed_arrays["momenta"]["kx2"],
                        needed_arrays["momenta"]["ky"], 
                        needed_arrays["momenta"]["ky2"]] # squared momentum operators
    
    kxky = needed_arrays["cross-momenta"]["kxky"] # cross momentum operators
    
    en_h, en_l = [needed_arrays["energies_z"]["e_hl"], 
                  needed_arrays["energies_z"]["e_lh"]] # ground state energies for triangular well centered at origo
    brah_ketl, brah_kz_ketl = [needed_arrays["expectations_z"]["<phi_H|phi_L>"], 
                               needed_arrays["expectations_z"]["<phi_H|kz|phi_L>"]] # <phi_H|phi_L>, <phi_H|kz|phi_L>
    
    # calculating each term in h_lk
    ff = f(kx2, ky2, en_h, g1, g2, g3)
    gg = g(kx2, ky2, en_l, g1, g2, g3)
    ll = l(kx, ky, brah_kz_ketl, g1, g2, g3)
    mm = m(kx2, ky2, kxky, brah_ketl, g1, g2, g3)
        
    # magnetic field free HLK
    h_lk = block_array([[ff, None, ll, mm],
                        [None, ff, conjugate(mm), - conjugate(ll)],
                        [conjugate(ll), mm, gg, None],
                        [conjugate(mm), -ll, None, gg]], format = "coo") 
    
    return h_lk

def h_z(Ix, Iy,
        kappa = 1, B = [0,0,0]):
    """Returns zeeman hamiltonian.
    Ix, Iy, Iz are the discretized identities in space for the three dimensions.
    kappa is the magnetic g-factor.
    B is the magnetic field vector.
    """ 
    return kron(kron(Ix, Iy), kappa * (Jx * B[0] + Jy * B[1] + Jz * B[2]))

def h_tot(needed_arrays, 
          Ix, Iy,
          g1 = 13.35, g2 = 4.25, g3 = 5.69,
          kappa = 1, B = [0,0,0]):
    lk = h_lk(needed_arrays, g1, g2, g3) # lk hamiltonian
    z = h_z(Ix, Iy, kappa, B)
    return - lk + z

    