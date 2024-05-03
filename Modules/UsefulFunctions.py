import numpy as np
import scipy.sparse as sparse
#%% Spin-3/2 matrices
J0 = sparse.coo_matrix(np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]]))

Jx = sparse.coo_matrix(0.5 * np.array([[0, 0, np.sqrt(3), 0],
                                       [0, 0, 0, np.sqrt(3)],
                                       [np.sqrt(3), 0, 0, 2],
                                       [0, np.sqrt(3), 2, 0]]))
Jy = sparse.coo_matrix(0.5 * np.array([[0, 0, -1j * np.sqrt(3), 0],
                                       [0, 0, 0, 1j * np.sqrt(3)],
                                       [1j * np.sqrt(3), 0, 0, -2j],
                                       [0, -1j * np.sqrt(3), 2j, 0]]))

Jz = sparse.coo_matrix(0.5 * np.array([[3, 0, 0, 0],
                                       [0, -3, 0, 0],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, -1]]))
Jv = np.array([Jx, Jy, Jz])
#%% Finite diff. method differential operators
def dd_i(dim, dx, bc = 0):
    """Returns sparse differential operator d/dx according to finite element method.
    dim is discretizastion number along the selected direction.
    dx is spacing between nodes of grid.
    bc is the selected boundary condition (bc=0 -> Dirichlect; bc=1 -> Neumann).
    NOTE: BC have no implementation yet.
    """
    main = np.zeros(dim) # elements along main diagonal
    offd = np.ones(dim) # elements along shifted diagonals
    if bc == 0: # if Dirichlect
        k_i = sparse.spdiags(np.array([-offd, main, offd], complex), np.array([-1,0,1]), dim, dim) # first order differential operator
        return k_i / (2*dx)
    else:
        print("Other boundary conditions not yet supported")

def dd2_i(dim, dx, bc = 0):
    """Returns sparse differential operator d^2/dx^2 according to finite element method.
    dim is discretizastion number along the selected direction.
    dx is spacing between nodes of grid.
    bc is the selected boundary condition (bc=0 -> Dirichlect; bc=1 -> Neumann).
    NOTE: BC have no implementation yet.
    """
    main = -2 * np.ones(dim) # elements along main diagonal
    offd = np.ones(dim) # elements along shifted diagonals
    if bc == 0: # if Dirichlect
        k2_i = sparse.spdiags(np.array([offd, main, offd], complex), np.array([-1,0,1]), dim, dim) # second order differential operator
        return k2_i / dx**2
    else:
        print("Other boundary conditions not yet supported")

def k_i(dim, dx, bc = 0):
    """Returns sparse momentum operator k = p / h, according to finite element method.
    dim is discretizastion number along the selected direction.
    dx is spacing between nodes of grid.
    bc is the selected boundary condition (bc=0 -> Dirichlect; bc=1 -> Neumann).
    NOTE: BC have no implementation yet.
    """
    return -1j * dd_i(dim, dx, bc)

def k2_i(dim, dx, bc = 0):
    """Returns sparse momentum operator k = p / h, according to finite element method.
    dim is discretizastion number along the selected direction.
    dx is spacing between nodes of grid.
    bc is the selected boundary condition (bc=0 -> Dirichlect; bc=1 -> Neumann).
    NOTE: BC have no implementation yet.
    """
    return - dd2_i(dim, dx, bc)

#%% Useful Functions
def anti_comm(a, b):
    """Returns anti commutator between two arrays.
    """
    return a @ b + b @ a
#%% Useful functions for projection method
def possible_states(nx_max, ny_max, nz_max):
    """Returns all possible basis states in an ordered fashion.
    ni_max is the highest state expanded into along direction i.
    ground state is defined as n = 1"""
 
    # obtaining all possible permutations of basis states
    N = nx_max * ny_max * nz_max # all possible permutations of basis states
    possible_vectors = np.zeros((N, 3)) # initializing empty array
    idx = 0 # dummy index
    for x in range(1, nx_max + 1):
        for y in range(1, ny_max + 1):
            for z in range(1, nz_max + 1):
                possible_vectors[idx, :] = np.array([x, y, z])
                idx += 1
    return possible_vectors

def converter(state, possible_states):
    """Returns index of basis state |s, nx, ny, nz> -> |i>.
    state is a tuple of the form (s, nx, ny, nz).
    possible_states is an array generated via PossibleStates().
    """
    return np.where((possible_states == state).all(axis = 1))[0][0]
    
    
        
