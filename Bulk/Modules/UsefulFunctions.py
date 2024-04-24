import numpy as np
import scipy.sparse as sparse
#%% Spin-3/2 matrices
J0 = np.eye(4)

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
    
    
        
