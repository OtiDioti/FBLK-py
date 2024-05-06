from numpy import array, zeros
from numpy import sum as Sum
from numpy.linalg import norm
from stqdm import stqdm # tqdm-like progress bar
from HamiltonianBulkProjection import h_tot_v, get_ks
from scipy.sparse.linalg import eigsh
from streamlit import spinner
#%% Projection solvers
def projection_solver_static(A, 
                            Lx, Ly, Lz,
                            g1, g2, g3,
                            kappa, Bx, By, Bz, 
                            nr_of_soln, dim, possible_statess):
    """Returns a 3-element tuple with arrays for eigenvalues and both the orbital and
    spin components of the eigenvectors obtained for static magnetic field and no time dep.
    A is magnetic vector potential.
    L_i are well-depths in the three dimensions.
    g_i are the Luttinger parameters.
    kappa is magnetic g-factor.
    Bi_vals are the array for the values of the magnetic field in the three dimesions.
    nr_of_soln is the number of eigenstates to solve for.
    dim is orbital dimensionality of the problem.
    possible_statess is array with all permutations of basis states.
    """
    with spinner('Diagonalizing the problem'):
        # Constructing Hamiltonian
        kikj, k2 = get_ks(possible_statess, 
                          A, 
                          Lx, Ly, Lz,
                          dim)
        H = h_tot_v(k2, kikj,
                    dim,
                    g1 = g1, g2 = g2, g3 = g3,
                    kappa = kappa, B = [Bx,By,Bz])
        
        # Diagonalizing hamiltonian
        eigvals, eigvects = eigsh(H, k = nr_of_soln, which = "SM") # for CPU
        eigvects = eigvects / norm(eigvects, axis = 0)[None, :] # normalizing eigenvectors
        tmpzip = zip(eigvects.T, eigvals) # zipping eigenvectors with eigenvalues
        sort = sorted(tmpzip, key=lambda x: x[1]) # sorting vectors according to their eigenvalue in increasing order
        eigvects = array([sort[i][0] for i in range(len(sort))]) # extracting sorted eigenvectors
        eigvals = array([sort[i][1] for i in range(len(sort))]) # extracting sorted eigenvalues
        
        # Tracing out spin components to obtain plottable eigenvectors
        spin_component = zeros((nr_of_soln, dim, 4), complex) # here we will store the eigenvectors spin components
        traced_out_psi = zeros((nr_of_soln, dim), complex) # here we will store eigenvectors with spin component traced out
        for n in range(nr_of_soln): # iterating through eigenvectors
            for j in range(dim): # iterating through basis states
                spin_component[n, j, :] = eigvects[n][j*4:j*4 + 4] # for each basis state we append the spin components (eg: the first 4 element of tmp correspond to |1,1,1,3/2>, |1,1,1,-3/2>, |1,1,1,1/2>, |1,1,1,-1/2>)
            coeff =  Sum(spin_component[n], axis = 1) # tracing out spin components
            traced_out_psi[n] = coeff / norm(coeff) 
    return eigvals, traced_out_psi, spin_component

def projection_solver_var_b(A, 
                            Lx, Ly, Lz,
                            g1, g2, g3,
                            kappa, Bx_vals, By_vals, Bz_vals, 
                            points_b, nr_of_soln, dim, possible_statess):
    """Returns a 3-element tuple with arrays for eigenvalues and both the orbital and
    spin components of the eigenvectors obtained for variable magnetic field.
    A is magnetic vector potential.
    L_i are well-depths in the three dimensions.
    g_i are the Luttinger parameters.
    kappa is magnetic g-factor.
    Bi_vals are the array for the values of the magnetic field in the three dimesions.
    points_b is length of Bi_vals.
    nr_of_soln is the number of eigenstates to solve for.
    dim is orbital dimensionality of the problem.
    possible_statess is array with all permutations of basis states.
    """
    eigenvalues = zeros((points_b, nr_of_soln)) # here we'll store the eigenvectors for all b-values
    eigenvectors_orbi = zeros((points_b, nr_of_soln, dim), complex) # here we'll store the eigenvectors orbital components for all b-values
    eigenvectors_spin = zeros((points_b, nr_of_soln, dim, 4), complex) # here we'll store the eigenvectors spin components for all b-values
    for i in stqdm(range(points_b), desc = r"Iterating through $B$-values"):
        # Constructing Hamiltonian (JIT)
        kikj, k2 = get_ks(possible_statess, 
                          A[i], 
                          Lx, Ly, Lz,
                          dim)
        H = h_tot_v(k2, kikj,
                    dim,
                    g1 = g1, g2 = g2, g3 = g3,
                    kappa = kappa, B = [Bx_vals[i], By_vals[i], Bz_vals[i]])
        
        # Diagonalizing hamiltonian
        eigvals, eigvects = eigsh(H, k = nr_of_soln, which = "SM") # for CPU
        eigvects = eigvects / norm(eigvects, axis = 0)[None, :] # normalizing eigenvectors
        tmpzip = zip(eigvects.T, eigvals) # zipping eigenvectors with eigenvalues
        sort = sorted(tmpzip, key=lambda x: x[1]) # sorting vectors according to their eigenvalue in increasing order
        eigvects = array([sort[j][0] for j in range(len(sort))]) # extracting sorted eigenvectors
        eigvals = array([sort[j][1] for j in range(len(sort))]) # extracting sorted eigenvalues
        
        # Tracing out spin components to obtain plottable eigenvectors
        spin_component = zeros((nr_of_soln, dim, 4), complex) # here we will store the eigenvectors spin components
        traced_out_psi = zeros((nr_of_soln, dim), complex) # here we will store eigenvectors with spin component traced out
        for n in range(nr_of_soln): # iterating through eigenvectors
            for j in range(dim): # iterating through basis states
                spin_component[n, j, :] = eigvects[n][j*4:j*4 + 4] # for each basis state we append the spin components (eg: the first 4 element of tmp correspond to |1,1,1,3/2>, |1,1,1,-3/2>, |1,1,1,1/2>, |1,1,1,-1/2>)
            coeff =  Sum(spin_component[n], axis = 1) # tracing out spin components
            traced_out_psi[n] = coeff / norm(coeff) 
        eigenvalues[i] = eigvals # appending eigenvalues of all solutions for this b-value
        eigenvectors_orbi[i] = traced_out_psi # appending orbital component of all solutions for this b-value
        eigenvectors_spin[i] = spin_component # appending spin component of all solutions for this b-value
    return eigenvalues, eigenvectors_orbi, eigenvectors_spin

def projection_solver_var_t(A, 
                            Lx, Ly, Lz,
                            g1, g2, g3,
                            kappa, Bx, By, Bz, 
                            points_t, nr_of_soln, dim, possible_statess):
    """Returns a 3-element tuple with arrays for eigenvalues and both the orbital and
    spin components of the eigenvectors obtained for static magnetic field, but with time dep.
    A is magnetic vector potential.
    L_i are well-depths in the three dimensions.
    g_i are the Luttinger parameters.
    kappa is magnetic g-factor.
    Bi are the array for the values of the magnetic field in the three dimesions.
    points_b is length of Bi_vals.
    nr_of_soln is the number of eigenstates to solve for.
    dim is orbital dimensionality of the problem.
    possible_statess is array with all permutations of basis states.
    """
    eigenvalues = zeros((points_t, nr_of_soln)) # here we'll store the eigenvectors for all b-values
    eigenvectors_orbi = zeros((points_t, nr_of_soln, dim), complex) # here we'll store the eigenvectors orbital components for all b-values
    eigenvectors_spin = zeros((points_t, nr_of_soln, dim, 4), complex) # here we'll store the eigenvectors spin components for all b-values
    for i in stqdm(range(points_t), desc = r"Iterating through time"):
        # Constructing Hamiltonian (JIT)
        kikj, k2 = get_ks(possible_statess, 
                          A[i], 
                          Lx, Ly, Lz,
                          dim)
        H = h_tot_v(k2, kikj,
                    dim,
                    g1 = g1, g2 = g2, g3 = g3,
                    kappa = kappa, B = [Bx[i],By[i],Bz[i]])
        
        # Diagonalizing hamiltonian
        eigvals, eigvects = eigsh(H, k = nr_of_soln, which = "SM") # for CPU
        eigvects = eigvects / norm(eigvects, axis = 0)[None, :] # normalizing eigenvectors
        tmpzip = zip(eigvects.T, eigvals) # zipping eigenvectors with eigenvalues
        sort = sorted(tmpzip, key=lambda x: x[1]) # sorting vectors according to their eigenvalue in increasing order
        eigvects = array([sort[j][0] for j in range(len(sort))]) # extracting sorted eigenvectors
        eigvals = array([sort[j][1] for j in range(len(sort))]) # extracting sorted eigenvalues
        
        # Tracing out spin components to obtain plottable eigenvectors
        spin_component = zeros((nr_of_soln, dim, 4), complex) # here we will store the eigenvectors spin components
        traced_out_psi = zeros((nr_of_soln, dim), complex) # here we will store eigenvectors with spin component traced out
        for n in range(nr_of_soln): # iterating through eigenvectors
            for j in range(dim): # iterating through basis states
                spin_component[n, j, :] = eigvects[n][j*4:j*4 + 4] # for each basis state we append the spin components (eg: the first 4 element of tmp correspond to |1,1,1,3/2>, |1,1,1,-3/2>, |1,1,1,1/2>, |1,1,1,-1/2>)
            coeff =  Sum(spin_component[n], axis = 1) # tracing out spin components
            traced_out_psi[n] = coeff / norm(coeff) 
        eigenvalues[i] = eigvals # appending eigenvalues of all solutions for this A-value
        eigenvectors_orbi[i] = traced_out_psi # appending orbital component of all solutions for this A-value
        eigenvectors_spin[i] = spin_component # appending spin component of all solutions for this A-value
    return eigenvalues, eigenvectors_orbi, eigenvectors_spin