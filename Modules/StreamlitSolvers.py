from numpy import array, zeros, conjugate
from numpy import sum as Sum
from numpy.linalg import norm
from stqdm import stqdm # tqdm-like progress bar
from HamiltonianBulkProjection import h_tot_v, get_ks
from scipy.sparse.linalg import eigsh, expm
from streamlit import spinner, session_state
from GetInputProjection_Vectorized import get_input
#%% Repeated functions
def spin_tracer(eigvects, 
                nr_of_soln, dim):
    """Returns 2-element tuple containing arrays for the orbital and spin components
    of eigenfunctions respectively.
    eigvects is array containing all eigenvectors found.
    nr_of_soln is the number of eigenvectors found.
    dim is orbital dimensionality.
    """
    
    spin_component = zeros((nr_of_soln, dim, 4), complex) # here we will store the eigenvectors spin components
    traced_out_psi = zeros((nr_of_soln, dim), complex) # here we will store eigenvectors with spin component traced out
    for n in range(nr_of_soln): # iterating through eigenvectors
        for j in range(dim): # iterating through basis states
            spin_component[n, j, :] = eigvects[n][j*4:j*4 + 4] # for each basis state we append the spin components (eg: the first 4 element of tmp correspond to |1,1,1,3/2>, |1,1,1,-3/2>, |1,1,1,1/2>, |1,1,1,-1/2>)
        coeff =  Sum(spin_component[n], axis = 1) # tracing out spin components
        traced_out_psi[n] = coeff / norm(coeff) 
    return traced_out_psi, spin_component
#%% Projection solvers
def projection_solver_static(Ac, 
                             Lx, Ly, Lz,
                             g1, g2, g3,
                             kappa, Bx, By, Bz, 
                             nr_of_soln, dim, possible_statess):
    """Returns a 3-element tuple with arrays for eigenvalues and both the orbital and
    spin components of the eigenvectors obtained for static magnetic field and no time dep.
    Ac is the vector potential coefficients array to be used in minimal coup. substitution. This is of the form 
                           A[i] = [Ax_x, Ax_y, Ax_z, Ax_c, 
                                   Ay_x, Ay_y, Ay_z, Ay_c, 
                                   Az_x, Az_y, Az_z, Ax_c]
    and Ai_j is the ith component of A multipling the j operator in that component.
    L_i are well-depths in the three dimensions.
    g_i are the Luttinger parameters.
    kappa is magnetic g-factor.
    Bi_vals are the array for the values of the magnetic field in the three dimesions.
    nr_of_soln is the number of eigenstates to solve for.
    dim is orbital dimensionality of the problem.
    possible_statess is array with all permutations of basis states.
    """
    with spinner("Computing expectation values"):
        inputt = get_input(possible_statess, Ac, Lx, Ly, Lz, dim)
        kikj, k2 = get_ks(inputt)
    with spinner('Diagonalizing the problem'):
        # Constructing Hamiltonian
        t = 0 # "dummy time" index
        H = h_tot_v(k2[0][t], k2[1][t], k2[2][t], 
                    kikj[0][t], kikj[1][t], kikj[2][t], kikj[3][t], kikj[4][t], kikj[5][t],
                    dim,
                    g1, g2, g3,
                    kappa, B = [Bx[t],By[t],Bz[t]])
        
        # Diagonalizing hamiltonian
        eigvals, eigvects = eigsh(H, k = nr_of_soln, which = "SM") # for CPU
        eigvects = eigvects / norm(eigvects, axis = 0)[None, :] # normalizing eigenvectors
        tmpzip = zip(eigvects.T, eigvals) # zipping eigenvectors with eigenvalues
        sort = sorted(tmpzip, key=lambda x: x[1]) # sorting vectors according to their eigenvalue in increasing order
        eigvects = array([sort[i][0] for i in range(len(sort))]) # extracting sorted eigenvectors
        eigvals = array([sort[i][1] for i in range(len(sort))]) # extracting sorted eigenvalues
        
        traced_out_psi, spin_component = spin_tracer(eigvects, nr_of_soln, dim)
    return eigvals, traced_out_psi, spin_component

def projection_solver_var_b(Ac, 
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
    with spinner("Calculating expectation values"):
        inputt = get_input(possible_statess, Ac, Lx, Ly, Lz, dim)
        kikj, k2 = get_ks(inputt)
        
        eigenvalues = zeros((points_b, nr_of_soln)) # here we'll store the eigenvectors for all b-values
        eigenvectors_orbi = zeros((points_b, nr_of_soln, dim), complex) # here we'll store the eigenvectors orbital components for all b-values
        eigenvectors_spin = zeros((points_b, nr_of_soln, dim, 4), complex) # here we'll store the eigenvectors spin components for all b-values
    for t in stqdm(range(points_b), desc = r"Iterating through $B$-values"):
        # Constructing Hamiltonian
        
        H = h_tot_v(k2[0][t], k2[1][t], k2[2][t], 
                    kikj[0][t], kikj[1][t], kikj[2][t], kikj[3][t], kikj[4][t], kikj[5][t],
                    dim,
                    g1, g2, g3,
                    kappa, B = [Bx_vals[t],By_vals[t],Bz_vals[t]])
        
        # Diagonalizing hamiltonian
        eigvals, eigvects = eigsh(H, k = nr_of_soln, which = "SM") # for CPU
        eigvects = eigvects / norm(eigvects, axis = 0)[None, :] # normalizing eigenvectors
        tmpzip = zip(eigvects.T, eigvals) # zipping eigenvectors with eigenvalues
        sort = sorted(tmpzip, key=lambda x: x[1]) # sorting vectors according to their eigenvalue in increasing order
        eigvects = array([sort[j][0] for j in range(len(sort))]) # extracting sorted eigenvectors
        eigvals = array([sort[j][1] for j in range(len(sort))]) # extracting sorted eigenvalues
        eigenvalues[t] = eigvals # appending eigenvalues of all solutions for this b-value
        eigenvectors_orbi[t], eigenvectors_spin[t] = spin_tracer(eigvects, nr_of_soln, dim)  # appending orbital and spin components of all solutions for this b-value
        
    return eigenvalues, eigenvectors_orbi, eigenvectors_spin

def projection_solver_var_t(Ac, 
                            Lx, Ly, Lz,
                            g1, g2, g3,
                            kappa, Bx, By, Bz, 
                            points_t, dim, dt, possible_statess):
    """Returns a 3-element tuple with arrays for eigenvalues and both the orbital and
    spin components of the eigenvectors obtained for static magnetic field, but with time dep.
    A is magnetic vector potential.
    L_i are well-depths in the three dimensions.
    g_i are the Luttinger parameters.
    kappa is magnetic g-factor.
    Bi are the array for the values of the magnetic field in the three dimesions.
    points_b is length of Bi_vals.
    dim is orbital dimensionality of the problem.
    dt is time discretization number.
    possible_statess is array with all permutations of basis states.
    """    
    with spinner("Calculating expectation values"):
        inputt = get_input(possible_statess, Ac, Lx, Ly, Lz, dim)
        kikj, k2 = get_ks(inputt)
    
    with spinner('Obtaining ground state'):
       t = 0
       H = h_tot_v(k2[0][t], k2[1][t], k2[2][t], 
                   kikj[0][t], kikj[1][t], kikj[2][t], kikj[3][t], kikj[4][t], kikj[5][t],
                   dim,
                   g1, g2, g3,
                   kappa, B = [Bx[t],By[t],Bz[t]])
       eigvals, eigvects = eigsh(H, k = 5, which = "SM") # for CPU            
       eigvects = eigvects / norm(eigvects, axis = 0)[None, :] # normalizing eigenvectors
       tmpzip = zip(eigvects.T, eigvals) # zipping eigenvectors with eigenvalues
       sort = sorted(tmpzip, key=lambda x: x[1]) # sorting vectors according to their eigenvalue in increasing order
       eigvects = array([sort[i][0] for i in range(len(sort))]) # extracting sorted eigenvectors
       eigvals = array([sort[i][1] for i in range(len(sort))]) # extracting sorted eigenvalues
       session_state["init_state"] = eigvals[0], eigvects[0] # obtaining ground state
    
    eigenvalues = zeros(points_t + 1) # here we'll store the eigenvectors for all t-values
    psi_t = zeros((points_t + 1, dim * 4), complex) # storing total eigenvectors
    eigenvalues[0], psi_t[0] = session_state["init_state"] # storing initial energy and state value
    
    eigenvectors_orbi = zeros((points_t + 1, dim), complex) # here we'll store the eigenvectors orbital components for all t-values
    eigenvectors_spin = zeros((points_t + 1, dim, 4), complex) # here we'll store the eigenvectors spin components for all t-values
    eigenvectors_orbi[0], eigenvectors_spin[0] = spin_tracer([psi_t[0]], 1, dim)
    for i in stqdm(range(points_t), desc = "Integrating through time"):
        
        H = h_tot_v(k2[0][i], k2[1][i], k2[2][i], 
                    kikj[0][i], kikj[1][i], kikj[2][i], kikj[3][i], kikj[4][i], kikj[5][i],
                    dim,
                    g1, g2, g3,
                    kappa, B = [Bx[t],By[t],Bz[t]]) # Constructing Hamiltonian 
        
        psi_t[i + 1] = expm(-1j * dt * H) @ psi_t[i] # evolving state
        psi_t[i + 1] = psi_t[i + 1] / norm(psi_t[i + 1]) # normalizing state
        
        eigenvalues[i + 1] = Sum(conjugate(psi_t) * (H @ psi_t[i + 1])) # energy expectation value
        eigenvectors_orbi[i + 1], eigenvectors_spin[i + 1] = spin_tracer([psi_t[i + 1]], 1, dim)
            
    return eigenvalues, eigenvectors_orbi, eigenvectors_spin