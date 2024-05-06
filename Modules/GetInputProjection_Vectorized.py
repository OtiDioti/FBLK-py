from numpy import array, pi, repeat, transpose
from streamlit import write
#%%
def get_input(possible_states, 
              Ac, 
              Lx, Ly, Lz,
              dim):
    """Returns input dictionary to be used in the following calculations.
    Ac is the vector potential coefficients array to be used in minimal coup. substitution. This is of the form 
                           A[i] = [Ax_x, Ax_y, Ax_z, Ax_c, 
                                   Ay_x, Ay_y, Ay_z, Ay_c, 
                                   Az_x, Az_y, Az_z, Ax_c]
    and Ai_j is the ith component of A multipling the j operator in that component.
    L_i are the well widths in the three directions.
    dim is the orbital dimensionality of the system.
    """
    
    nx_bra, ny_bra, nz_bra = possible_states[:,0][None,:], possible_states[:,1][None,:], possible_states[:,2][None,:] # extracting orbital states for each direction
    nx_ket, ny_ket, nz_ket = possible_states[:,0][:,None], possible_states[:,1][:,None], possible_states[:,2][:,None] # extracting orbital states for each direction

    dx = (nx_bra==nx_ket)[None,:,:] # d_{nx=mx} (adding time axis at the front to match shape of Aij)
    dy = (ny_bra==ny_ket)[None,:,:] # d_{ny=my} (adding time axis at the front to match shape of Aij)
    dz = (nz_bra==nz_ket)[None,:,:] # d_{nz=mz} (adding time axis at the front to match shape of Aij)
    dxdy = dx*dy # d_{nx=mx} d_{ny=my}
    dxdz = dx*dz # d_{nx=mx} d_{nz=mz}
    dydz = dy*dz # d_{ny=my} d_{nz=mz}
    dxdydz = dxdy*dz # d_{nx=mx} d_{ny=my} d_{nz=mz} 

    x, y, z, x2, y2, z2 = get_pos_operators(possible_states, Lx, Ly, Lz) # position operators
    A = vect_pot_A(possible_states, Ac, x, y, z, dim) # magnetic vector potential

    kx, ky, kz = get_k(possible_states, Lx, Ly, Lz) # <n_i|ki|m_i> (adding time axis at the front to match shape of Aij)
    xkx, kxx, yky, kyy, zkz, kzz = get_iki(possible_states) # expecation values <n_i|i ki|m_i> and <n_i|ki i|m_i>
    return {"n_bra" : [nx_bra, ny_bra, nz_bra], # all possible permutations of all possible bra states
            "n_ket" : [nx_ket, ny_ket, nz_ket], # all possible permutations of all possible ket states
            "well widths":[Lx, Ly, Lz], # well widths in the three different directions
            "linear position op":[x,y,z], # expect. val. of all position operators for all permutations of all basis states
            "squared position op":[x2,y2,z2], # expect. val. of all squared position operators for all permutations of all basis states
            "linear momentum op":[kx, ky, kz],  # expect. val. of all momentum operators for all permutations of all basis states
            "i ki and ki i op":[xkx, kxx, yky, kyy, zkz, kzz],  # expecation values <n_i|i ki|m_i> and <n_i|ki i|m_i> for all permutations of all basis states
            "vector potential coefficients":  Ac, # coefficients of pos. operators within vector potential
            "vector potential" : A, # expectation value for the vector potential for all permutations of all basis states
            "dirac deltas": [dx, dy, dz, dxdy, dxdz, dydz, dxdydz] # dirac delta functions for all permutations of all basis states
            }

#%% defining momentum and position operators for infinite well orbital states
def get_pos_operators(possible_states,
                      Lx, Ly, Lz,
                      soft = 1e-15):
    """Returns expectation values of all position operators (linear and squared), 
    for all permutations of basis states.
    possible_statess is array of all possible permutation of basis states.
    L_i are the well widths in the three directions.
    """
    
    pos_op = lambda n_bra, n_ket, L: 4 * (-1 + (-1)**(n_ket + n_bra)) * n_ket * n_bra * L / ((n_ket**2 - n_bra**2 + soft)**2 * pi**2)    
    pos_op2 = lambda n_bra, n_ket, L: 4 * (1 + (-1)**(n_ket + n_bra)) * n_ket * n_bra * L**2 / ((n_ket**2 - n_bra**2 + soft)**2 * pi**2) * (n_bra!=n_ket) + 1/12 * L**2 * (1 - 6/(n_ket**2 * pi**2)) * (n_bra==n_ket) 
    
    nx_bra, ny_bra, nz_bra = possible_states[:,0][None,:], possible_states[:,1][None,:], possible_states[:,2][None,:] # extracting orbital states for each direction
    nx_ket, ny_ket, nz_ket = possible_states[:,0][:,None], possible_states[:,1][:,None], possible_states[:,2][:,None] # extracting orbital states for each direction

    x, y, z, x2, y2, z2 = [pos_op(nx_bra, nx_ket, Lx)[None,:,:],
                           pos_op(ny_bra, ny_ket, Ly)[None,:,:],
                           pos_op(nz_bra, nz_ket, Lz)[None,:,:],
                           pos_op2(nx_bra, nx_ket, Lx)[None,:,:],
                           pos_op2(ny_bra, ny_ket, Ly)[None,:,:],
                           pos_op2(nz_bra, nz_ket, Lz)[None,:,:]] # position expectation values (also adding time axis at the fron to match shape of A)
    
    return x,y,z,x2,y2,z2
    

def vect_pot_A(possible_states, 
               Ac,
               x, y, z,
               dim,
               soft = 1e-15):
    """Returns vector potential (first axis contains information about time/variable mag.field/...
    possible_statess is array of all possible permutation of basis states.
    Ac is vector potential coefficients. This is of the form 
                           A[i] = [Ax_x, Ax_y, Ax_z, Ax_c, 
                                Ay_x, Ay_y, Ay_z, Ay_c, 
                                Az_x, Az_y, Az_z, Ax_c]
    where Ai_j is the ith component of A multipling the j operator in that component.
    x,y,z are expectation values for all possible permutations of basis states.
    dim is orbital dimensionality.
    """
    
    Axx, Axy, Axz, Axc = [Ac[:,0,0] * x, Ac[:,0,1] * y, Ac[:,0,2] * z, Ac[:,0,3]]
    Ayx, Ayy, Ayz, Ayc = [Ac[:,1,0] * x, Ac[:,1,1] * y, Ac[:,1,2] * z, Ac[:,1,3]]
    Azx, Azy, Azz, Azc = [Ac[:,2,0] * x, Ac[:,2,1] * y, Ac[:,2,2] * z, Ac[:,2,3]]
    repeater = lambda a: repeat(repeat(a, dim, axis=1), dim, axis=2)

    Axc, Ayc, Azc = [repeater(Axc), 
                     repeater(Ayc), 
                     repeater(Azc)]

    A = array([[Axx, Axy, Axz, Axc], 
               [Ayx, Ayy, Ayz, Ayc],
               [Azx, Azy, Azz, Azc]])
    A = transpose(A, axes = (2,0,1,3,4))
    return A

def get_k(possible_states, 
          Lx, Ly, Lz,
          soft = 1e-15):
    """Returns expectation value of momentum operators in all three directions for 
    all permutations of basis states.
    possible_statess is array of all possible permutation of basis states.
    L_i are the well-depths in the three directions.
    """
    k = lambda n_bra, n_ket, L: - 2j * (-1 + (-1)**(n_ket + n_bra)) * n_ket * n_bra / (L * (n_ket**2 - n_bra**2 + soft))
    
    nx_bra, ny_bra, nz_bra = possible_states[:,0][None,:], possible_states[:,1][None,:], possible_states[:,2][None,:] # extracting orbital states for each direction
    nx_ket, ny_ket, nz_ket = possible_states[:,0][:,None], possible_states[:,1][:,None], possible_states[:,2][:,None] # extracting orbital states for each direction
    
    kx = k(nx_bra, nx_ket, Lx)[None,:,:] # <n_x|kx|m_x> (adding time axis at the beginning)
    ky = k(ny_bra, ny_ket, Ly)[None,:,:] # <n_y|ky|m_y> (adding time axis at the beginning)
    kz = k(nz_bra, nz_ket, Lz)[None,:,:] # <n_z|kz|m_z> (adding time axis at the beginning)
    return kx, ky, kz
   

def get_iki(possible_states, soft = 1e-15):
    """Returns the expecation values <n_i|i ki|m_i> and <n_i|ki i|m_i>.
    possible_statess is array of all possible permutation of basis states.
    """
    nx_bra, ny_bra, nz_bra = possible_states[:,0][None,:], possible_states[:,1][None,:], possible_states[:,2][None,:] # extracting orbital states for each direction
    nx_ket, ny_ket, nz_ket = possible_states[:,0][:,None], possible_states[:,1][:,None], possible_states[:,2][:,None] # extracting orbital states for each direction
    
    iki = lambda n_bra, n_ket: -1j * (1 + (-1)**(n_ket + n_bra)) * n_ket * n_bra / (n_ket**2 - n_bra**2 + soft) * (n_bra!=n_ket) + 0.5j * (n_bra==n_ket)
    kii = lambda n_bra, n_ket: iki(n_bra, n_ket) * (n_bra!=n_ket)  - 0.5j * (n_bra==n_ket) 
    
    xkx = iki(nx_bra, nx_ket) # <n_x|x kx|m_x>
    kxx = kii(nx_bra, nx_ket) # <n_x|kx x|m_x>
    yky = iki(ny_bra, ny_ket) # <n_y|y k|m_y> 
    kyy = kii(ny_bra, ny_ket) # <n_y|ky y|m_y>
    zkz = iki(nz_bra, nz_ket) # <n_z|z kz|m_z>
    kzz = kii(nz_bra, nz_ket) # <n_z|kz z|m_z>
    return xkx[None,:,:], kxx[None,:,:], yky[None,:,:], kyy[None,:,:], zkz[None,:,:], kzz[None,:,:]
