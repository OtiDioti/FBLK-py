# my imports
import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__)) # current directory
sys.path.append(dir_path+'/Modules') # appending modules folder
from HamiltonianBulk import get_potential_wire, preparing_terms, h_tot

# plotting imports 
import plotly.graph_objects as go

# other imports
import streamlit as st 
from numpy import array, linspace, zeros, transpose, mgrid
from numpy import round as Round
from numpy import sum as Sum
from numpy import abs as Abs
from numpy.linalg import norm
from scipy.sparse.linalg import eigsh
from stqdm import stqdm # tqdm-like progress bar
#%% page settings
st.set_page_config(page_title = 'FBLK', 
                   layout = 'centered', 
                   page_icon = ':atom_symbol:',
                   menu_items={
                   'Get Help': 'https://github.com/OtiDioti/FBLK-py/issues',
                   'Report a bug': "https://github.com/OtiDioti/FBLK-py/issues",
                   'About': "**The app is work in progress: any comment/suggestion/request is welcome!**"},
                   initial_sidebar_state="collapsed")
st.title("Four band Luttinger Kohn Hamiltonian")

st.divider()
#%% setting page layout
row1 = st.columns(3) # first row has 3 colums
row2 = st.columns(2) # second row has 2 columns
row3 = st.columns(1) # third row has 1 column
#%% Determining the problem size
r1_1 = row1[0].container()
with r1_1:
    st.write("## Settings")
    ### Pick discretization number in three dimensions
    dimx = st.number_input("nr. of x-steps.", 
                           min_value=30, max_value=None, 
                           value = 30, step=None, format=None, 
                           help="""Select the number of steps to discretize x dimension
                           between x = -1 and x = 1""",
                           label_visibility="visible") # discretization number in x-direction
    
    dimy = st.number_input("nr. of y-steps.", 
                           min_value=30, max_value=None, 
                           value = 30 , step=None, format=None, 
                           help="""Select the number of steps to discretize y dimension
                           between y = -1 and y = 1""",
                           label_visibility="visible") # discretization number in y-direction
    
    dimz = st.number_input("nr. of z-steps.", 
                           min_value=30, max_value=None, 
                           value = 30 , step=None, format=None, 
                           help="""Select the number of steps to discretize z dimension
                           between z = -1 and z = 1""",
                           label_visibility="visible") # discretization number in z-direction
    nr_of_soln = st.number_input("nr. of eigensolutions.", 
                           min_value=1, max_value=None, 
                           value = 3 , step=None, format=None, 
                           help="""Select the number of solutions with lowest eigen energy to solve for.""",
                           label_visibility="visible") # discretization number in z-direction
#%% Selecting luttinger parameters
r1_2 = row1[1].container()
with r1_2:
    st.write("## $\gamma$-parameters")
    ### Pick discretization number in three dimensions
    g1 = st.number_input(r"$\gamma_1$", 
                           min_value=0.0, max_value=None, 
                           value = 13.35, step=None, format=None, 
                           help=r"""Select the value for the luttinger parameter $\gamma_1$. Default: Ge.""",
                           label_visibility="visible") # discretization number in x-direction
    
    g2 = st.number_input(r"$\gamma_2$", 
                           min_value=0.0, max_value=None, 
                           value = 4.25 , step=None, format=None, 
                           help=r"""Select the value for the luttinger parameter $\gamma_2$. Default: Ge.""",
                           label_visibility="visible") # discretization number in y-direction
    
    g3 = st.number_input(r"$\gamma_3$", 
                           min_value=0.0, max_value=None, 
                           value = 5.69 , step=None, format=None, 
                           help=r"""Select the value for the luttinger parameter $\gamma_3$. Default: Ge.""",
                           label_visibility="visible") # discretization number in z-direction
#%% Settign magnetic field
r1_3 = row1[2].container()
with r1_3:
    st.write("## $B$-field")
    variable_b_toggle = st.toggle("Variable $B$", value=False, help="Toggle to study variable magnetic field.")
    
    if variable_b_toggle:
        ### Pick magnetic field interval
        Bx_min, Bx_max = st.slider(r"$B_x$ range.", 0.0, 10.0, (0.0, 0.0),
                                   help ="Use this slider to select between which values to vary the x-component of the magnetic field." ) # obtaining range of values for Bx
        
        By_min, By_max = st.slider(r"$B_y$ range.", 0.0, 10.0, (0.0, 0.0),
                                   help ="Use this slider to select between which values to vary the y-component of the magnetic field." ) # obtaining range of values for By
                               
        Bz_min, Bz_max = st.slider(r"$B_z$ range.", 0.0, 10.0, (0.0, 1.0),
                                   help ="Use this slider to select between which values to vary the z-component of the magnetic field." ) # obtaining range of values for Bz
        
        points = st.number_input("nr. of of B-values.", 
                               min_value=2, max_value=None, 
                               value = 10, step=None, format=None, 
                               help= r"""Select the number of steps between $B_0$ and $B_f$""",
                               label_visibility="visible") # discretization number for B         
        
        kappa = st.number_input(r"$\kappa$", 
                               min_value=0.0, max_value=None, 
                               value = 1.0 , step=None, format=None, 
                               help=r"""Select the value for the magnetic g-factor of the system.""",
                               label_visibility="visible") # value of magnetic g-factor
        Bx_vals = linspace(Bx_min, Bx_max, points)
        By_vals = linspace(By_min, By_max, points)
        Bz_vals = linspace(Bz_min, Bz_max, points)
        zero = zeros(points)
        
        A = 0.5 * array([[zero,-Bz_vals,By_vals,zero],[Bz_vals,zero,-Bx_vals,zero],[-By_vals,Bx_vals,zero,zero]]) # general vector potential for B = (Bx, By, Bz)
        A = transpose(A, axes = (2, 0, 1))
        # st.write(f"{A.shape}")
    else:
        ### Pick discretization number in three dimensions
        Bx = st.number_input(r"$B_x$", 
                               min_value=0.0, max_value=None, 
                               value = 0.0, step=None, format=None, 
                               help=r"""Select the value for the x-component of the magnetic field.""",
                               label_visibility="visible") # discretization number in x-direction
        
        By = st.number_input(r"$B_y$", 
                               min_value=0.0, max_value=None, 
                               value = 0.0 , step=None, format=None, 
                               help=r"""Select the value for the y-component of the magnetic field.""",
                               label_visibility="visible") # discretization number in y-direction
        
        Bz = st.number_input(r"$B_z$", 
                               min_value=0.0, max_value=None, 
                               value = 0.0 , step=None, format=None, 
                               help=r"""Select the value for the z-component of the magnetic field.""",
                               label_visibility="visible") # discretization number in z-direction
        kappa = st.number_input(r"$\kappa$", 
                               min_value=0.0, max_value=None, 
                               value = 1.0 , step=None, format=None, 
                               help=r"""Select the value for the magnetic g-factor of the system.""",
                               label_visibility="visible") # value of magnetic g-factor.
        A = 0.5 * array([[0,-Bz,By,0],[Bz,0,-Bx,0],[-By,Bx,0,0]]) # general vector potential for B ) (Bx, By, Bz)
#%% Potential Plot container 

r2_1 = row2[0].container()
with r2_1:
    st.write(r"## Confinement $V(x,y,z)$")
    
    boundx_low = -1 # lower bound in x
    boundx_upp = 1 # upper bound in x
    
    boundy_low = -1 # lower bound in y
    boundy_upp = 1 # upper bound in y
    
    boundz_low = -1 # lower bound in z
    boundz_upp = 1 # upper bound in z
    
    ### Pick the infinite well potential profile
    half_Lx = st.slider("Width of x-well.", min_value = 0.1, max_value = 2.0, value=2.0, 
                   step = 0.1, format = None, key=None, 
                   help = """Use this slider to select the length of the well in the x direction.
                   Note: the well is centered at the origin.""", 
                  label_visibility="visible") / 2 # width of well in x
    
    half_Ly = st.slider("Width of y-well.", min_value = 0.1, max_value = 2.0, value=2.0, 
                   step = 0.1, format = None, key=None, 
                   help = """Use this slider to select the length of the well in the y direction.
                   Note: the well is centered at the origin.""", 
                  label_visibility="visible") / 2 # width of well in y
    
    half_Lz = st.slider("Width of z-well.", min_value = 0.1, max_value = 2.0, value=2.0, 
                   step = 0.1, format = None, key=None, 
                   help = """Use this slider to select the length of the well in the z direction.
                   Note: the well is centered at the origin.""", 
                   label_visibility="visible") / 2 # width of well in z
    
    ### Plotting infinite well profile
    
    X, Y, Z = mgrid[boundx_low : boundx_upp : dimx*1j, 
                    boundy_low : boundy_upp : dimy*1j,
                    boundz_low : boundz_upp : dimz*1j] # meshgrid
    
    potential, u = get_potential_wire(X, Y, Z, 
                                      half_Lx, half_Ly, half_Lz, 
                                      infinity = 1e10) # obtaining potential profile
r2_2 = row2[1].container()
with r2_2:    
    chart = st.empty()
    fig = go.Figure(data=go.Isosurface(
        x = X.flatten(),
        y = Y.flatten(),
        z = Z.flatten(),
        value = u.flatten(),
        isomin = 0,
        isomax = 1,
        opacity = 0.6,
        caps=dict(x_show=True, y_show=True),
        showscale=False
        ))
    
    chart.plotly_chart(fig, use_container_width=True)

#%% Plotting eigenfunction for the obtained parameters
r3 = row3[0].container()
with r3:
    button = st.button("Calculate", key=None, 
                       help="Click when ready to execute calculations.",
                       disabled=False, 
                       use_container_width=True)
    state_chart = st.empty() # initializing empty plot
    energy_val = st.empty() # initializing energy value
    mag_val = st.empty() # initializing magnetic field value
    
    if variable_b_toggle:
        if button:
            eigenvalues = zeros((points, nr_of_soln)) # here we'll store the eigenvectors for all b-values
            eigenvectors = zeros((points, nr_of_soln, dimx, dimy, dimz, 4), complex) # here we'll store the eigenvectors for all b-values
            for i in stqdm(range(points), desc = r"Iterating through $B$-values"):
                needed_terms = preparing_terms(boundx_low, boundx_upp, 
                                               boundy_low, boundy_upp, 
                                               boundz_low, boundz_upp,
                                               dimx, dimy, dimz,
                                               A = A[i],
                                               coeff_x = 1, coeff_y = 1, coeff_z = 1,
                                               bc = 0) # preliminary needed terms
                
                needed_terms["well-walls"]["half_Lx"] = half_Lx # updating needed_terms
                needed_terms["well-walls"]["half_Ly"] = half_Ly # updating needed_terms
                needed_terms["well-walls"]["half_Lz"] = half_Lz # updating needed_terms
                
                hamiltonian = h_tot(needed_terms, 
                          g1 = g1, g2 = g2, g3 = g3,
                          kappa = kappa, B = [Bx_vals[i],By_vals[i],Bz_vals[i]], infinity = 1e10,
                          conf = "wire")[0]
                eigvals, eigvects = eigsh(hamiltonian, k = nr_of_soln, which = "SM") # for CPU
                eigvects = eigvects / norm(eigvects, axis = 0)[None, :] # normalizing eigenvectors
                tmpzip = zip(eigvects.T, eigvals) # zipping eigenvectors with eigenvalues
                sort = sorted(tmpzip, key=lambda x: x[1]) # sorting vectors according to their eigenvalue in increasing order
                eigvects = array([sort[i][0] for i in range(len(sort))]) # extracting sorted eigenvectors
                eigvals = array([sort[i][1] for i in range(len(sort))]) # extracting sorted eigenvalues
                eigenvalues[i] = eigvals # storing eigenvalues
                eigenvectors[i] = eigvects.reshape((nr_of_soln, dimx, dimy, dimz, 4)) # storing eigenvectors
                
            st.session_state['eig-solns_variable_b'] = [eigenvalues, eigenvectors] # storing in session state
            
        if 'eig-solns_variable_b' in st.session_state: # if we calculated the solutions
            eigvals, eigvects = st.session_state['eig-solns_variable_b']
        
            bn = st.number_input("$B$-value", 
                                      min_value=0, max_value= int(points - 1), 
                                       value = 0 , step=None, format=None, 
                                       help="""Select for which magnetic field value to plot.""",
                                       label_visibility="visible") # magnetic field index
            
            n_level = st.number_input("Energy level", 
                                      min_value=0, max_value= int(nr_of_soln - 1), 
                                       value = 0 , step=None, format=None, 
                                       help="""Select the energy level to display.""",
                                       label_visibility="visible") # energy level index
            
            p_dist = Sum(Abs(eigvects[bn, n_level])**2, axis = 3)

            fig_state = go.Figure(data=go.Isosurface(
                x = X.flatten(),
                y = Y.flatten(),
                z = Z.flatten(),
                value = p_dist.flatten(),
                isomin = 1e-5,
                isomax = None,
                opacity = 0.6,
                colorscale = "rdbu_r",
                surface_count = 6,
                colorbar_nticks = 6,
                caps=dict(x_show=False, y_show=False)
                ))
            
            
            state_chart.plotly_chart(fig_state, use_container_width=True)
            energy_val.write(f"E = {Round(eigvals[bn, n_level],2)}")
            mag_val.write(f"B = ({Round(Bx_vals[bn],2)},{Round(By_vals[bn],2)},{Round(Bz_vals[bn],2)}) ")
    else: # if we havent toggled variable magnetic field
        if button:
            with st.spinner('Diagonalizing the problem'):
                needed_terms = preparing_terms(boundx_low, boundx_upp, 
                                               boundy_low, boundy_upp, 
                                               boundz_low, boundz_upp,
                                               dimx, dimy, dimz,
                                               A = A,
                                               coeff_x = 1, coeff_y = 1, coeff_z = 1,
                                               bc = 0)
                
                needed_terms["well-walls"]["half_Lx"] = half_Lx # updating needed_terms
                needed_terms["well-walls"]["half_Ly"] = half_Ly # updating needed_terms
                needed_terms["well-walls"]["half_Lz"] = half_Lz # updating needed_terms
                
                hamiltonian = h_tot(needed_terms, 
                          g1 = g1, g2 = g2, g3 = g3,
                          kappa = kappa, B = [Bx,By,Bz], infinity = 1e10,
                          conf = "wire")[0]
                eigvals, eigvects = eigsh(hamiltonian, k = nr_of_soln, which = "SM") # for CPU
                eigvects = eigvects / norm(eigvects, axis = 0)[None, :] # normalizing eigenvectors
                tmpzip = zip(eigvects.T, eigvals) # zipping eigenvectors with eigenvalues
                sort = sorted(tmpzip, key=lambda x: x[1]) # sorting vectors according to their eigenvalue in increasing order
                eigvects = array([sort[i][0] for i in range(len(sort))]) # extracting sorted eigenvectors
                eigvals = array([sort[i][1] for i in range(len(sort))]) # extracting sorted eigenvalues
                st.session_state['eig-solns'] = [eigvals, eigvects] # storing in session state
            
        if 'eig-solns' in st.session_state: # if we calculated the solutions
            eigvals, eigvects = st.session_state['eig-solns']
            def get_v(n):
                return eigvects[n].reshape((dimx, dimy, dimz, 4))
            
            n_level = st.number_input("Energy level", 
                                      min_value=0, max_value= int(nr_of_soln - 1), 
                                       value = 0 , step=None, format=None, 
                                       help="""Select the energy level to display.""",
                                       label_visibility="visible") # energy level index
            
            p_dist = Sum(Abs(get_v(n_level))**2, axis = 3)
    
            fig_state = go.Figure(data=go.Isosurface(
                x = X.flatten(),
                y = Y.flatten(),
                z = Z.flatten(),
                value = p_dist.flatten(),
                isomin = 1e-5,
                isomax = None,
                opacity = 0.6,
                colorscale = "rdbu_r",
                surface_count = 6,
                colorbar_nticks = 6,
                caps=dict(x_show=False, y_show=False)
                ))
            
            state_chart.plotly_chart(fig_state, use_container_width=True)
            energy_val.write(f"E = {Round(eigvals[n_level],2)}")


        
    
        
    

 
