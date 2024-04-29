# my imports
import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__)) # current directory
sys.path.append(dir_path+'/Modules') # appending modules folder
from HamiltonianBulk import get_potential_wire, preparing_terms, h_tot

# plotting imports 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# other imports
import streamlit as st 
from numpy import array
from numpy import sum as Sum
from numpy import abs as Abs
from numpy.linalg import norm
from scipy.sparse.linalg import eigsh


#%% page settings
st.set_page_config(page_title = 'FBLK', 
                   layout = 'centered', 
                   page_icon = ':house:',
                   menu_items={
                   'Get Help': 'https://github.com/OtiDioti/FBLK-py/issues',
                   'Report a bug': "https://github.com/OtiDioti/FBLK-py/issues",
                   'About': "**The app is work in progress: any comment/suggestion/request is welcome!**"},
                   initial_sidebar_state="collapsed")
st.title("The time-independent problem.")
st.divider()
#%% setting page layout
row1 = st.columns(3) # first row has 3 colums
row2 = st.columns(2) # second row has 2 columns
row3 = st.columns(2) # third row has 2 colums
#%% Determining the problem size
r1_1 = row1[0].container()
with r1_1:
    st.title("Problem's settings")
    ### Pick discretization number in three dimensions
    dimx = st.number_input("nr. of x-steps.", 
                           min_value=5, max_value=None, 
                           value = 30, step=None, format=None, 
                           help="""Select the number of steps between to discretize x dimensions
                           between x = -1 and x = 1""",
                           label_visibility="visible") # discretization number in x-direction
    
    dimy = st.number_input("nr. of y-steps.", 
                           min_value=5, max_value=None, 
                           value = 30 , step=None, format=None, 
                           help="""Select the number of steps between to discretize y dimensions
                           between y = -1 and x = 1""",
                           label_visibility="visible") # discretization number in y-direction
    
    dimz = st.number_input("nr. of z-steps.", 
                           min_value=5, max_value=None, 
                           value = 30 , step=None, format=None, 
                           help="""Select the number of steps between to discretize z dimensions
                           between z = -1 and x = 1""",
                           label_visibility="visible") # discretization number in z-direction
    nr_of_soln = st.number_input("nr. of eigensolutions.", 
                           min_value=1, max_value=None, 
                           value = 3 , step=None, format=None, 
                           help="""Select the number of solutions with lowest eigen energy to solve for.""",
                           label_visibility="visible") # discretization number in z-direction
#%% Selecting luttinger parameters
r1_2 = row1[1].container()
with r1_2:
    st.title("Luttinger parameters")
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
    st.title("Magnetic Field")
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
                           label_visibility="visible") # discretization number in z-direction
    A = 0.5 * array([[0,-Bz,By,0],[Bz,0,-Bx,0],[-By,Bx,0,0]]) # general vector potential for B ) (Bx, By, Bz)
#%% Potential Plot container 
r2_1 = row2[0].container()
with r2_1:
    st.title("Confinement potential")
    chart = st.empty()
    
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
    
    
    X, Y, Z = [needed_terms["grids"]["X"],
               needed_terms["grids"]["Y"],
               needed_terms["grids"]["Z"]] # extracting meshgrid
    
    potential, u = get_potential_wire(X, Y, Z, 
                                      half_Lx, half_Ly, half_Lz, 
                                      infinity = 1e10) # obtaining potential profile
    
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
r2_2 = row2[1].container()
with r2_2:
    st.title("Eigenfunctions")
    button = st.button("Calculate", key=None, 
                       help="Click when ready to execute calculations.",
                       disabled=False, 
                       use_container_width=True)
    state_chart = st.empty() # initializing empty plot
    energy_val = st.empty() # initializing energy value

    if button:
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
                                   label_visibility="visible") # discretization number in z-direction
        
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
        energy_val.write(f"E = {eigvals[n_level]}")


        
    
        
    

 
