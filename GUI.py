# my imports
import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__)) # current directory
sys.path.append(dir_path+'/Modules') # appending modules folder
from HamiltonianBulk import get_potential_wire, preparing_terms, h_tot
from HamiltonianBulkProjection import eigfn, h_tot_v, get_ks
from UsefulFunctions import possible_states
from StreamlitSolvers import projection_solver_static, projection_solver_var_b, projection_solver_var_t

# plotting imports 
import plotly.graph_objects as go

# other imports
import streamlit as st 
from numpy import array, linspace, zeros, ones, transpose, mgrid, diff, cos
from numpy import round as Round
from numpy import sum as Sum
from numpy import abs as Abs
from numpy.linalg import norm
from scipy.sparse.linalg import eigsh, expm
from stqdm import stqdm # tqdm-like progress bar
#%%Constants & functions
infinity = 1e10
#%% page settings
st.set_page_config(page_title = 'FBLK', 
                   layout = 'wide', 
                   page_icon = ':atom_symbol:',
                   menu_items={
                   'Get Help': 'https://github.com/OtiDioti/FBLK-py/issues',
                   'Report a bug': "https://github.com/OtiDioti/FBLK-py/issues",
                   'About': "**The app is work in progress: any comment/suggestion/request is welcome!**"},
                   initial_sidebar_state="collapsed")
st.title("Four band Luttinger Kohn Hamiltonian")

st.divider()
#%% setting page layout
error_box = st.empty()
row1 = st.columns(8) # problem settings
row2 = st.columns(8) # paramters picker
row3 = st.columns(8) # magnetic field settings
st.divider()
st.write(r"## Confinement $V(x,y,z)$")
row4 = st.columns(2) # potential plotter

calculate_button = st.button("Calculate", key=None, 
                             help="Click when ready to execute calculations.",
                             disabled=False, 
                             use_container_width=True)

row5 = st.columns(2) # results row
row6 = st.columns(1) # results row
#%% Including Toggles
with row1[-1].container():
    method = st.selectbox("Method", ("Projection", "FEM"),
                                 help = "Select with which method to perform the analysis.")

with row2[-1].container():
    variable_b_toggle = st.toggle("Variable $B$", value=False, help="Toggle to study variable magnetic field.")
    
with row3[-1].container():
    time_dependence_toggle = st.toggle("Oscillating field", value=False, help="Toggle to inlcude the effects of an oscillating field.")
#%% If Pojection is chosen method
if method == "Projection":
    ### Determining the problem size
    with row1[0].container():
        st.write("#### Settings")
    ### Pick discretization number in three dimensions
    with row1[1].container():
        nx_max = st.number_input(r"\# of $|n_x\rangle$ .", 
                               min_value=1, max_value=None, 
                               value = 10, step=None, format=None, 
                               help="""Select the number of orbital basis states to project to in x.""",
                               label_visibility="visible") # highest state projected on x
    with row1[2].container():
        ny_max = st.number_input(r"\# of $|n_y\rangle$ .", 
                               min_value=1, max_value=None, 
                               value = 10, step=None, format=None, 
                               help="""Select the number of orbital basis states to project to in y.""",
                               label_visibility="visible") # highest state projected on x
    with row1[3].container():
        nz_max = st.number_input(r"\# of $|n_z\rangle$ .", 
                               min_value=1, max_value=None, 
                               value = 10, step=None, format=None, 
                               help="""Select the number of orbital basis states to project to in z.""",
                               label_visibility="visible") # highest state projected on x
    with row1[4].container():
        nr_of_soln = st.number_input("\# of solutions", 
                               min_value=1, max_value=None, 
                               value = 3 , step=None, format=None, 
                               help="""Select the number of solutions with lowest eigen energy to solve for.""",
                               label_visibility="visible") # discretization number in z-direction
    dim = nx_max * ny_max * nz_max # orbital dimensionality of the system
    possible_statess = possible_states(nx_max, ny_max, nz_max) # permutation of all possible states

        
    if time_dependence_toggle and not(variable_b_toggle): # if time dependence is included and no variable b
        with row1[5].container():
            points_t = st.number_input("\# of t-steps.", 
                                       min_value=10, max_value=None, 
                                       value = 10, step=None, format=None, 
                                       help="""Select the number of steps to discretize time""",
                                       label_visibility="visible") # discretization number in z-direction
        with row1[6].container():
            tmax = st.number_input(r"$t_f$", 
                                   min_value=1, max_value=None, 
                                   value = 1, step=None, format=None, 
                                   help="""Select the final value of time""",
                                   label_visibility="visible") # discretization number in z-direction
        tvals = linspace(0, tmax, points_t) # range of time values
        dt = diff(tvals)[0] # discretization number for time 
#%% if FEM is Chosen
elif method == "FEM":
    error_box.error("""This implementation for the FEM method is not complete and
                    may take some time to run or return inexact results. For now, 
                    we suggest using the Projection method for faster and more accurate
                    results.""", icon="🚨")
    ### Determining the problem size
    with row1[0].container():
        st.write("#### Settings")
    ### Pick discretization number in three dimensions
    with row1[1].container():
        
        dimx = st.number_input("\# of x-steps.", 
                               min_value=5, max_value=None, 
                               value = 30, step=None, format=None, 
                               help="""Select the number of steps to discretize x dimension
                               between x = -1 and x = 1""",
                               label_visibility="visible") # discretization number in x-direction
    with row1[2].container():
        dimy = st.number_input("\# of y-steps.", 
                               min_value=5, max_value=None, 
                               value = 30 , step=None, format=None, 
                               help="""Select the number of steps to discretize y dimension
                               between y = -1 and y = 1""",
                               label_visibility="visible") # discretization number in y-direction
    with row1[3].container():
        dimz = st.number_input("\# of z-steps.", 
                               min_value=5, max_value=None, 
                               value = 30 , step=None, format=None, 
                               help="""Select the number of steps to discretize z dimension
                               between z = -1 and z = 1""",
                               label_visibility="visible") # discretization number in z-direction
    with row1[4].container():
        nr_of_soln = st.number_input("\# of solutions", 
                               min_value=1, max_value=None, 
                               value = 3 , step=None, format=None, 
                               help="""Select the number of solutions with lowest eigen energy to solve for.""",
                               label_visibility="visible") # discretization number in z-direction
        
    if time_dependence_toggle and not(variable_b_toggle): # if time dependence is included and no variable b
        with row1[5].container():
            points_t = st.number_input("\# of t-steps.", 
                                       min_value=10, max_value=None, 
                                       value = 10, step=None, format=None, 
                                       help="""Select the number of steps to discretize time""",
                                       label_visibility="visible") # discretization number in z-direction
        with row1[6].container():
            tmax = st.number_input(r"$t_f$", 
                                   min_value=1, max_value=None, 
                                   value = 1, step=None, format=None, 
                                   help="""Select the final value of time""",
                                   label_visibility="visible") # discretization number in z-direction
        tvals = linspace(0, tmax, points_t) # range of time values
        dt = diff(tvals)[0] # discretization number for time         
#%% Selecting  parameters
with row2[0].container():
    st.write("#### Parameters")
### Pick discretization number in three dimensions
with row2[1].container():    
    g1 = st.number_input(r"$\gamma_1$", 
                           min_value=0.0, max_value=None, 
                           value = 13.35, step=None, format=None, 
                           help=r"""Select the value for the luttinger parameter $\gamma_1$. Default: Ge.""",
                           label_visibility="visible") # discretization number in x-direction
with row2[2].container():  
    g2 = st.number_input(r"$\gamma_2$", 
                           min_value=0.0, max_value=None, 
                           value = 4.25 , step=None, format=None, 
                           help=r"""Select the value for the luttinger parameter $\gamma_2$. Default: Ge.""",
                           label_visibility="visible") # discretization number in y-direction
with row2[3].container():  
    g3 = st.number_input(r"$\gamma_3$", 
                           min_value=0.0, max_value=None, 
                           value = 5.69 , step=None, format=None, 
                           help=r"""Select the value for the luttinger parameter $\gamma_3$. Default: Ge.""",
                           label_visibility="visible") # discretization number in z-direction
with row2[4].container():  
    kappa = st.number_input(r"$\kappa$", 
                           min_value=0.0, max_value=None, 
                           value = 1.0 , step=None, format=None, 
                           help=r"""Select the value for the magnetic g-factor of the system.""",
                           label_visibility="visible") # value of magnetic g-factor

#%% Settign magnetic field
with row3[0].container():
    if time_dependence_toggle and not(variable_b_toggle): # if time dependence is included and no variable b
        st.write("#### Ext. field")
    else: # if variable_b_toggle is True
        st.write("#### $B$-field")
        
if variable_b_toggle:
    ### Pick magnetic field interval
    with row3[1].container(): 
        Bx_min, Bx_max = st.slider(r"$B_x$ range.", 0.0, 10.0, (0.0, 0.0),
                                   help ="Use this slider to select between which values to vary the x-component of the magnetic field." ) # obtaining range of values for Bx
    with row3[2].container(): 
        By_min, By_max = st.slider(r"$B_y$ range.", 0.0, 10.0, (0.0, 0.0),
                                   help ="Use this slider to select between which values to vary the y-component of the magnetic field." ) # obtaining range of values for By
    with row3[3].container():                        
        Bz_min, Bz_max = st.slider(r"$B_z$ range.", 0.0, 10.0, (0.0, 1.0),
                                   help ="Use this slider to select between which values to vary the z-component of the magnetic field." ) # obtaining range of values for Bz
    with row3[4].container():
        points_b = st.number_input("nr. of of B-values.", 
                               min_value=2, max_value=None, 
                               value = 10, step=None, format=None, 
                               help= r"""Select the number of steps between $B_0$ and $B_f$""",
                               label_visibility="visible") # discretization number for B         

    Bx_vals = linspace(Bx_min, Bx_max, points_b)
    By_vals = linspace(By_min, By_max, points_b)
    Bz_vals = linspace(Bz_min, Bz_max, points_b)
    zero = zeros(points_b)
    
    A = 0.5 * array([[zero,-Bz_vals,By_vals,zero],[Bz_vals,zero,-Bx_vals,zero],[-By_vals,Bx_vals,zero,zero]]) # general vector potential for B = (Bx, By, Bz)
    A = transpose(A, axes = (2, 0, 1))
    # st.write(f"{A.shape}")
else: # if variable_b_toggle is False
    if time_dependence_toggle: # if time dependence is True
        zero = zeros(points_t)
        ones = ones(points_t)
        ### Pick magnetic field values
        with row3[1].container(): 
            Bx = st.number_input(r"$B_x$", 
                                   min_value=0.0, max_value=None, 
                                   value = 0.0, step=None, format=None, 
                                   help=r"""Select the value for the x-component of the magnetic field.""",
                                   label_visibility="visible") * ones# discretization number in x-direction
        with row3[2].container(): 
            By = st.number_input(r"$B_y$", 
                                   min_value=0.0, max_value=None, 
                                   value = 0.0 , step=None, format=None, 
                                   help=r"""Select the value for the y-component of the magnetic field.""",
                                   label_visibility="visible") * ones # discretization number in y-direction
        with row3[3].container(): 
            Bz = st.number_input(r"$B_z$", 
                                   min_value=0.0, max_value=None, 
                                   value = 0.0 , step=None, format=None, 
                                   help=r"""Select the value for the z-component of the magnetic field.""",
                                   label_visibility="visible") * ones # discretization number in z-direction
        with row3[4].container(): 
            E0 = st.number_input(r"$E_0$", 
                                 min_value=0.0, max_value=None, 
                                 value = 0.0 , step=None, format=None, 
                                 help=r"""Select the value for the amplitude of the oscillating field.""",
                                 label_visibility="visible") # discretization number in z-direction
        with row3[5].container(): 
            w = st.number_input(r"$\omega$", 
                                min_value=0.1, max_value=None, 
                                value = 0.1 , step=None, format=None, 
                                help=r"""Select the value for the frequency of the oscillating field.""",
                                label_visibility="visible") # discretization number in z-direction
        
        A = 0.5 * array([[zero,-Bz,By,zero],[Bz,zero,-Bx,zero],[-By,Bx,zero, E0/w * cos(w * tvals)]]) # general vector potential for B ) (Bx, By, Bz)
        A = transpose(A, axes = (2, 0, 1))
    else: # if time_dependence_toggle is False
        ### Pick magnetic field values
        with row3[1].container(): 
            Bx = st.number_input(r"$B_x$", 
                                   min_value=0.0, max_value=None, 
                                   value = 0.0, step=None, format=None, 
                                   help=r"""Select the value for the x-component of the magnetic field.""",
                                   label_visibility="visible") # discretization number in x-direction
        with row3[2].container(): 
            By = st.number_input(r"$B_y$", 
                                   min_value=0.0, max_value=None, 
                                   value = 0.0 , step=None, format=None, 
                                   help=r"""Select the value for the y-component of the magnetic field.""",
                                   label_visibility="visible") # discretization number in y-direction
        with row3[3].container(): 
            Bz = st.number_input(r"$B_z$", 
                                   min_value=0.0, max_value=None, 
                                   value = 0.0 , step=None, format=None, 
                                   help=r"""Select the value for the z-component of the magnetic field.""",
                                   label_visibility="visible") # discretization number in z-direction
            A = 0.5 * array([[0,-Bz,By,0],[Bz,0,-Bx,0],[-By,Bx,0,0]]) # general vector potential for B ) (Bx, By, Bz)

#%% Potential Plot container 
with row4[0].container():
    boundx_low = -1 # lower bound in x
    boundx_upp = 1 # upper bound in x
    
    boundy_low = -1 # lower bound in y
    boundy_upp = 1 # upper bound in y
    
    boundz_low = -1 # lower bound in z
    boundz_upp = 1 # upper bound in z
    
    ### Pick the infinite well potential profile
    Lx = st.slider("Width of x-well.", min_value = 0.2, max_value = 2.0, value=2.0, 
                   step = 0.1, format = None, key=None, 
                   help = """Use this slider to select the length of the well in the x direction.
                   Note: the well is centered at the origin.""", 
                  label_visibility="visible") # width of well in x
    
    Ly = st.slider("Width of y-well.", min_value = 0.2, max_value = 2.0, value=2.0, 
                   step = 0.1, format = None, key=None, 
                   help = """Use this slider to select the length of the well in the y direction.
                   Note: the well is centered at the origin.""", 
                  label_visibility="visible") # width of well in y
    
    Lz = st.slider("Width of z-well.", min_value = 0.2, max_value = 2.0, value=2.0, 
                   step = 0.1, format = None, key=None, 
                   help = """Use this slider to select the length of the well in the z direction.
                   Note: the well is centered at the origin.""", 
                   label_visibility="visible") # width of well in z
    half_Lx, half_Ly, half_Lz = [Lx * 0.5, Ly * 0.5, Lz * 0.5] # half of well width z
    
    if method == "Projection": # in this case we have to define here dimx, dimy, dimz, since user does not pick these for this method
        dimx = 35
        dimy = 35
        dimz = 35
    
    ### Plotting infinite well profile
    X, Y, Z = mgrid[boundx_low : boundx_upp : dimx*1j, 
                    boundy_low : boundy_upp : dimy*1j,
                    boundz_low : boundz_upp : dimz*1j] # meshgrid
    
    potential, u = get_potential_wire(X, Y, Z, 
                                      half_Lx, half_Ly, half_Lz, 
                                      infinity = infinity) # obtaining potential profile
with row4[1].container():    
    chart = st.empty()
    fig = go.Figure(data=go.Isosurface(
        x = X.flatten(),
        y = Y.flatten(),
        z = Z.flatten(),
        value = u.flatten(),
        isomin = None,
        isomax = 1,
        opacity = 0.6,
        caps=dict(x_show=True, y_show=True, z_show=True),
        showscale=False
        ))
    
    chart.plotly_chart(fig, use_container_width=True, 
                       config = {'displayModeBar': False})

#%% Solver Projection
if method == "Projection":
    if variable_b_toggle:
        if calculate_button:
            if 'eig-solns_variable_b' in st.session_state: # if we had previously made a run for variable b
                del st.session_state['eig-solns_variable_b'] # emtpies previous run of varying magnetic field
            st.session_state["eig-solns_variable_b"] = projection_solver_var_b(A, 
                                                                               Lx, Ly, Lz,
                                                                               g1, g2, g3,
                                                                               kappa, Bx_vals, By_vals, Bz_vals, 
                                                                               points_b, nr_of_soln, dim, possible_statess)
        # Plotting solutions
        with row5[0].container():   
            if 'eig-solns_variable_b' in st.session_state: # if we calculated the solutions
                eigvals, eigvects_orbi, eigvects_spin = st.session_state['eig-solns_variable_b']
                energy_val = st.empty() # initializing energy value
                mag_val = st.empty() # initializing magnetic field value
                
                bn = st.number_input("$B$-value", 
                                     min_value=0, max_value= int(points_b - 1), 
                                     value = 0 , step=None, format=None, 
                                     help="""Select for which magnetic field value to plot.""",
                                     label_visibility="visible") # magnetic field index
                n_level = st.number_input("Energy level", 
                                          min_value=0, max_value= int(nr_of_soln - 1), 
                                          value = 0 , step=None, format=None, 
                                          help="""Select the energy level to display.""",
                                          label_visibility="visible") # energy level index
                dimx = st.number_input(r"dim$_x$", 
                                     min_value=1, max_value=None, 
                                     value = 30 , step=None, format=None, 
                                     help="""Select discretization number for x-direction.""",
                                     label_visibility="visible") # number of points in x direction
                dimy = st.number_input(r"dim$_y$", 
                                     min_value=1, max_value=None, 
                                     value = 30 , step=None, format=None, 
                                     help="""Select discretization number for y-direction.""",
                                     label_visibility="visible") # number of points in y direction
                dimz = st.number_input(r"dim$_z$", 
                                     min_value=1, max_value=None, 
                                     value = 30 , step=None, format=None, 
                                     help="""Select discretization number for z-direction.""",
                                     label_visibility="visible") # number of points in z direction
            
                X, Y, Z = mgrid[boundx_low : boundx_upp : dimx*1j, 
                                boundy_low : boundy_upp : dimy*1j,
                                boundz_low : boundz_upp : dimz*1j] # meshgrid
                
                p_dist = Abs(eigfn(X, Y, Z,
                                   eigvects_orbi[bn, n_level], possible_statess,
                                   Lx, Ly, Lz))**2
    
                fig_state = go.Figure(data=go.Isosurface(
                    x = X.flatten(),
                    y = Y.flatten(),
                    z = Z.flatten(),
                    value = p_dist.flatten(),
                    isomin = None,
                    isomax = None,
                    opacity = 0.6,
                    colorscale = "rdbu_r",
                    surface_count = 6,
                    colorbar_nticks = 6,
                    caps=dict(x_show=False, y_show=False, z_show=False)
                    ))
                
                energy_val.write(f"E = {Round(eigvals[bn, n_level],2)}")
                mag_val.write(f"B = ({Round(Bx_vals[bn],2)},{Round(By_vals[bn],2)},{Round(Bz_vals[bn],2)}) ")
                
        # Plotting 
        if 'eig-solns_variable_b' in st.session_state: # if we calculated the solutions
            with row5[1].container():
                state_chart = st.empty() # initializing empty plot
                state_chart.plotly_chart(fig_state, use_container_width=True, 
                                         config = {'displayModeBar': False})
            with row6[0].container():    
                st.line_chart(eigvals)

    else: # if we havent toggled variable b
        if time_dependence_toggle: # if we toggled time dependence and not variable b
            if calculate_button:
                if 'eig-solns_t' in st.session_state: # if we had previously made a run for variable b
                    del st.session_state['eig-solns_t'] # emtpies previous run of varying magnetic field
                st.session_state['eig-solns_t'] = projection_solver_var_t(A, 
                                                                          Lx, Ly, Lz,
                                                                          g1, g2, g3,
                                                                          kappa, Bx, By, Bz, 
                                                                          points_t, nr_of_soln, dim, possible_statess)
                
            # Plotting solutions
            with row5[0].container():   
                if 'eig-solns_t' in st.session_state: # if we calculated the solutions
                    eigvals, eigvects_orbi, eigvects_spin = st.session_state['eig-solns_t']
                    energy_val = st.empty() # initializing energy value
                    
                    tn = st.number_input("$t$-value", 
                                         min_value=0, max_value= int(points_t - 1), 
                                         value = 0 , step=None, format=None, 
                                         help="""Select for which moment in time to plot.""",
                                         label_visibility="visible") # magnetic field index
                    n_level = st.number_input("Energy level", 
                                              min_value=0, max_value= int(nr_of_soln - 1), 
                                              value = 0 , step=None, format=None, 
                                              help="""Select the energy level to display.""",
                                              label_visibility="visible") # energy level index
                    dimx = st.number_input(r"dim$_x$", 
                                         min_value=1, max_value=None, 
                                         value = 30 , step=None, format=None, 
                                         help="""Select discretization number for x-direction.""",
                                         label_visibility="visible") # number of points in x direction
                    dimy = st.number_input(r"dim$_y$", 
                                         min_value=1, max_value=None, 
                                         value = 30 , step=None, format=None, 
                                         help="""Select discretization number for y-direction.""",
                                         label_visibility="visible") # number of points in y direction
                    dimz = st.number_input(r"dim$_z$", 
                                         min_value=1, max_value=None, 
                                         value = 30 , step=None, format=None, 
                                         help="""Select discretization number for z-direction.""",
                                         label_visibility="visible") # number of points in z direction
                
                    X, Y, Z = mgrid[boundx_low : boundx_upp : dimx*1j, 
                                    boundy_low : boundy_upp : dimy*1j,
                                    boundz_low : boundz_upp : dimz*1j] # meshgrid
                    
                    p_dist = Abs(eigfn(X, Y, Z,
                                       eigvects_orbi[tn, n_level], possible_statess,
                                       Lx, Ly, Lz))**2
        
                    fig_state = go.Figure(data=go.Isosurface(
                        x = X.flatten(),
                        y = Y.flatten(),
                        z = Z.flatten(),
                        value = p_dist.flatten(),
                        isomin = None,
                        isomax = None,
                        opacity = 0.6,
                        colorscale = "rdbu_r",
                        surface_count = 6,
                        colorbar_nticks = 6,
                        caps=dict(x_show=False, y_show=False, z_show=False)
                        ))
                    
                    energy_val.write(f"E = {Round(eigvals[tn, n_level],2)}")
                    
            # Plotting 
            if 'eig-solns_t' in st.session_state: # if we calculated the solutions
                with row5[1].container():
                    state_chart = st.empty() # initializing empty plot
                    state_chart.plotly_chart(fig_state, use_container_width=True, 
                                             config = {'displayModeBar': False})
                with row6[0].container():    
                    st.line_chart(eigvals)


        else: # if we just want static solution
            if calculate_button:
                if 'eig-solns' in st.session_state: # if we had previously made a run for static solutions
                    del st.session_state['eig-solns'] # emtpies previous run for static solutions
                st.session_state['eig-solns'] = projection_solver_static(A, 
                                                                         Lx, Ly, Lz,
                                                                         g1, g2, g3,
                                                                         kappa, Bx, By, Bz, 
                                                                         nr_of_soln, dim, possible_statess)
                
            # Plotting solutions
            with row5[0].container():   
                if 'eig-solns' in st.session_state: # if we calculated the solutions
                    eigvals, eigvects_orbi, eigvects_spin = st.session_state['eig-solns']
                    energy_val = st.empty() # initializing energy value
                    
                    n_level = st.number_input("Energy level", 
                                              min_value=0, max_value= int(nr_of_soln - 1), 
                                              value = 0 , step=None, format=None, 
                                              help="""Select the energy level to display.""",
                                              label_visibility="visible") # energy level index
                    dimx = st.number_input(r"dim$_x$", 
                                         min_value=1, max_value=None, 
                                         value = 30 , step=None, format=None, 
                                         help="""Select discretization number for x-direction.""",
                                         label_visibility="visible") # number of points in x direction
                    dimy = st.number_input(r"dim$_y$", 
                                         min_value=1, max_value=None, 
                                         value = 30 , step=None, format=None, 
                                         help="""Select discretization number for y-direction.""",
                                         label_visibility="visible") # number of points in y direction
                    dimz = st.number_input(r"dim$_z$", 
                                         min_value=1, max_value=None, 
                                         value = 30 , step=None, format=None, 
                                         help="""Select discretization number for z-direction.""",
                                         label_visibility="visible") # number of points in z direction
                
                    X, Y, Z = mgrid[boundx_low : boundx_upp : dimx*1j, 
                                    boundy_low : boundy_upp : dimy*1j,
                                    boundz_low : boundz_upp : dimz*1j] # meshgrid
                    
                    p_dist = Abs(eigfn(X, Y, Z,
                                       eigvects_orbi[n_level], possible_statess,
                                       Lx, Ly, Lz))**2
        
                    fig_state = go.Figure(data=go.Isosurface(
                        x = X.flatten(),
                        y = Y.flatten(),
                        z = Z.flatten(),
                        value = p_dist.flatten(),
                        isomin = None,
                        isomax = None,
                        opacity = 0.6,
                        colorscale = "rdbu_r",
                        surface_count = 6,
                        colorbar_nticks = 6,
                        caps=dict(x_show=False, y_show=False, z_show=False)
                        ))
                    
                    energy_val.write(f"E = {Round(eigvals[n_level],2)}")                    
            # Plotting 
            if 'eig-solns' in st.session_state: # if we calculated the solutions
                with row5[1].container():
                    state_chart = st.empty() # initializing empty plot
                    state_chart.plotly_chart(fig_state, use_container_width=True, 
                                             config = {'displayModeBar': False})

#%% Solver FEM
elif method == "FEM":
    ### Plotting eigenfunction for the obtained parameters
    
    if variable_b_toggle:
        if calculate_button:
            if 'eig-solns_variable_b' in st.session_state:
                del st.session_state['eig-solns_variable_b'] # emtpies previous run of varying magnetic field
            eigenvalues = zeros((points_b, nr_of_soln)) # here we'll store the eigenvectors for all b-values
            eigenvectors = zeros((points_b, nr_of_soln, dimx, dimy, dimz, 4), complex) # here we'll store the eigenvectors for all b-values
            for i in stqdm(range(points_b), desc = r"Iterating through $B$-values"):
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
                          kappa = kappa, B = [Bx_vals[i],By_vals[i],Bz_vals[i]], infinity = infinity,
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
            
        with row5[0].container():   
            if 'eig-solns_variable_b' in st.session_state: # if we calculated the solutions
                eigvals, eigvects = st.session_state['eig-solns_variable_b']
                
                energy_val = st.empty() # initializing energy value
                mag_val = st.empty() # initializing magnetic field value
                
                bn = st.number_input("$B$-value", 
                                     min_value=0, max_value= int(points_b - 1), 
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
                    isomin = None,
                    isomax = None,
                    opacity = 0.6,
                    colorscale = "rdbu_r",
                    surface_count = 6,
                    colorbar_nticks = 6,
                    caps=dict(x_show=False, y_show=False, z_show=False)
                    ))
                
                energy_val.write(f"E = {Round(eigvals[bn, n_level],2)}")
                mag_val.write(f"B = ({Round(Bx_vals[bn],2)},{Round(By_vals[bn],2)},{Round(Bz_vals[bn],2)}) ")
        if 'eig-solns_variable_b' in st.session_state: # if we calculated the solutions
            with row5[1].container():
                state_chart = st.empty() # initializing empty plot
                state_chart.plotly_chart(fig_state, use_container_width=True, 
                                         config = {'displayModeBar': False})
            with row6[0].container():    
                st.line_chart(eigvals)            
                
    else: # if we havent toggled variable magnetic field
        if time_dependence_toggle:
            if calculate_button:
                if 'psi_t' in st.session_state:
                    del st.session_state['psi_t'] # emtpies previous run
                with st.spinner('Obtaining ground state'):
                    needed_terms = preparing_terms(boundx_low, boundx_upp, 
                                                   boundy_low, boundy_upp, 
                                                   boundz_low, boundz_upp,
                                                   dimx, dimy, dimz,
                                                   A = A[0],
                                                   coeff_x = 1, coeff_y = 1, coeff_z = 1,
                                                   bc = 0)
                    
                    needed_terms["well-walls"]["half_Lx"] = half_Lx # updating needed_terms
                    needed_terms["well-walls"]["half_Ly"] = half_Ly # updating needed_terms
                    needed_terms["well-walls"]["half_Lz"] = half_Lz # updating needed_terms
                    
                    hamiltonian = h_tot(needed_terms, 
                              g1 = g1, g2 = g2, g3 = g3,
                              kappa = kappa, B = [Bx[0],By[0],Bz[0]], infinity = infinity,
                              conf = "wire")[0]
                    eigvals, eigvects = eigsh(hamiltonian, k = 3, which = "SM") # for CPU
                    eigvects = eigvects / norm(eigvects, axis = 0)[None, :] # normalizing eigenvectors
                    tmpzip = zip(eigvects.T, eigvals) # zipping eigenvectors with eigenvalues
                    sort = sorted(tmpzip, key=lambda x: x[1]) # sorting vectors according to their eigenvalue in increasing order
                    eigvects = array([sort[i][0] for i in range(len(sort))]) # extracting sorted eigenvectors
                    eigvals = array([sort[i][1] for i in range(len(sort))]) # extracting sorted eigenvalues
                    st.session_state["init_state"] = eigvects[0] # obtaining ground state
                
                psi_t = zeros((points_t + 1, dimx * dimy * dimz * 4), complex) # will store evolved states
                psi_t[0] = st.session_state["init_state"] # initializing in ground state
                for n in stqdm(range(points_t), desc = "Integrating through time"):
                    t = tvals[n]
                    needed_terms = preparing_terms(boundx_low, boundx_upp, 
                                                   boundy_low, boundy_upp, 
                                                   boundz_low, boundz_upp,
                                                   dimx, dimy, dimz,
                                                   A = A[n],
                                                   coeff_x = 1, coeff_y = 1, coeff_z = 1,
                                                   bc = 0) # obtaining needed arrays (THIS CAN BE OPTIMIZED)
                    
                    needed_terms["well-walls"]["half_Lx"] = half_Lx # updating needed_terms
                    needed_terms["well-walls"]["half_Ly"] = half_Ly # updating needed_terms
                    needed_terms["well-walls"]["half_Lz"] = half_Lz # updating needed_terms
                    
                    hamiltonian = h_tot(needed_terms, 
                              g1 = g1, g2 = g2, g3 = g3,
                              kappa = kappa, B = [Bx[0],By[0],Bz[0]], infinity = infinity,
                              conf = "wire")[0] # obtaining hamiltonian
                    psi_t[n + 1] = expm(-1j * dt * hamiltonian) @ psi_t[n]
                    psi_t[n + 1] = psi_t[n + 1] / norm(psi_t[n + 1]) # normalizing state
                st.session_state["psi_t"] = psi_t
            with row5[0].container():
                if 'psi_t' in st.session_state: # if we calculated the ground state
                    psi_t = st.session_state['psi_t']
                    def get_v(n):
                        return psi_t[n].reshape((dimx, dimy, dimz, 4))
                    
                    t_point = st.number_input("time frame", 
                                              min_value=0, max_value= int(points_t - 1), 
                                               value = 0 , step=None, format=None, 
                                               help="""Select the time frame to display""",
                                               label_visibility="visible") # energy level index
                    
                    p_dist = Sum(Abs(get_v(t_point))**2, axis = 3)
            
                    fig_state = go.Figure(data=go.Isosurface(
                        x = X.flatten(),
                        y = Y.flatten(),
                        z = Z.flatten(),
                        value = p_dist.flatten(),
                        isomin = None,
                        isomax = None,
                        opacity = 0.6,
                        colorscale = "rdbu_r",
                        surface_count = 6,
                        colorbar_nticks = 6,
                        caps=dict(x_show=False, y_show=False, z_show=False)
                        ))
                    
            if 'init_state' in st.session_state: # if we calculated the initial state
                with row5[1].container():
                    state_chart = st.empty() # initializing empty plot
                    state_chart.plotly_chart(fig_state, use_container_width=True, 
                                             config = {'displayModeBar': False})     
                
        else: # if we havent toggled time dependence    
            if calculate_button:
                if 'eig-solns' in st.session_state:
                    del st.session_state['eig-solns'] # emtpies previous run of static magnetic field
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
                              kappa = kappa, B = [Bx,By,Bz], infinity = infinity,
                              conf = "wire")[0]
                    eigvals, eigvects = eigsh(hamiltonian, k = nr_of_soln, which = "SM") # for CPU
                    eigvects = eigvects / norm(eigvects, axis = 0)[None, :] # normalizing eigenvectors
                    tmpzip = zip(eigvects.T, eigvals) # zipping eigenvectors with eigenvalues
                    sort = sorted(tmpzip, key=lambda x: x[1]) # sorting vectors according to their eigenvalue in increasing order
                    eigvects = array([sort[i][0] for i in range(len(sort))]) # extracting sorted eigenvectors
                    eigvals = array([sort[i][1] for i in range(len(sort))]) # extracting sorted eigenvalues
                    st.session_state['eig-solns'] = [eigvals, eigvects] # storing in session state
                    
            with row5[0].container():
                if 'eig-solns' in st.session_state: # if we calculated the solutions
                    eigvals, eigvects = st.session_state['eig-solns']
                    def get_v(n):
                        return eigvects[n].reshape((dimx, dimy, dimz, 4))
                    energy_val = st.empty() # initializing energy value
                    
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
                        isomin = None,
                        isomax = None,
                        opacity = 0.6,
                        colorscale = "rdbu_r",
                        surface_count = 10,
                        colorbar_nticks = 10,
                        caps=dict(x_show=False, y_show=False, z_show=False)
                        ))
                    energy_val.write(f"E = {Round(eigvals[n_level],2)}")
            if 'eig-solns' in st.session_state: # if we calculated the solutions
                with row5[1].container():
                    state_chart = st.empty() # initializing empty plot
                    
                    state_chart.plotly_chart(fig_state, use_container_width=True, 
                                             config = {'displayModeBar': False})
                        

        
    
        
    

 
