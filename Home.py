# my imports
import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__)) # current directory
sys.path.append(dir_path+'/Modules') # appending modules folder
from HamiltonianBulk import get_potential_wire, preparing_terms

# plotting imports 
import plotly.graph_objects as go

# other imports
import streamlit as st 
from numpy import array


#%% page settings
st.set_page_config(page_title = 'FBLK', 
                   layout = 'centered', 
                   page_icon = ':house:',
                   menu_items={
                   'Get Help': 'https://github.com/OtiDioti/FBLK-py/issues',
                   'Report a bug': "https://github.com/OtiDioti/FBLK-py/issues",
                   'About': "**The app is work in progress: any comment/suggestion/request is welcome!**"},
                   initial_sidebar_state="collapsed")

#%% setting page layout
row1 = st.columns(3) # first row has 3 colums
row2 = st.columns(3) # second row has 3 columns
row3 = st.columns(3) # third row has 3 colums
#%% Determining the problem size
c1 = row1[0].container()
with c1:
    st.title("Discretization numbers")
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
                           between x = -1 and x = 1""",
                           label_visibility="visible") # discretization number in y-direction
    
    dimz = st.number_input("nr. of z-steps.", 
                           min_value=5, max_value=None, 
                           value = 30 , step=None, format=None, 
                           help="""Select the number of steps between to discretize z dimensions
                           between x = -1 and x = 1""",
                           label_visibility="visible") # discretization number in z-direction
#%% Potential Plot container 
c2 = row1[2].container()
with c2:
    st.title("Define the confinement potential")
    chart = st.empty()
    
    boundx_low = -1 # lower bound in x
    boundx_upp = 1 # upper bound in x
    
    boundy_low = -1 # lower bound in y
    boundy_upp = 1 # upper bound in y
    
    boundz_low = -1 # lower bound in z
    boundz_upp = 1 # upper bound in z
    
    ### Pick the infinite well potential profile
    half_Lx = st.slider("Width of x-well.", min_value = 0.1, max_value = 2.0, value=None, 
                   step = 0.1, format = None, key=None, 
                   help = """Use this slider to select the length of the well in the x direction.
                   Note: the well is centered at the origin.""", 
                  label_visibility="visible") / 2 # width of well in x
    
    half_Ly = st.slider("Width of y-well.", min_value = 0.1, max_value = 2.0, value=None, 
                   step = 0.1, format = None, key=None, 
                   help = """Use this slider to select the length of the well in the y direction.
                   Note: the well is centered at the origin.""", 
                  label_visibility="visible") / 2 # width of well in y
    
    half_Lz = st.slider("Width of z-well.", min_value = 0.1, max_value = 2.0, value=None, 
                   step = 0.1, format = None, key=None, 
                   help = """Use this slider to select the length of the well in the z direction.
                   Note: the well is centered at the origin.""", 
                   label_visibility="visible") / 2 # width of well in z
    
    ### Plotting infinite well profile
    
    
    needed_terms = preparing_terms(boundx_low, boundx_upp, 
                                   boundy_low, boundy_upp, 
                                   boundz_low, boundz_upp,
                                   dimx, dimy, dimz,
                                   A = array([[0,0,0,0],[0,0,0,0],[0,0,0,0]]),
                                   coeff_x = 1, coeff_y = 1, coeff_z = 1,
                                   bc = 0)
    
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

