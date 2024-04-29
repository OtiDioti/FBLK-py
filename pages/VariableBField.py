# my imports
import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__)) # current directory
prev_dir = os.path.abspath(os.path.join(dir_path, os.pardir)) # parent of current directory
sys.path.append(prev_dir+'/Modules') # appending modules folder

# other imports
import streamlit as st 
from streamlit_extras.switch_page_button import switch_page

#%% page settings
st.set_page_config(page_title = r"Variable $\mathbf{B}$-field.", 
                   layout = 'centered', 
                   page_icon = ':atom_symbol:',
                   menu_items={
                   'Get Help': 'https://github.com/OtiDioti/FBLK-py/issues',
                   'Report a bug': "https://github.com/OtiDioti/FBLK-py/issues",
                   'About': "**The app is work in progress: any comment/suggestion/request is welcome!**"},
                   initial_sidebar_state="collapsed")
st.title(r"Variable $\mathbf{B}$-field.")
st.divider()
#%%