import streamlit as st 
#%% page settings
st.set_page_config(page_title = 'FBLK', 
                   layout = 'centered', 
                   page_icon = ':house:',
                   menu_items={
                   'Get Help': 'https://github.com/OtiDioti/FBLK-py/issues',
                   'Report a bug': "https://github.com/OtiDioti/FBLK-py/issues",
                   'About': "**The app is work in progress: any comment/suggestion/request is welcome!**"},
                   initial_sidebar_state="collapsed")

#%% Title
st.title(""" Welcome to FBLK""")
#%% Info text
st.title("""A simple tool to explore the dynamics of holes in semiconductors valence band""")