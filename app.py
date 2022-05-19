import streamlit as st
from predict_page import show_predict_page

st.markdown('<style>body{background-color: Blue;}</style>',unsafe_allow_html=True)
show_predict_page()