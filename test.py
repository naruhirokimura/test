import pickle
import numpy as np
import pandas as pd
import streamlit as st

st.write("hellow")

with open('model.pickle', 'rb') as f:
    model = pickle.load(f)
