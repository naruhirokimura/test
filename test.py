import pickle
import numpy as np
import pandas as pd
import streamlit as st

st.write("hellowed")
st.button('Hit me')
st.slider('Slide me', min_value=0, max_value=10)
st.select_slider('Slide to select', options=[1,'2'])

Tbil = st.sidebar.slider(label='T-Bil (mg/dL)', min_value=0.2, max_value=4.3,value=1.0, step=0.1)
TP = st.sidebar.slider(label='Total protein (g/dL)', min_value=0.2, max_value=8.9,value=1.0, step=0.1)
ALT = st.sidebar.slider(label='ALT (IU/L)', min_value=30, max_value=1500,value=100)

sample = np.array([['Tbil','TP','ALP'],[Tbil, TP, ALT]])
dfsample = pd.DataFrame(data=[[Tbil, TP, ALT]], columns=['T-Bil (md/dl)', 'Total protein (g/dL)', 'ALT (IU/L)'])
st.write(dfsample)    


with open('model.pickle', 'rb') as f:
    model = pickle.load(f)

    
pd1=model.predict_proba(dfsample)
