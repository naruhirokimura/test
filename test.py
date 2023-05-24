import pickle
import numpy as np
import pandas as pd
import streamlit as st

# 学習データCSVファイル読み込み
df = pd.read_csv('data for streamlit 20230524.csv')
Y = df['target']
X = df.drop(columns=['target'])
X_train, X_test, Y_train, Y_test = train_test_split(X2, Y, test_size=0.30, random_state=2)
model = XGBClassifier(scale_pos_weight=3, 
                       colsample_bytree= 0.8,
                       min_child_weight= 0.9,
                       max_depth = 35,
                       n_estimators = 40).fit(X2_train, Y_train)

Tbil = st.sidebar.slider(label='T-Bil (mg/dL)', min_value=0.2, max_value=4.3,value=1.0, step=0.1)
TP = st.sidebar.slider(label='Total protein (g/dL)', min_value=0.2, max_value=8.9,value=1.0, step=0.1)
ALT = st.sidebar.slider(label='ALT (IU/L)', min_value=30, max_value=1500,value=100)

sample = np.array([['Tbil','TP','ALP'],[Tbil, TP, ALT]])
dfsample = pd.DataFrame(data=[[Tbil, TP, ALT]], columns=['T-Bil (md/dl)', 'Total protein (g/dL)', 'ALT (IU/L)'])
st.write(dfsample)    
    
pd1=model.predict_proba(dfsample)
