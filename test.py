import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

st.title('The prediction model for treatrment response in PBC patients')
st.write('This app aims to predict treatment response for Primary Biliary Cholangitis patients base on Machine learning')
st.write('Please enter pre-treatment data by moving the slide bar.')

# 学習データCSVファイル読み込み
df = pd.read_csv('data for streamlit 20230524.csv')
Y = df['target']
X = df.drop(columns=['target'])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=2)
model = XGBClassifier(scale_pos_weight=3, 
                       colsample_bytree= 0.8,
                       min_child_weight= 0.9,
                       max_depth = 35,
                       n_estimators = 40).fit(X_train, Y_train)

TP = st.sidebar.slider(label='Total protein (g/dL)', min_value=5.5, max_value=9.3,value=8.0, step=0.1)
ALT = st.sidebar.slider(label='ALT (IU/L)', min_value=8, max_value=1058,value=80)
Tbil = st.sidebar.slider(label='T-Bil (mg/dL)', min_value=0.2, max_value=4.3,value=1.0, step=0.1)

sample = np.array([['TP','ALT','Tbil'],[TP, ALT, Tbil]])
dfsample = pd.DataFrame(data=[[TP, ALT, Tbil]], columns=['TP','ALT','Tbil'])
st.write(dfsample)    
    
pd1=model.predict_proba(dfsample)

if pd1[0,1] <0.841:
  st.write("This patient may not archieve Paris II criteria, please consider additional treatment.")
else:
  st.write('This patient will archieve Paris II criteria')

fig = plt.figure()
fig.set_size_inches(7, 7)
ax1 = fig.add_subplot(111, projection='3d')
df3Dnon = pd.read_csv('pcbmlvalidation3Dtarget-XGB-.csv')
df3Dres = pd.read_csv('pcbmlvalidation3Dtarget+XGB+.csv')
df3Dresnon = pd.read_csv('pcbmlvalidation3Dtarget+XGB-.csv')
df3Dnonres = pd.read_csv('pcbmlvalidation3Dtarget-XGB+.csv')
# X,Y,Z軸にラベルを設定
ax1.set_xlabel("Total Protein")
ax1.set_ylabel("ALT")
ax1.set_zlabel("T-Bil")
sc = ax1.scatter(df3Dnon.TP, df3Dnon.GPT, df3Dnon.Tbil, s=100,color="Black")
sc = ax1.scatter(df3Dres.TP, df3Dres.GPT, df3Dres.Tbil, s=100, color="purple")
sc = ax1.scatter(df3Dresnon.TP, df3Dresnon.GPT, df3Dresnon.Tbil, s=100, color="red")
sc = ax1.scatter(df3Dnonres.TP, df3Dnonres.GPT, df3Dnonres.Tbil, s=100, color="blue")
sc = ax1.scatter(dfsample.TP, dfsample.ALT, dfsample.Tbil, s=100, color="green")
st.pyplot(fig)
st.write("Black: Non responder and Machine learning predicted")
st.write("Blue: non responder but Machine learning mis predicted")
st.write("Red: responder but Machine learning mis predicted")
st.write("Purple: responder and Machine learning predicted")
st.write("Green: sample data")
