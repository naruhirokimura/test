import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

# 学習データCSVファイル読み込み
df = pd.read_csv('data for streamlit 20230524.csv')
Y = df['target']
X = df.drop(columns=['target'])

