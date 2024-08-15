#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
import joblib
import os

def train():
    train_data = pd.read_csv('train_data.csv')
    X_train = train_data.drop(columns=['class'])

    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X_train)

    path = os.path.join('/opt/ml/model', 'model.joblib')
    joblib.dump(model, path)

if __name__ == "__main__":
    train()

