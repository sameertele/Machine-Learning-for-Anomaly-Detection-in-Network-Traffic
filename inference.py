#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import joblib
import pandas as pd

def model_fn(model_dir):
    model = joblib.load(f"{model_dir}/model.joblib")
    return model

def input_fn(input_data, content_type):
    if content_type == 'text/csv':
        return pd.read_csv(input_data)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    predictions = model.predict(input_data)
    return predictions

