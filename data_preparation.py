#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import boto3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

S3_BUCKET = 'network-anomaly-detection-bucket'
S3_KEY_TRAIN = 'train_data.csv'
S3_KEY_TEST = 'test_data.csv'

DATASET_PATH = 'UNSW_NB15_training-set.csv'

def preprocess_data():
    df = pd.read_csv(DATASET_PATH)
    df = df.drop(columns=['id', 'label'])  
    df['class'] = df['attack_cat'].apply(lambda x: 0 if x == 'Normal' else 1)
    df = df.drop(columns=['attack_cat'])

    X = df.drop(columns=['class'])
    y = df['class']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    train_data = pd.DataFrame(X_train)
    train_data['class'] = y_train.values
    test_data = pd.DataFrame(X_test)
    test_data['class'] = y_test.values

    return train_data, test_data

def upload_to_s3(train_data, test_data):
    s3 = boto3.client('s3')
    train_data.to_csv('/tmp/train_data.csv', index=False)
    test_data.to_csv('/tmp/test_data.csv', index=False)

    s3.upload_file('/tmp/train_data.csv', S3_BUCKET, S3_KEY_TRAIN)
    s3.upload_file('/tmp/test_data.csv', S3_BUCKET, S3_KEY_TEST)
    print(f"Data uploaded to S3: s3://{S3_BUCKET}/{S3_KEY_TRAIN} and s3://{S3_BUCKET}/{S3_KEY_TEST}")

if __name__ == "__main__":
    train_data, test_data = preprocess_data()
    upload_to_s3(train_data, test_data)

