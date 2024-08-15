#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Testing model with new data
import boto3
import pandas as pd
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer

S3_BUCKET = 'network-anomaly-detection-bucket'
S3_KEY_TEST = 'test_data.csv'

s3 = boto3.client('s3')
s3.download_file(S3_BUCKET, S3_KEY_TEST, 'test_data.csv')
test_data = pd.read_csv('test_data.csv').drop(columns=['class'])

endpoint_name = 'endpoint1'

predictor = Predictor(endpoint_name=endpoint_name)
predictor.serializer = CSVSerializer()

predictions = predictor.predict(test_data.to_csv(index=False, header=False))

print("Predictions:", predictions)

