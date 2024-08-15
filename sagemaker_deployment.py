#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import boto3
import sagemaker
from sagemaker.sklearn.model 
import SKLearnModel

S3_BUCKET = 'network-anomaly-detection-bucket'
model_artifact = f's3://{S3_BUCKET}/output/model.tar.gz'

role = sagemaker.get_execution_role()

# Deploying the model
model = SKLearnModel(model_data=model_artifact, role=role, entry_point='inference.py')
predictor = model.deploy(instance_type='ml.m5.large', initial_instance_count=1)

print(f"Model deployed to endpoint: {predictor.endpoint_name}")

