#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.sklearn.estimator import SKLearn

S3_BUCKET = 'network-anomaly-detection-bucket'
S3_KEY_TRAIN = 'train_data.csv'
S3_KEY_TEST = 'test_data.csv'

role = get_execution_role()
sagemaker_session = sagemaker.Session()

input_data = sagemaker.inputs.TrainingInput(s3_data=f's3://{S3_BUCKET}/{S3_KEY_TRAIN}', content_type='csv')

sklearn_estimator = SKLearn(
    entry_point='train.py',
    role=role,
    instance_count=1,
    instance_type='ml.m5.large',
    framework_version='0.23-1',
    py_version='py3',
    output_path=f's3://{S3_BUCKET}/output'
)

# Train the model
sklearn_estimator.fit({'train': input_data})

