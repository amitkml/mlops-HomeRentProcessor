# -*- coding: utf-8 -*-
from __future__ import annotations

import datetime
import json
import os
import pickle
import urllib.parse
from sre_constants import SUCCESS

import boto3
import mlflow
import pandas as pd
from botocore.exceptions import ClientError
from dateutil.relativedelta import relativedelta
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.experimental import enable_halving_search_cv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score
from sklearn.metrics import r2_score
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
# from datetime import date
# from datetime import datetime
# from datetime import timedelta


scalerfile = 'scaler.sav'
result_slacerfile = 'result_scaler.sav'
output_file = '/tmp/output.csv'
try:
    scaler = pickle.load(open(scalerfile, 'rb'))
    scaler_result = pickle.load(open(result_slacerfile, 'rb'))
except Exception as e:
    print(e)
    print('Error getting model transform sclaer')

s3 = boto3.client('s3')
mlflow_run_id = "b9cf70dad0af4a9fb8db7875b8431947"
# MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# mlflow.set_experiment("house-rent-prediction-experiment")
# logged_model = f'{mlflow_run_id}/stacking'
logged_model = f'{mlflow_run_id}/artifacts/models/stacking'
print(logged_model)
model = mlflow.pyfunc.load_model(logged_model)
print("HomeRent_Model...")
print(model)
# except Exception as e:
#     print(e)
#     print('Error getting model defitnion logged_model')

def read_file_s3_bucket_by_pandas(bucket, key):
    # s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket, Key=key)
    print("HomeRent_File_CONTENT TYPE: " + response['ContentType'])
    home_record_df = pd.read_csv(response['Body'])
    print(home_record_df.head())
    return home_record_df

def read_prepare_feature(homerecord_df):
    X = homerecord_df
    sc_X = StandardScaler()
    X_test = scaler.fit_transform(X)
    print("Feature Scaling Done")
    return X_test, homerecord_df

def predict(features, model):
    preds = model.predict(features)
    pred_list =[]
    for l in range(len(preds)):
        pred_list.append(preds[l])
    return pred_list

def upload_s3(output, key, bucket):
    # workbook_name = 'cost_explorer_report-' + today.strftime('%Y-%m') + '.xlsx'
    try:
        s3_resource = boto3.resource('s3')
        s3_bucket = s3_resource.Bucket(bucket)
        print(f"s3 bucket object:{s3_bucket}")
        # s3_bucket.upload_file(output, key, ExtraArgs={'ACL': 'bucket-owner-full-control'})
        s3_bucket.upload_file(output, key)
    except ClientError as err:
        logger.error(err.response['Error']['Message'])
        raise err


def lambda_handler(event, _):
    # data_payload = event.get('data')
    # if not data_payload:
    #     return
    # pred = get_text_summarization_prediction(data_payload)
    # print(pred)
    # Get the object from the event and show its content type
    bucket = event['Records'][0]['s3']['bucket']['name']
    bucket_dest = os.environ["ServiceConfiguration__DEST_BUCKET"]

    Key_homerent = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
    print(f"HomeRent_File_Record:{event}")
    print(f"HomeRent_File_Name:{Key_homerent}")
    try:
        home_record_df = read_file_s3_bucket_by_pandas(bucket, Key_homerent)
        X_test, _ = read_prepare_feature(home_record_df)
        pred_list = predict(X_test, model)
        home_record_df['Predicted_Rent'] = pred_list
        home_record_df.to_csv(output_file, index=False)
        # response = s3.get_object(Bucket=bucket, Key=Key_homerent)
        # print("HomeRent_File_CONTENT TYPE: " + response['ContentType'])
        today = datetime.datetime.now()
        key = Key_homerent + '_' + today.strftime("%m_%d_%Y_%H_%M_%S") + '.csv'
        upload_s3(output_file, key, bucket_dest)
        return True

    except Exception as e:
        print(e)
        print('Error getting object {} from bucket {}. Make sure they exist and your bucket is in the same region as this function.'.format(Key_homerent, bucket))
        raise e
