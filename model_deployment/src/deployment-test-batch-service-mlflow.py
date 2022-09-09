# -*- coding: utf-8 -*-
## importing all libs
from __future__ import annotations

import os
import pickle
import random
import warnings
from datetime import date
from datetime import datetime
from datetime import timedelta

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
import requests as req
import seaborn as sns
from dateutil.relativedelta import relativedelta
from mlflow.tracking import MlflowClient
from prefect import flow
from prefect import get_run_logger
from prefect import task
from prefect.task_runners import SequentialTaskRunner
from sklearn import metrics
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.experimental import enable_halving_search_cv
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import BayesianRidge
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
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from wordcloud import WordCloud
##
## importing all libs
##
warnings.filterwarnings("ignore")
import pickle
scalerfile = 'scaler.sav'
result_slacerfile = 'result_scaler.sav'

scaler = pickle.load(open(scalerfile, 'rb'))
scaler_result = pickle.load(open(result_slacerfile, 'rb'))

mlflow_run_id = "b9cf70dad0af4a9fb8db7875b8431947"
# MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# mlflow.set_experiment("house-rent-prediction-experiment")
logged_model = f'mlruns/6/{mlflow_run_id}/artifacts/models/stacking'
model = mlflow.pyfunc.load_model(logged_model)
print(model)

def read_prepare_feature(filepath):
    rent_data = pd.read_csv(filepath)
    s = np.load('mean_train.npy')
    m = np.load('std_train.npy')
    # rent_data = rent_data.drop(['Posted On','Area Locality','Floor'],axis=1)
    # rent_data = pd.get_dummies(rent_data, columns=['Area Type', 'City', 'Furnishing Status', 'Tenant Preferred', 'Point of Contact'])
    X = rent_data
    sc_X = StandardScaler()
    X_test = scaler.fit_transform(X)
    return X_test, rent_data

def predict(features, model):
    preds = model.predict(features)
    print(type(preds))
    print(preds)
    # print(preds[0])
    pred_list =[]
    for l in range(len(preds)):
        pred_list.append(preds[l])
    print(pred_list)
    return pred_list


def run():
    filepath = "test_predict.csv"
    X_test, rent_data = read_prepare_feature(filepath)
    pred_list = predict(X_test, model)
    rent_data['Predicted_Rent'] = pred_list
    rent_data.to_csv("test_predict_result.csv",index=False)

if __name__ == '__main__':
    run()
