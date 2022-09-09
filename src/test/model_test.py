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
from dateutil.relativedelta import (
    relativedelta,
)
from mlflow.tracking import (
    MlflowClient,
)
from prefect import flow
from prefect import get_run_logger
from prefect import task
from prefect.task_runners import (
    SequentialTaskRunner,
)
from sklearn import (
    metrics,
)
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.experimental import (
    enable_halving_search_cv,
)
from sklearn.feature_extraction import (
    DictVectorizer,
)
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
from sklearn.neighbors import (
    KNeighborsRegressor,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import (
    SVR,
)
from sklearn.tree import (
    DecisionTreeRegressor,
)
from wordcloud import (
    WordCloud,
)
## importing all libs
##

##


warnings.filterwarnings(
    "ignore"
)
import pickle

scalerfile = '/home/ubuntu/mlops-zoomcamp/07-project/capstone/src/scaler.sav'
result_slacerfile = '/home/ubuntu/mlops-zoomcamp/07-project/capstone/src/result_scaler.sav'


# scaler_result = pickle.load(open(result_slacerfile, 'rb'))
# def test_model_loaded():
#     assert (
#         1
#         == 1
#     )


def test_load_feature_transform_file():
    scaler = pickle.load(
        open(
            scalerfile,
            'rb',
        )
    )
    print(
        scaler
    )
    assert (
        scaler.mean_[
            0
        ]
        > 0
    )


def test_read_prepare_feature(
    filepath="/home/ubuntu/mlops-zoomcamp/07-project/capstone/src/test_predict.csv",
):
    rent_data = pd.read_csv(
        filepath
    )
    # rent_data = rent_data.drop(['Posted On','Area Locality','Floor'],axis=1)
    # rent_data = pd.get_dummies(rent_data, columns=['Area Type', 'City', 'Furnishing Status', 'Tenant Preferred', 'Point of Contact'])
    X = rent_data
    sc_X = (
        StandardScaler()
    )
    scaler = pickle.load(
        open(
            scalerfile,
            'rb',
        )
    )
    X_test = scaler.fit_transform(
        X
    )
    assert (
        len(
            X_test
        )
        > 0
    )


def test_predict_output(
    filepath="/home/ubuntu/mlops-zoomcamp/07-project/capstone/src/test_predict.csv",
):
    '''
    testing if the model has returned prediction value for all input record
    '''
    # X = rent_data
    rent_data = pd.read_csv(
        filepath
    )
    sc_X = (
        StandardScaler()
    )
    scaler = pickle.load(
        open(
            scalerfile,
            'rb',
        )
    )
    X_test = scaler.fit_transform(
        rent_data
    )
    mlflow_run_id = "b9cf70dad0af4a9fb8db7875b8431947"
    logged_model = f'/home/ubuntu/mlops-zoomcamp/07-project/capstone/src/mlruns/6/{mlflow_run_id}/artifacts/models/stacking'
    model = mlflow.pyfunc.load_model(
        logged_model
    )
    preds = model.predict(
        X_test
    )
    print(
        preds
    )
    # print(preds[0])
    pred_list = (
        []
    )
    for l in range(
        len(
            preds
        )
    ):
        pred_list.append(
            preds[
                l
            ]
        )
    assert len(
        X_test
    ) == len(
        pred_list
    )


def test_predict_rent_value(
    filepath="/home/ubuntu/mlops-zoomcamp/07-project/capstone/src/test_predict.csv",
):
    '''
    testing if the model has returned prediction value of > 0
    '''
    rent_data = pd.read_csv(
        filepath
    )
    sc_X = (
        StandardScaler()
    )
    scaler = pickle.load(
        open(
            scalerfile,
            'rb',
        )
    )
    X_test = scaler.fit_transform(
        rent_data
    )
    mlflow_run_id = "b9cf70dad0af4a9fb8db7875b8431947"
    logged_model = f'/home/ubuntu/mlops-zoomcamp/07-project/capstone/src/mlruns/6/{mlflow_run_id}/artifacts/models/stacking'
    model = mlflow.pyfunc.load_model(
        logged_model
    )
    preds = model.predict(
        X_test
    )
    print(
        preds
    )
    # print(preds[0])
    pred_list = (
        []
    )
    for l in range(
        len(
            preds
        )
    ):
        pred_list.append(
            preds[
                l
            ]
        )
    assert (
        len(
            [
                *filter(
                    lambda x: x
                    >= 30,
                    pred_list,
                )
            ]
        )
        > 0
    )
