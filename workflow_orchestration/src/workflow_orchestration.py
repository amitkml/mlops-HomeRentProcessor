# -*- coding: utf-8 -*-
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
## importing all libs
##
warnings.filterwarnings("ignore")


@task
def read_input_file(filepath='../data/House_Rent_Dataset.csv'):
    '''
    This function will read the input files.
    '''
    rent_data = pd.read_csv(filepath)
    rent_data.isna().sum()
    rent_data.head()
    print("File read done. Displaying record stats")
    print(rent_data.describe())
    return rent_data

@task
def preprocess_record(rent_data):
    '''
    This function will do data preprocessing works.
    '''
    rent_data = rent_data.drop(['Posted On','Area Locality','Floor'],axis=1)
    print("preprocessing of records done. Displaying record stats")
    print(rent_data.describe())
    return rent_data

@task
def make_dataprep_model(rent_data, test_size=0.3, random_state=42):
    '''
    This function will prepare data for modelling purpose
    '''
    rent_data = pd.get_dummies(rent_data, columns=['Area Type', 'City', 'Furnishing Status', 'Tenant Preferred', 'Point of Contact'])
    rent_data.head()
    X = rent_data.drop('Rent',axis=1)
    y = rent_data['Rent']
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=test_size, random_state=random_state)
    print("data prep for modelling done")
    return  X, y, X_train, X_test, y_train, y_test

@task
def scale_data(X_train, X_test, y_train, y_test):
    '''
    This function will Scaling the data
    '''
    y_train= y_train.values.reshape(-1,1)
    y_test= y_test.values.reshape(-1,1)
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.fit_transform(X_test)
    y_train = sc_X.fit_transform(y_train)
    y_test = sc_y.fit_transform(y_test)
    print("Data scaling done")
    return  X_train, X_test, y_train, y_test

@task
def save_model(model, model_name):
    '''
    Saves the new version of a model to the prediction_service/model.pkl
    '''
    path = "prediction_service"
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
        print("The new directory is created!")

    filepath = "prediction_service/" + model_name +".pkl"
    with open(filepath, 'wb') as f_out:
        pickle.dump((model), f_out)
    print(f"{model_name} Model save completed")
    return True

@task
def model_lr(X_train, X_test, y_train, y_test):
    '''
    This function will do linear regression
    '''
    lm = LinearRegression()
    lm.fit(X_train,y_train)
    lm_prediction = lm.predict(X_test)
    print("LR model training completed. Model Evaluation metrics are:")
    mae_lm = metrics.mean_absolute_error(y_test, lm_prediction)
    mse_lm =  metrics.mean_squared_error(y_test, lm_prediction)
    rmse_lm =  np.sqrt(mse_lm)
    print('=============MAE:', mae_lm)
    print('=============MSE:', mse_lm)
    print('=============RMSE:', rmse_lm)
    metrices = {"MAE":mae_lm, "MSE": mse_lm, "RMSE": rmse_lm}

    return lm, lm_prediction, metrices

@task
def model_dt(X_train, X_test, y_train, y_test, random_state=100):
    '''
    This function will do Decision tree
    '''
    dt = DecisionTreeRegressor(random_state = random_state)
    dt.fit(X_train, y_train)
    dt_prediction = dt.predict(X_test)
    print("Decision Tree model training completed. Model Evaluation metrics are:")
    # Evaluation metrics
    mae_dt = metrics.mean_absolute_error(y_test, dt_prediction)
    mse_dt =  metrics.mean_squared_error(y_test, dt_prediction)
    rmse_dt =  np.sqrt(mse_dt)
    print('=============MAE:', mae_dt)
    print('=============MSE:', mse_dt)
    print('=============RMSE:', rmse_dt)
    metrices_dt = {"MAE":mae_dt, "MSE": mse_dt, "RMSE": rmse_dt}
    return dt, dt_prediction, metrices_dt


def five_cv_prarm_grid(PARAM_DICT, ESTIMATOR,X_train,y_train,random_state):
    sh = HalvingGridSearchCV(ESTIMATOR, PARAM_DICT, cv=10, scoring='neg_mean_absolute_error',min_resources="smallest",random_state=random_state).fit(X_train, y_train)
    best_estimator = sh.best_estimator_
    best_param = sh.best_params_
    print(best_estimator)
    print(f"10-CV Best Parameters = {best_param}")
    print(f"10-CV Best Score = {sh.best_score_}")
    return best_estimator, best_param

@task
def model_dt_regressor(X_train, X_test, y_train, y_test, random_state=100):
    '''
    This function will do Decision tree with grid search
    '''
    PARAM_DICT = {'max_depth': [None,3,4,5,6,7,8,9,10,11,15,20,25,30,35,40,45],'min_samples_split': [2,3,4,5],'min_samples_leaf':[1,2,3,4,5]}
    print("Starting Decision Tree regressor:")
    ESTIMATOR =  DecisionTreeRegressor(random_state=random_state)
    best_dtc_estimator, best_param = five_cv_prarm_grid(PARAM_DICT, ESTIMATOR,X_train,y_train,random_state)
    best_dtc_estimator.fit(X_train, y_train)
    dt_prediction = best_dtc_estimator.predict(X_test)
    # Evaluation metrics
    mae_dtr = metrics.mean_absolute_error(y_test, dt_prediction)
    mse_dtr =  metrics.mean_squared_error(y_test, dt_prediction)
    rmse_dtr =  np.sqrt(mse_dtr)
    print(best_dtc_estimator)
    print("Completed Decision Tree regressor:")
    print('=============MAE:', mae_dtr)
    print('=============MSE:', mse_dtr)
    print('=============RMSE:', rmse_dtr)
    metrices_dt_regr = {"MAE":mae_dtr, "MSE": mse_dtr, "RMSE": rmse_dtr}
    return best_dtc_estimator, dt_prediction, metrices_dt_regr

@task
def model_svm(X_train, X_test, y_train, y_test, random_state=100):
    '''
    This function will do svm
    '''
    svr = SVR()
    svr.fit(X_train, y_train)
    svr_prediction = svr.predict(X_test)
    print("SVM model training completed. Model Evaluation metrics are:")
    # Evaluation metrics
    mae_svr = metrics.mean_absolute_error(y_test, svr_prediction)
    mse_svr =  metrics.mean_squared_error(y_test, svr_prediction)
    rmse_svr =  np.sqrt(mse_svr)

    print('=============MAE:', mae_svr)
    print('=============MSE:', mse_svr)
    print('=============RMSE:', rmse_svr)
    metrices_dt_svm = {"MAE":mae_svr, "MSE": mse_svr, "RMSE": rmse_svr}
    return svr, svr_prediction, metrices_dt_svm

@task
def model_rf(X_train, X_test, y_train, y_test, n_estimators=100):
    '''
    This function will do random forest
    '''
    rf = RandomForestRegressor(n_estimators = n_estimators)
    rf.fit(X_train, y_train)
    rf_prediction = rf.predict(X_test)
    print("Random Forest model training completed. Model Evaluation metrics are:")
    # Evaluation metrics
    mae_rf = metrics.mean_absolute_error(y_test, rf_prediction)
    mse_rf =  metrics.mean_squared_error(y_test, rf_prediction)
    rmse_rf =  np.sqrt(mse_rf)

    print('=============MAE:', mae_rf)
    print('=============MSE:', mse_rf)
    print('=============RMSE:', rmse_rf)
    metrices_dt_rf = {"MAE":mae_rf, "MSE": mse_rf, "RMSE": rmse_rf}

    return rf, rf_prediction, metrices_dt_rf

@task
def model_rf_regressor(X_train, X_test, y_train, y_test, random_state=100):
    '''
    This function will do Random forest with grid search
    '''
    # RandomForestRegressor Tune Parameter
    PARAM_DICT = {'n_estimators': [10,50,100,200,300],'min_samples_leaf':[1,2,3]}
    print("Starting Random Forest Tree regressor:")
    ESTIMATOR =  RandomForestRegressor(random_state=random_state)
    best_drfc_estimator, best_param = five_cv_prarm_grid(PARAM_DICT, ESTIMATOR,X_train,y_train,random_state)
    best_drfc_estimator.fit(X_train, y_train)
    drfc_prediction = best_drfc_estimator.predict(X_test)
    # Evaluation metrics
    mae_drfc = metrics.mean_absolute_error(y_test, drfc_prediction)
    mse_drfc =  metrics.mean_squared_error(y_test, drfc_prediction)
    rmse_drfc =  np.sqrt(mse_drfc)
    print(best_drfc_estimator)
    print("Completed Random Forest regressor:")
    print('=============MAE:', mae_drfc)
    print('=============MSE:', mse_drfc)
    print('=============RMSE:', rmse_drfc)
    metrices_dt_rfr = {"MAE":mae_drfc, "MSE": mse_drfc, "RMSE": rmse_drfc}

    return best_drfc_estimator, drfc_prediction, metrices_dt_rfr

@task
def model_knn_regressor(X_train, X_test, y_train, y_test, random_state=100):
    '''
    This function will do KNN with grid search
    '''
    # KNN Regressor Tune Parameter
    PARAM_DICT = {'n_neighbors': [5,7,9,11,13],'weights': ['uniform', 'distance']}
    print("Starting KNN regressor:")
    ESTIMATOR =  KNeighborsRegressor()
    best_KNN_estimator, best_param = five_cv_prarm_grid(PARAM_DICT, ESTIMATOR,X_train,y_train,random_state)
    best_KNN_estimator.fit(X_train, y_train)
    KNN_test  = best_KNN_estimator.predict(X_test)
    # Evaluation metrics
    mae_knnr = metrics.mean_absolute_error(y_test, KNN_test)
    mse_knnr =  metrics.mean_squared_error(y_test, KNN_test)
    rmse_kknr =  np.sqrt(mse_knnr)
    print(best_KNN_estimator)
    print("Completed KNN regressor:")
    print('=============MAE:', mae_knnr)
    print('=============MSE:', mse_knnr)
    print('=============RMSE:', rmse_kknr)
    metrices_knn = {"MAE":mae_knnr, "MSE": mse_knnr, "RMSE": rmse_kknr}
    return best_KNN_estimator, KNN_test, metrices_knn

@task
def model_voting_regressor(X_train, X_test, y_train, y_test, best_KNN_estimator, best_dtc_estimator, best_rfc_estimator, random_state=100):
    print("Starting voting regressor:")
    # model_voting = VotingRegressor(estimators = [('lr',LinearRegression()),('br', BayesianRidge()), ('knn', best_KNN_estimator), \
    #                                           ('dtc', best_dtc_estimator), \
    #                                           ('rfc', best_rfc_estimator)])
    model_voting = VotingRegressor(estimators = [('knn', best_KNN_estimator), \
                                              ('dtc', best_dtc_estimator), \
                                              ('rfc', best_rfc_estimator)])
    model_voting.fit(X_train, y_train)
    print("Completed Voting regressor:")
    voting_test = model_voting.predict(X_test)
    # Evaluation metrics
    mae_voting = metrics.mean_absolute_error(y_test, voting_test)
    mse_voting =  metrics.mean_squared_error(y_test, voting_test)
    rmse_voting =  np.sqrt(mse_voting)
    print(voting_test)
    print("Completed Voting regressor:")
    print('=============MAE:', mae_voting)
    print('=============MSE:', mse_voting)
    print('=============RMSE:', rmse_voting)
    metrices_voting = {"MAE":mae_voting, "MSE": mse_voting, "RMSE": rmse_voting}
    return model_voting, voting_test, metrices_voting

@task
def model_stacking_regressor(X_train, X_test, y_train, y_test, best_KNN_estimator, best_dtc_estimator, best_rfc_estimator, random_state=100):
    print("Starting stacking regressor:")
    # model_voting = VotingRegressor(estimators = [('lr',LinearRegression()),('br', BayesianRidge()), ('knn', best_KNN_estimator), \
    #                                           ('dtc', best_dtc_estimator), \
    #                                           ('rfc', best_rfc_estimator)])
    model_stacking = StackingRegressor(estimators = [('knn', best_KNN_estimator), \
                                              ('dtc', best_dtc_estimator), \
                                              ('rfc', best_rfc_estimator)])
    model_stacking.fit(X_train, y_train)
    print("Completed Stacking regressor:")
    stacking_test = model_stacking.predict(X_test)
    # Evaluation metrics
    mae_stacking = metrics.mean_absolute_error(y_test, stacking_test)
    mse_stacking =  metrics.mean_squared_error(y_test, stacking_test)
    rmse_stacking =  np.sqrt(mse_stacking)
    print(model_stacking)
    print("Completed Voting regressor:")
    print('=============MAE:', mae_stacking)
    print('=============MSE:', mse_stacking)
    print('=============RMSE:', rmse_stacking)
    metrices_stacking = {"MAE":mae_stacking, "MSE": mse_stacking, "RMSE": rmse_stacking}
    return model_stacking, stacking_test, metrices_stacking

MLFLOW_TRACKING_URI = "sqlite:///mlops_zoomcamp_final_project.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("house-rent-prediction-experiment")
# mlflow.sklearn.autolog()

@flow(name="run", task_runner=SequentialTaskRunner())
def run():
    with mlflow.start_run() as run:
        ## Client of an MLflow Tracking Server that creates and manages experiments and runs, and of an MLflow Registry Server that creates and manages registered models and model versions
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        model_name = "house-rent-prediction-model"
        mlflow.set_tag("developer", "akayal")

        rent_data = read_input_file().result()
        rent_data = preprocess_record(rent_data).result()
        X, y, X_train, X_test, y_train, y_test = make_dataprep_model(rent_data, test_size=0.3, random_state=42).result()
        X_train, X_test, y_train, y_test = scale_data(X_train, X_test, y_train, y_test).result()
        ## doing LR
        lm, lm_prediction, metrics = model_lr(X_train, X_test, y_train, y_test).result()
        save_model(lm, "linear_regression")
        mlflow.log_param("LR metrices", metrics)
        ## doing decision tree
        dt, dt_prediction,metrics = model_dt(X_train, X_test, y_train, y_test).result()
        save_model(dt, "decision_tree")
        mlflow.log_param("decision_tree metrices", metrics)
        ## doing decision tree regressor
        best_dtc_estimator, mse_dtr, metrics = model_dt_regressor(X_train, X_test, y_train, y_test, random_state=100).result()
        save_model(best_dtc_estimator, "decision_tree_regressor")
        mlflow.log_param("decision_tree_regressor metrices", metrics)
        ## doing svm
        svr, svr_prediction, metrics = model_svm(X_train, X_test, y_train, y_test).result()
        save_model(svr, "svm")
        mlflow.log_param("svm metrices", metrics)
        ## doing random forerst
        rf, rf_prediction, metrics = model_rf(X_train, X_test, y_train, y_test, n_estimators=100).result()
        mlflow.log_param("random_forest metrices", metrics)
        save_model(rf, "random_forest")
        ## doing random forest regressor
        best_rfr_estimator, drfc_prediction, metrics = model_rf_regressor(X_train, X_test, y_train, y_test).result()
        save_model(best_rfr_estimator, "random_forest_regressor")
        mlflow.log_param("random_forest_regressor metrices", metrics)
        ## knn regressor
        best_KNN_estimator, KNN_test, metrics = model_knn_regressor(X_train, X_test, y_train, y_test).result()
        save_model(best_KNN_estimator, "knn_regressor")
        mlflow.log_param("knn_regressor metrices", metrics)
        ## voting regressor
        model_voting, voting_test, metrics = model_voting_regressor(X_train, X_test, y_train, y_test, best_KNN_estimator, best_dtc_estimator, best_rfr_estimator).result()
        save_model(model_voting, "voting_regressor")
        mlflow.log_param("voting_regressor metrices", metrics)
        mlflow.sklearn.log_model(model_voting, artifact_path="models/voting")
        # mlflow.log_artifacts(local_dir="artifacts")

        ## stacking regressor
        model_stacking, stacking_test, metrics = model_stacking_regressor(X_train, X_test, y_train, y_test, best_KNN_estimator, best_dtc_estimator, best_rfr_estimator).result()
        save_model(model_stacking, "stacking_regressor")
        mlflow.log_param("stacking_regressor metrices", metrics)
        mlflow.sklearn.log_model(model_stacking, artifact_path="models/stacking")
        # mlflow.log_artifacts(local_dir="artifacts")

        mlflow.register_model(model_uri=model_uri, name=model_name)
        print(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")

        client.transition_model_version_stage(
            name=model_name,
            version=1,
            stage="Production",
            archive_existing_versions=False,
        )



if __name__ == '__main__':
    run()
