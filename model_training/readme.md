# Model Training and Register



## Model Training

Here is how model can be trained once we have the EC2 instance created as per guide from ML Zoomcamp and then conda instance created as per requirements.txt.

Model can be created as shared below.

```
(mlflow-mlops) ubuntu@ip-172-31-28-166:~/mlops-zoomcamp/07-project/capstone/src$ python train.py
```

Here is the output that will be there as part of training and morning the task into prefect. The artefact url

```

2022/09/09 15:52:41 INFO mlflow.store.db.utils: Creating initial MLflow database tables...
2022/09/09 15:52:41 INFO mlflow.store.db.utils: Updating database tables
INFO  [alembic.runtime.migration] Context impl SQLiteImpl.
INFO  [alembic.runtime.migration] Will assume non-transactional DDL.
INFO  [alembic.runtime.migration] Running upgrade  -> 451aebb31d03, add metric step
INFO  [alembic.runtime.migration] Running upgrade 451aebb31d03 -> 90e64c465722, migrate user column to tags
INFO  [alembic.runtime.migration] Running upgrade 90e64c465722 -> 181f10493468, allow nulls for metric values
INFO  [alembic.runtime.migration] Running upgrade 181f10493468 -> df50e92ffc5e, Add Experiment Tags Table
INFO  [alembic.runtime.migration] Running upgrade df50e92ffc5e -> 7ac759974ad8, Update run tags with larger limit
INFO  [alembic.runtime.migration] Running upgrade 7ac759974ad8 -> 89d4b8295536, create latest metrics table
INFO  [89d4b8295536_create_latest_metrics_table_py] Migration complete!
INFO  [alembic.runtime.migration] Running upgrade 89d4b8295536 -> 2b4d017a5e9b, add model registry tables to db
INFO  [2b4d017a5e9b_add_model_registry_tables_to_db_py] Adding registered_models and model_versions tables to database.
INFO  [2b4d017a5e9b_add_model_registry_tables_to_db_py] Migration complete!
INFO  [alembic.runtime.migration] Running upgrade 2b4d017a5e9b -> cfd24bdc0731, Update run status constraint with killed
INFO  [alembic.runtime.migration] Running upgrade cfd24bdc0731 -> 0a8213491aaa, drop_duplicate_killed_constraint
INFO  [alembic.runtime.migration] Running upgrade 0a8213491aaa -> 728d730b5ebd, add registered model tags table
INFO  [alembic.runtime.migration] Running upgrade 728d730b5ebd -> 27a6a02d2cf1, add model version tags table
INFO  [alembic.runtime.migration] Running upgrade 27a6a02d2cf1 -> 84291f40a231, add run_link to model_version
INFO  [alembic.runtime.migration] Running upgrade 84291f40a231 -> a8c4a736bde6, allow nulls for run_id
INFO  [alembic.runtime.migration] Running upgrade a8c4a736bde6 -> 39d1c3be5f05, add_is_nan_constraint_for_metrics_tables_if_necessary
INFO  [alembic.runtime.migration] Running upgrade 39d1c3be5f05 -> c48cb773bb87, reset_default_value_for_is_nan_in_metrics_table_for_mysql
INFO  [alembic.runtime.migration] Running upgrade c48cb773bb87 -> bd07f7e963c5, create index on run_uuid
INFO  [alembic.runtime.migration] Context impl SQLiteImpl.
INFO  [alembic.runtime.migration] Will assume non-transactional DDL.
2022/09/09 15:52:41 INFO mlflow.tracking.fluent: Experiment with name 'house-rent-prediction-experiment' does not exist. Creating a new experiment.
File read done. Displaying record stats
               BHK          Rent         Size     Bathroom
count  4746.000000  4.746000e+03  4746.000000  4746.000000
mean      2.083860  3.499345e+04   967.490729     1.965866
std       0.832256  7.810641e+04   634.202328     0.884532
min       1.000000  1.200000e+03    10.000000     1.000000
25%       2.000000  1.000000e+04   550.000000     1.000000
50%       2.000000  1.600000e+04   850.000000     2.000000
75%       3.000000  3.300000e+04  1200.000000     2.000000
max       6.000000  3.500000e+06  8000.000000    10.000000
preprocessing of records done. Displaying record stats
               BHK          Rent         Size     Bathroom
count  4746.000000  4.746000e+03  4746.000000  4746.000000
mean      2.083860  3.499345e+04   967.490729     1.965866
std       0.832256  7.810641e+04   634.202328     0.884532
min       1.000000  1.200000e+03    10.000000     1.000000
25%       2.000000  1.000000e+04   550.000000     1.000000
50%       2.000000  1.600000e+04   850.000000     2.000000
75%       3.000000  3.300000e+04  1200.000000     2.000000
max       6.000000  3.500000e+06  8000.000000    10.000000
data prep for modelling done
column names of Training Data:['BHK', 'Size', 'Bathroom', 'Area Type_Built Area', 'Area Type_Carpet Area', 'Area Type_Super Area', 'City_Bangalore', 'City_Chennai', 'City_Delhi', 'City_Hyderabad', 'City_Kolkata', 'City_Mumbai', 'Furnishing Status_Furnished', 'Furnishing Status_Semi-Furnished', 'Furnishing Status_Unfurnished', 'Tenant Preferred_Bachelors', 'Tenant Preferred_Bachelors/Family', 'Tenant Preferred_Family', 'Point of Contact_Contact Agent', 'Point of Contact_Contact Builder', 'Point of Contact_Contact Owner']
column names of Test Data:['BHK', 'Size', 'Bathroom', 'Area Type_Built Area', 'Area Type_Carpet Area', 'Area Type_Super Area', 'City_Bangalore', 'City_Chennai', 'City_Delhi', 'City_Hyderabad', 'City_Kolkata', 'City_Mumbai', 'Furnishing Status_Furnished', 'Furnishing Status_Semi-Furnished', 'Furnishing Status_Unfurnished', 'Tenant Preferred_Bachelors', 'Tenant Preferred_Bachelors/Family', 'Tenant Preferred_Family', 'Point of Contact_Contact Agent', 'Point of Contact_Contact Builder', 'Point of Contact_Contact Owner']
<class 'pandas.core.series.Series'>
<class 'numpy.ndarray'>
Data scaling done
LR model training completed. Model Evaluation metrics are:
=============MAE: 1.2562668981276902e+16
=============MSE: 4.286829407665826e+32
=============RMSE: 2.070465988048542e+16
linear_regression Model save completed
Decision Tree model training completed. Model Evaluation metrics are:
=============MAE: 17645.00341510861
=============MSE: 9569653036.682587
=============RMSE: 97824.60343227866
decision_tree Model save completed
Starting Decision Tree regressor:
DecisionTreeRegressor(max_depth=6, min_samples_leaf=2, min_samples_split=5,
                      random_state=100)
10-CV Best Parameters = {'max_depth': 6, 'min_samples_leaf': 2, 'min_samples_split': 5}
10-CV Best Score = -17893.78280990588
DecisionTreeRegressor(max_depth=6, min_samples_leaf=2, min_samples_split=5,
                      random_state=100)
Completed Decision Tree regressor:
=============MAE: 15070.97785729306
=============MSE: 2836149544.4295263
=============RMSE: 53255.511869003065
decision_tree_regressor Model save completed
SVM model training completed. Model Evaluation metrics are:
=============MAE: 24102.903580417125
=============MSE: 3842418546.384684
=============RMSE: 61987.24502980177
svm Model save completed
Random Forest model training completed. Model Evaluation metrics are:
=============MAE: 15076.751181182064
=============MSE: 4396499788.526886
=============RMSE: 66306.10672122806
random_forest Model save completed
Starting Random Forest Tree regressor:
RandomForestRegressor(n_estimators=300, random_state=100)
10-CV Best Parameters = {'min_samples_leaf': 1, 'n_estimators': 300}
10-CV Best Score = -39403.557579393375
RandomForestRegressor(n_estimators=300, random_state=100)
Completed Random Forest regressor:
=============MAE: 15313.184948624505
=============MSE: 4463828342.315149
=============RMSE: 66811.8877320133
random_forest_regressor Model save completed
Starting KNN regressor:
KNeighborsRegressor(weights='distance')
10-CV Best Parameters = {'n_neighbors': 5, 'weights': 'distance'}
10-CV Best Score = -26040.416154533475
KNeighborsRegressor(weights='distance')
Completed KNN regressor:
=============MAE: 13675.078542921305
=============MSE: 1760639791.5849617
=============RMSE: 41959.97845072089
knn_regressor Model save completed
Starting voting regressor:
Completed Voting regressor:
[ 16978.03476435  15087.25936713  38881.26458199 ...  18460.39150742
 204864.79445386  18068.59480755]
Completed Voting regressor:
=============MAE: 13675.12294799564
=============MSE: 2046610787.5088143
=============RMSE: 45239.48261760754
voting_regressor Model save completed
Starting stacking regressor:
Completed Stacking regressor:
StackingRegressor(estimators=[('knn', KNeighborsRegressor(weights='distance')),
                              ('dtc',
                               DecisionTreeRegressor(max_depth=6,
                                                     min_samples_leaf=2,
                                                     min_samples_split=5,
                                                     random_state=100)),
                              ('rfc',
                               RandomForestRegressor(n_estimators=300,
                                                     random_state=100))])
Completed Voting regressor:
=============MAE: 14731.61487217739
=============MSE: 2401222424.8391533
=============RMSE: 49002.26958865429
stacking_regressor Model save completed
Successfully registered model 'house-rent-prediction-model'.
2022/09/09 15:54:01 INFO mlflow.tracking._model_registry.client: Waiting up to 300 seconds for model version to finish creation.                     Model name: house-rent-prediction-model, version 1
Created version '1' of model 'house-rent-prediction-model'.
default artifacts URI: './mlruns/1/6b5ab9f1fd2e4e3d985ca3e590840d3b/artifacts'
```



## Model Register
I'm using a sklearn library, mlflow provides a way to register the model with the following command:

```
        ## stacking regressor
        model_stacking, stacking_test, metrics = model_stacking_regressor(X_train, X_test, y_train, y_test, best_KNN_estimator, best_dtc_estimator, best_rfr_estimator)
        save_model(model_stacking, "stacking_regressor")
        mlflow.log_param("stacking_regressor metrices", metrics)
        mlflow.sklearn.log_model(model_stacking, artifact_path="models/stacking")
        # mlflow.log_artifacts(local_dir="artifacts")

        mlflow.register_model(model_uri=model_uri, name=model_name)
```
