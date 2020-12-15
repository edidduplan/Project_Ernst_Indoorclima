#-------------------------------------------- Scope -------------------------------------------------------------
#
# - Read dataset: "arranque.csv" / "parada.csv" and ver./inv., each one containing DT05, ..., DT60
#   (12 dependent variables)
# - For each DTx:
#       + Extract features and extract DTx as dependent variable
#		+ Standardize features
# 		+ Assessment of pre-selected models: SVR, KNN and RF
#			- For SVR and KNN
#				* Hyperparameter selection including PCA and metric in nested cv
#			- For RF
#				* Hyperparameter selection and metric in nested cv
#       + Select best model based on min. MSE
#       + Perform hyperparameter tunning for selected model on whole data
#       + Train model
#       + write fitted model do file
#----------------------------------------------------------------------------------------------------------------

#---------------------------------------- Importing libraries ---------------------------------------------------
import pandas as pd
import numpy as np
from datetime import datetime
import os
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from numpy import array, mean, cov
from numpy.linalg import eig
from sklearn.preprocessing import StandardScaler
import scipy
import time
from math import sqrt
import sys
import re

# Estimators: regressors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.pipeline import Pipeline

# Metrics:
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, KFold
from sklearn.metrics import max_error
from sklearn.model_selection import GridSearchCV

# Serilization
from joblib import dump, load

#------------------------------------- Extract variables --------------------------------------------------------------
def extract_features(data, p):
    # features = df.drop(list(df.filter(regex='DT\d\d')), axis =1)
    df = data.copy()
    df[df.shift(periods=-1, axis=1).isnull()] = np.nan
    features = df[list(df.columns.drop(list(df.filter(regex='P\d\d|DT\d\d')))) + [p]].dropna()
    # features = df[list(df.columns.drop(list(df.filter(regex='P\d\d|DT\d\d'))))].dropna()
    features.drop([p], axis= "columns", inplace= True)
    return features

def extract_dependent(df, dt, features):
    dependent = df[dt].loc[features.index]
    return dependent

#------------------------------------- PC transformation ------------------------------------------------

def standardize(features):
    mean = features.describe().loc["mean"]
    std = features.describe().loc["std"]
    features_std = (features - mean) / std
    return features_std


#------- Model assessment functions with nested Cross-validation including hyperparameter tunning----------

def get_ncv_metric(model, X, y, n_jobs, **kwargs):
    # Establisch pipeline: PCA -> model
    estimators = [('reduce_dim', PCA()), ('regressor', model)]
    pipe = Pipeline(estimators)
    # Grid search CV
    n_components = np.arange(1, len(X.columns) + 1)
    param_grid = {"reduce_dim__n_components": n_components}
    for key, value in kwargs.items():
        param_grid["regressor__" + str(key)] = value
    kf_10 = KFold(n_splits=10, shuffle=True, random_state= 42) # n_splits changed to 5 for testing
    grid = GridSearchCV(
        pipe,
        param_grid,
        scoring= "neg_mean_squared_error",
        n_jobs= n_jobs,
        cv = kf_10,
        error_score= 0,
        verbose= 1
    )
    kf_10_nested = KFold(n_splits=10 , shuffle=True, random_state= 13) # n_splits changed to 5 for testing
    nested_cv = cross_val_score(
        estimator= grid,
        X= X,
        y= y,
        scoring="neg_mean_squared_error",
        cv= kf_10_nested,
        n_jobs= n_jobs,
        verbose= 1
    )
    return nested_cv

def get_ncv_metric_rf(model, X, y, n_jobs, **kwargs):
    # Grid search CV
    param_grid = {}
    for key, value in kwargs.items():
        param_grid[str(key)] = value
    kf_10 = KFold(n_splits=10, shuffle=True, random_state= 42) # n_splits changed to 5 for testing
    grid = GridSearchCV(
        model,
        param_grid,
        scoring= "neg_mean_squared_error",
        n_jobs= n_jobs,
        cv = kf_10,
        error_score= 0,
        verbose= 1
    )
    kf_10_nested = KFold(n_splits=10, shuffle=True, random_state= 13) # n_splits changed to 5 for testing
    nested_cv = cross_val_score(
        estimator= grid,
        X= X,
        y= y,
        scoring="neg_mean_squared_error",
        cv= kf_10_nested,
        n_jobs= n_jobs,
        verbose= 1
    )
    return nested_cv

#----------------------------- Train functions ----------------------

def train_hyp_select(model, X, y, n_jobs, **kwargs):
    # Establisch pipeline: PCA -> model
    estimators = [('reduce_dim', PCA()), ('regressor', model)]
    pipe = Pipeline(estimators)
    # Grid search CV
    n_components = np.arange(1, len(X.columns) + 1)
    param_grid = {"reduce_dim__n_components": n_components}
    for key, value in kwargs.items():
        param_grid["regressor__" + str(key)] = value
    kf_10 = KFold(n_splits=10, shuffle=True, random_state= 42) # n_splits changed to 5 for testing
    grid = GridSearchCV(
        pipe,
        param_grid,
        scoring= "neg_mean_squared_error",
        n_jobs= n_jobs,
        cv = kf_10,
        error_score= 0,
        verbose= 1,
        refit= True
    )
    return grid.fit(X, y)

def train_hyp_select_rf(model, X, y, n_jobs, **kwargs):
    # Grid search CV
    param_grid = {}
    for key, value in kwargs.items():
        param_grid[str(key)] = value
    kf_10 = KFold(n_splits=10, shuffle=True, random_state=42) # n_splits changed to 5 for testing
    grid = GridSearchCV(
        model,
        param_grid,
        scoring="neg_mean_squared_error",
        n_jobs=n_jobs,
        cv=kf_10,
        error_score=0,
        verbose=1,
        refit= True
    )
    return grid.fit(X, y)

#--------------------- Fit function --------------------------------------------------

def fit_writecsv(data, file_name, file_path):

    # Loading data
    # data = pd.read_csv(csv)
    # Creating model variables
    svr_rbf = SVR(kernel='rbf')
    knn = KNeighborsRegressor(weights = "distance")
    rf = RandomForestRegressor(criterion='mse', max_depth=None, min_samples_split=2)
    gbtr = GradientBoostingRegressor()
    np.random.seed(31415)
    var_index = ["05", "10", "15", "20", "25", "30", "35", "40", "45", "50", "55"]
    # Creates log file
    log = pd.DataFrame(columns= ["model", "ncv_mse"], index= [var_index])
    for i in var_index:
        print("DT" + i)
        # X and Y
        features = extract_features(data, "P" + i)
        dependent = extract_dependent(data, "DT" + i, features)
        # Standardize
        features_std = standardize(features)
        #---------------------------------------------
        # Model assessment (incl. hyperparameter tunning) with 10-fold outer 10-fold inner cv
        # 0) KNN
        neighbors = [9, 11, 13, 15, 17]
        power = [1, 2, 3]
        ncv_knn = get_ncv_metric(knn, features_std, dependent, 2, p= power, n_neighbors= neighbors)
        mse_mean = [abs(ncv_knn.mean())]
        mse_std = [ncv_knn.std()]
        # 1) SVR radial basis function
        c = [0.1, 1, 10, 100]
        g = [0.1, 1, 10, "auto", "scale"]
        ncv_svr = get_ncv_metric(svr_rbf, features_std, dependent, 2, C= c, gamma= g)
        mse_mean.append(abs(ncv_svr.mean()))
        mse_std.append(ncv_svr.std())
        # 2) RF
        m = ["auto", "sqrt"]
        trees = [200, 400, 500]
        ncv_rf = get_ncv_metric_rf(rf, features_std, dependent, 2, max_features=m, n_estimators= trees)
        mse_mean.append(abs(ncv_rf.mean()))
        mse_std.append(ncv_rf.std())
        #----------------------------------------------
        # Model selection
        #               0       1        2
        model_list = ["KNN", "SVR_rbf", "RF"]
        best_model = np.argmin(mse_mean)
        best_model_mse = mse_mean[best_model]
        best_model_mse_std = mse_std[best_model]
        #-------------------------------------------------
        # Filling metrics in log file
        log.loc[i, "model"] = model_list[best_model]
        log.loc[i, "ncv_mse"] = best_model_mse
        log.loc[i, "ncv_std"] = best_model_mse_std
        #-------------------------------------------------
        # Hyperparameter tunning of best model and train in whole data
        if best_model == 0:
            fit = train_hyp_select(knn, features_std, dependent, 2, n_neighbors= neighbors, p= power)
        else:
            if best_model == 1:
                fit = train_hyp_select(svr_rbf, features_std, dependent, 2, C= c, gamma= g)
            else:
                fit = train_hyp_select_rf(rf, features_std, dependent, 2, max_features=m, n_estimators= trees)
        # Log best parameters
        for key, value in fit.best_params_.items():
            log.loc[i, key] = value
        #----------------------------------------------------
        # Serialize model fit to file
        dump(fit, os.path.join(file_path, file_name + "_fit_dt" + i))
        # Dump standardization to csv
        standardization = features.describe().loc[["mean", "std"]]
        standardization.to_csv(os.path.join(file_path, file_name + "_standardization_dt" + i + ".csv"), index= True, header= True)
    now = datetime.now().strftime("%Y%m%d%H%M")
    log.to_csv(os.path.join(file_path, file_name + "_log_" + str(now) + ".csv"), index= True, header= True)

    return log

#===================================    Main   ==========================================

#-----------------------------------------
def main(loc_id, filepath_data, filepath_model):
    start_time = time.time()
    complete_path_data = os.path.join(os.getcwd(), filepath_data)
    file_list = os.listdir(complete_path_data)
    # Looping on the 4 subsets
    subset_list = ["ver_arranque", "ver_parada", "inv_arranque", "inv_parada"]
    for subset in subset_list:
        pattern = subset + str(loc_id)
        csv = False
        for i in file_list:
            if re.search(pattern, i) is not None:
                csv = i
        if csv:
            data = pd.read_csv(os.path.join(complete_path_data, csv))
            fit_writecsv(data, pattern, filepath_model)
        else:
            print(pattern, ": no csv available")
    print("--- %s seconds ---" % (time.time() - start_time))

#--------------------------------- Excecution ---------------------------------------------
if __name__ == "__main__":
    # main(sys.argv[1], sys.argv[2], sys.argv[3])
    main(195, r"..\test\data", r"..\test\models")

#--------------------------------- Test -----------------------------