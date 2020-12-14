#-------------------------------------------- Scope -------------------------------------------------------------
#
# - Consider folder structure
# - Read inercia2 de API:
#   + verano / invierno
#   + ultimos registros para todos los ps_id
#   + Status on/off (parada / arranque) --> from value of power from registros
# - Load log.csv
# - Read infoloc
#   + power_on
#   + power_off
# - x0
#   + read cli_id from command line
#   + read parameters from API
#   + feature selection --> take function(s) from preprocessing.py --> inercia2
#   + read last register from API for the feature selection --> no necesario
#   + read average power --> where to get average power from? --> inercia2 API ? ? ?
# - For each DT:
#   + load standardization csv
#   + load pc_eigenvectors csv
#   + load model fit
#   + standardize x0 --> x0_std
#   + pc_transform x0_std --> pc_x0
#   + load PC# from log file
#   + subset pc_x0
#   + predict(model, pc_x0)
#   + append prediction in dataframe
# - Dump dataframe to csv
#----------------------------------------------------------------------------------------------------------------

#---------------------------------------- Importing libraries ---------------------------------------------------
from bs4 import BeautifulSoup
import requests
import json
import xmltodict
import datetime
import re

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

# Estimators: regressors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn import linear_model

# Metrics:
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import max_error
from sklearn.model_selection import GridSearchCV

# Serilization
from joblib import dump, load

#=============== API: user data ================
apiUserData = pd.read_csv(os.path.join(os.getcwd(), "userdata.csv"), index_col= 0)
password = apiUserData.loc["password"].values[0]
user = apiUserData.loc["user"].values[0]
api_url = apiUserData.loc["api_url"].values[0]

#----------------------------- Fetching inercia2 -----------------------------
def read_registers(cli_id):
    inercia2_url = api_url + user + "/" + password + "/" + "inercia2?id=" + str(cli_id)
    # Sending request to the url
    response = requests.get(inercia2_url)
    # Parsing ordered dictionary with xmltodic
    inercia2_dic = xmltodict.parse(response.content)
    # Navigating down in the dictionary to get to the registers level
    inercia2_dic = inercia2_dic["info_sgclima"]["inercia2"]
    # Extracting x0
    registers = pd.DataFrame()
    for i, param in enumerate(inercia2_dic):
        ps_id = str(param["ps_id"])
        loc_id = str(param["loc_id"])
        registers.loc[0, ps_id + "_" + loc_id] = param["valor"]
    return registers.astype("float64")


def state_installation(registers, power_on, power_off, features_names):
    # Getting current potencia
    pattern_pot = "p\d+_"
    potencia_ps_id = False
    for feature in features_names:
        if re.search(pattern_pot, feature) is not None:
            potencia_ps_id = feature
    potencia_actual = ""
    if potencia_ps_id:
        potencia_actual = registers[potencia_ps_id[1:]][0]
    else:
        print("No feature marked with p")
    # Finding the current state of the installation
    if potencia_actual is not "":
        ver_inv = registers.filter(regex= ("ver-inv.*")).iloc[0,0]
        if ver_inv == 0:
            pattern = "ver_"
        else:
            pattern = "inv_"
        if potencia_actual < power_off:
            pattern = pattern + "arranque"
        else:
            if potencia_actual > power_on:
                pattern = pattern + "parada"
            else:
                print("Current state of installation is undefined")
                pattern = False
    else:
        pattern = False
    return pattern


def standardize_x0(registers, std_df):
    # set x0 based on registers and variables in std_df
    features_names = std_df.keys().to_list()
    x0 = pd.DataFrame()
    x0_std = pd.DataFrame()
    for feature in features_names:
        pattern = "\d+_\d+"
        psid_locid = re.search(pattern, feature)[0]
        print(feature, "\n", psid_locid)
        x0.loc[0, psid_locid] = registers.loc[0, psid_locid]
        x0_std.loc[0, psid_locid] = (registers.loc[0, psid_locid] - std_df.loc["mean", feature]) / std_df.loc["std", feature]
    return x0_std


def read_power_thresholds(cli_id):
    url = api_url + user + "/" + password + "/" + "infocli?id=" + str(cli_id)
    # Sending request to the url
    response = requests.get(url)
    # Parsing ordered dictionary with xmltodic
    infoloc_dic = xmltodict.parse(response.content)
    # Navigating down in the dictionary to get to the registers level
    infos = infoloc_dic["info_sgclima"]["infocli"]
    power_on = 50 #infos["pot_arranque"]
    power_off = 10 #infos["pot_parada"]
    return power_on, power_off


def feature_selection(cli_id, filepath_models):
    # Setting up where to look for the std_csv
    complete_path_model = os.path.join(os.getcwd(), filepath_models)
    file_list = os.listdir(complete_path_model)
    # Looking for the std_csv as per cli_id
    pattern_std = "\D+" + str(cli_id) + "_standardization_dt05"
    filename_std = False
    for f in file_list:
        if re.search(pattern_std, f) is not None:
            filename_std = f
    std_df = pd.DataFrame()
    if filename_std:
        std_df = pd.read_csv(os.path.join(complete_path_model, filename_std), index_col = 0)
    else:
        print(pattern_std, ": no standardization file available")
    if not std_df.empty:
        features_names = std_df.columns.to_list()
    else:
        features_names = False
    return features_names


def lookfor_models(pattern, dt, filepath_models):
    # Setting up where to look for the models
    complete_path_model = os.path.join(os.getcwd(), filepath_models)
    file_list = os.listdir(complete_path_model)
    # Looking for the models as per value of "pattern"
    pattern_fit = pattern + "_fit_dt" + dt
    pattern_std = pattern + "_standardization_dt" + dt
    filename_fit = False
    filename_std = False
    for f in file_list:
        if re.search(pattern_fit, f) is not None:
            filename_fit = f
        if re.search(pattern_std, f) is not None:
            filename_std = f
    fit = False
    std_df = pd.DataFrame()
    if filename_fit:
        fit = load(os.path.join(complete_path_model, filename_fit))
        std_df = pd.read_csv(os.path.join(complete_path_model, filename_std), index_col = 0)
    else:
        print(pattern_fit + dt, ": no fit available")
    return fit, std_df

#===================================== MAIN ==============================================================

def main(cli_id, loc_id, filepath_models, filepath_prediction):
    # Getting last registers from inercia2
    registers = read_registers(cli_id)
    # Getting power thresholds from infoloc
    power_on, power_off = read_power_thresholds(cli_id)
    # Getting the current state of the instalation
    features_names = feature_selection(cli_id, filepath_models)
    pattern = state_installation(registers, power_on, power_off, features_names)
    if pattern:
        pattern = pattern + str(loc_id)
        # Making 11 predicitons and writing csv
        var_index = ["05", "10", "15", "20", "25", "30", "35", "40", "45", "50", "55"]
        prediction = pd.DataFrame()
        for dt in var_index:
            # Looking for the model and standardization
            fit, std_df = lookfor_models(pattern, dt, filepath_models)
            # Getting x0_std
            if not std_df.empty:
                x0_std= standardize_x0(registers, std_df)
            else:
                print(pattern, "DT", dt, ": model not available")
            # Predicting
            if fit:
                print("predicting DT" + dt)
                prediction.loc[dt + "_min", pattern + "_DT"] = fit.predict(x0_std)
        # Dump prediction dataframe into csv
        now = datetime.now().strftime("%Y%m%d%H%M")
        prediction.to_csv(os.path.join(os.getcwd(), filepath_prediction,
                                       "prediction" + str(cli_id)+ "_" + str(now) + ".csv") , index=True, header=True)
    else:
        print("Current operating power not found")
        quit()
    return prediction

main(195, 195, r"..\test\models", r"..\test\prediction")