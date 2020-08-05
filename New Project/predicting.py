#-------------------------------------------- Scope -------------------------------------------------------------
#
# - Consider folder structure
# - Read inercia2 de API:
#   + verano / invierno
#   + x0: registros para ps_id seleccionados (feature selection)
#   + Status on/off (parada / arranque) --> from value of power from registros
# - Load log.csv
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
password = "jFC4u5jj3megxXp"
user = "uk9JVfRAyK"
#user = "uk9JVfRAhT"
api_url = "http://sgclima.indoorclima.com/api.php/"

#----------------------------- Fetching inercia2 -----------------------------
def read_inercia2(cli_id):
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
        ps_id = "p_" + str(param["ps_id"])
        registers.loc[0, ps_id] = param["valor"]
    return registers

def read_infoloc():
    infoloc_url = api_url + user + "/" + password + "/" + "infoloc"
    # Sending request to the url
    response = requests.get(infoloc_url)
    # Parsing ordered dictionary with xmltodic
    infoloc_dic = xmltodict.parse(response.content)
    # Navigating down in the dictionary to get to the registers level
    infoloc_dic = infoloc_dic["info_sgclima"]["infoloc"]
    # For loop to construct df based on "keys"
    infoloc_df = pd.DataFrame()
    for i, dic in enumerate(infoloc_dic):
        for key, value in dic.items():
            infoloc_df.loc[i, str(key)] = value
    return infoloc_df

def status_on_off(registers, infoloc, cli_id):
    ps_id_pot_posible = {"p_900"}
    registers_cols = set(registers.columns)
    ps_id_pot = registers_cols.intersection(ps_id_pot_posible)
    pot_0 = registers[list(ps_id_pot)[0]]
    pot_arranque = infoloc.loc[str(cli_id), "pot_arranque"]
    pot_parada = infoloc.loc[str(cli_id), "pot_parada"]
    if pot_0 >= pot_arranque:
        on_off = 1
    else:
        on_off = 0
    return on_off


def get_x0(registers, inercia):
    # - Are the ps_id's from inercia 2 the feature selection sub-set?
    # - Look for pot_promedio en inercia ?
    # - Concatenate feature selection with pot_promedio. Features most be sorted the same as in the training set.
    # Complete function
    return x0

def standardize_x0(x0, standard_csv):
    std_df = pd.read_csv(standard_csv)
    x0_std = (x0 - std_df.loc["mean"]) / std_df.loc["std"]
    return x0_std

def main(cli_id):
    # Getting parameters and x0
    registers = read_inercia2(cli_id)
    infoloc = read_infoloc()
    ps_id_potencia = {900}
    ver_inv = registers["p_ver-inv"]
    x0 = get_x0(registers)
    on_off = status_on_off(registers, infoloc, cli_id)
    # --------------------------------------------------
    # Set working directory
    if on_off == 1:
        if ver_inv == 1:
            fit_file_path = os.path.join("parada", "verano", "fit")
        else:
            fit_file_path = os.path.join("parada", "invierno", "fit")
    else:
        if ver_inv == 1:
            fit_file_path = os.path.join("arranque", "verano", "fit")
        else:
            fit_file_path = os.path.join("arranque", "invierno", "fit")
    # --------------------------------------------------
    # For each DTx:
    var_index = ["05", "10", "15", "20", "25", "30", "35", "40", "45", "50", "55", "60"]
    # Creates prediction dataframe
    prediction = pd.DataFrame()
    for i in var_index:
        # Load model fit
        fit = load(os.path.join(fit_file_path, "fit_dt" + i))
        # Standardize x0
        standard_csv = pd.read_csv(os.path.join(fit_file_path, "standardization_dt" + i))
        x0_std = standardize_x0(x0, standard_csv)
        # Predict on x0_std
        prediction.loc["y0_hat", "DT" + i] = fit.predict(x0_std)
    # Dump prediction dataframe into csv
    prediction.to_csv("prediction.csv", index= True, header= True)
    return prediction


