# This second version of func_extract_data filters the parameters to be extracted based on rules, 
# in order to read only those relevant for the machine learning algos.

#==============Importing libraries =====================
from bs4 import BeautifulSoup
import requests
import pandas as pd
import json
import xmltodict
import datetime
import os
import time
import re

#================File path =====================
datafilenamepath = "/Data" # File path where the extracted data will be copied

#=============== API: user data ================
apiUserData = pd.read_csv(os.path.join(os.getcwd(), "scripts", "userdata.csv"), index_col= 0)
password = apiUserData.loc["password"].values[0]
user = apiUserData.loc["user"].values[0]
api_url = apiUserData.loc["api_url"].values[0]

#============== Function: extract data ==============
def extract_data(loc_id, data_ini, data_fi):
    parameters_url = api_url + user + "/" + password + "/" + "parameters?id=" + loc_id
    # Sending request to the url
    parameters_response = requests.get(parameters_url)
    # Parsing ordered dictionary with xmltodic
    parameters_dic = xmltodict.parse(parameters_response.content)
    # Navigating down in the dictionary to get to the registers level
    parameters = parameters_dic["info_sgclima"]["parameter"]
    # Registers: parsing of registers for all parameters
    start_time = time.time()
    df_dic = {}
    for n, par in enumerate(parameters):
        # if {par["alias"]}.intersection(para_to_drop) == set():
        # Loading attributes from parameters
        ps_id = str(par["ps_id"])
        familia = str(par["familia"])
        subfamilia = str(par["subfamilia"])
        origen = str(par["origen"])
        objetivo = str(par["objetivo"])
        # Rules to select variables
        origen_rule = re.findall("_maq|instalacion", str(par["origen"]))[0] if re.findall("_maq|instalacion", str(par["origen"])) else ""
        rule1 = {familia}
        rule2 = {familia + subfamilia}
        rule3 = {familia + subfamilia + origen_rule}
        rule4 = {familia + subfamilia + origen_rule + ps_id}
        rule5 = {objetivo}
        # Set of rules to select only relevant parameters: features
        rules = {"Controlmodo_func", "Controlcompuerta_maq", "potenciaelectrica_activainstalacion1",
                 "potenciaelectrica_activainstalacion900", "potenciaelectrica_activa_maq", "presiones",
                 "temperaturasexterior", "temperaturasimpulsion_maq", "temperaturasretorno_maq", "1"}
        if (rule1.intersection(rules) != set()) | \
                (rule2.intersection(rules) != set()) | \
                (rule3.intersection(rules) != set()) | \
                (rule4.intersection(rules) != set()) | \
                (rule5.intersection(rules) != set()):
            # Registers
            registers_url = api_url + user + "/" + password + "/" + "registers?id=" + loc_id + "&ps_id=" + ps_id + "&data_ini=" + data_ini + "&data_fi=" + data_fi
            registers_response = requests.get(registers_url)
            registers_soup = BeautifulSoup(registers_response.text, 'lxml')
            col_names = ["valor", "data", "hora", "ps_id"]
            registers_df = pd.DataFrame()
            for i in col_names:
                registers_df[i] = [x.get_text() for x in registers_soup.find_all(i)]
            # Naming variables
            df_dic[familia + "_" + subfamilia + "_" + origen + "_" + ps_id + "_" + objetivo] = registers_df
        print(str(n + 1), "/", str(len(parameters)))
    # Changing data types and generating datetime column
    for i, df in df_dic.items():
        df["datetime"] = df["data"] + df["hora"]
        df["datetime"] = pd.to_datetime(df["datetime"], format="%Y-%m-%d%H:%M:%S")
        df_dic[i] = df.drop(["data", "hora", "ps_id"], axis=1)
    # Renaming the "valor" column in the dataframes
    for i, df in df_dic.items():
        df_dic[i] = df_dic[i].rename(columns={"valor": i})
    df_names = list(df_dic.keys())
    # Merging dataframes on "datetime"
    df_merge = df_dic[df_names[0]]
    for i, dic in enumerate(df_dic.items()):
        if i > 0:
            df_merge = pd.merge(df_merge, dic[1], how="outer", on="datetime")
    # Sorting df on datetime
    df_merge = df_merge.sort_values("datetime")
    df_merge.set_index("datetime", inplace= True)
    # Writing csv
    #df_merge.to_csv(os.path.join(
    #    datafilenamepath,
    #    (loc_id + "_df" + "_" + data_ini +"_to_" + data_fi + ".csv")), index=None, header=True)
    print("--- %s seconds ---" % (time.time() - start_time))
    return df_merge

#============== Getting the csv's ==============
extract_data("195", "2019-01-01", "2019-09-30")
extract_data("96", "2015-01-01", "2020-02-19")
extract_data("508", "2015-01-01", "2020-02-19")
extract_data("195", "2015-01-01", "2018-12-31")
extract_data("195", "2020-02-01", "2020-02-29")
extract_data("508", "2020-02-01", "2020-02-01")
df_FuncExtrData2= extract_data("195", "2020-01-01", "2020-01-31")
