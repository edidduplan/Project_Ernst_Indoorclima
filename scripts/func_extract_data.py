#==============Importing libraries =====================
from bs4 import BeautifulSoup
import requests
import pandas as pd
import json
import xmltodict
import datetime
import os
import time

#================File path =====================
datafilenamepath = "" # File path where the extracted data will be copied

#=============== API: user data ================
password = ""
user = ""
api_url = ""

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
        ps_id = par["ps_id"]
        alias = par["alias"]
        registers_url = api_url + user + "/" + password + "/" + "registers?id=" + loc_id + "&ps_id=" + ps_id + "&data_ini=" + data_ini + "&data_fi=" + data_fi
        registers_response = requests.get(registers_url)
        registers_soup = BeautifulSoup(registers_response.text, 'lxml')
        col_names = ["valor", "data", "hora", "ps_id"]
        registers_df = pd.DataFrame()
        for i in col_names:
            registers_df[i] = [x.get_text() for x in registers_soup.find_all(i)]
        df_dic[alias] = registers_df
        print(str(n + 1), "/", str(len(parameters)))
    print("--- %s seconds ---" % (time.time() - start_time))
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
    # Writing csv
    df_merge.to_csv(os.path.join(
        datafilenamepath,
        (loc_id + "_df" + "_" + data_ini +"_to_" + data_fi + ".csv")), index=None, header=True)

#============== Getting the csv's ==============
extract_data("195", "2019-01-01", "2019-09-30")
extract_data("96", "2015-01-01", "2020-02-19")
extract_data("508", "2015-01-01", "2020-02-19")
extract_data("195", "2015-01-01", "2018-12-31")
extract_data("195", "2020-02-01", "2020-02-29")