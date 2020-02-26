#==============Importing libraries =====================
from bs4 import BeautifulSoup
import requests
import pandas as pd
import json
import xmltodict
import datetime
import os
import time
#========================================================
datafilenamepath = "C:\\Users\\edidd\\Documents\\Ubiqum\\Data Analytics Course\\Project_Heike2\\Data"

#---------------- API: user data ------------------
password = "jFC4u5jj3megxXp"
user = "uk9JVfRAhT"
api_url = "https://sgclima.indoorclima.com/api.php/"

#================ Parameters ===============

#---------- Defining location -----------------
loc_id = "195"
#---------------------------------------------------------

parameters_url = api_url + user + "/" + password + "/" + "parameters?id=" + loc_id

# Sending request to the url
parameters_response = requests.get(parameters_url)

#Parsing ordered dictionary with xmltodic
parameters_dic = xmltodict.parse(parameters_response.content)

#--------- For loop to construct df based on "keys" ---------

# Navigating down in the dictionary to get to the registers level
parameters = parameters_dic["info_sgclima"]["parameter"]

parameters_df_195 = pd.DataFrame()
for i, dic in enumerate(parameters):
	for key, value in dic.items():
		parameters_df_195.loc[i, str(key)] = value

# Dropping parameters not needed for the model
parameters[0]["alias"]

#Checking NAs
parameters_df_195.isnull().any().any()

#Include if/else in order to account for the possibility of NAs

#=============== Registers: parsing of registers for all parameters ===========
# Setting dates
data_ini = "2019-01-01"
data_fi = "2019-09-30"

para_to_drop = {"Paro - marcha", "Alarma", "Algoritmos", "Rooftop", "Energía Contador", "CO2", "Potencia General",
				"Tª SET invierno", "Tª SET verano"}

para_set = set(parameters_df["alias"])

# For loop to parse multiple parameters and generate corresponding dataframes

start_time = time.time()
df_dic = {}
for par in parameters:
	# if {par["alias"]}.intersection(para_to_drop) == set():
		ps_id = par["ps_id"]
		alias = par["alias"]
		registers_url = api_url + user + "/" + password + "/" + "registers?id=" + loc_id + "&ps_id=" + ps_id + "&data_ini=" + data_ini + "&data_fi=" + data_fi
		registers_response = requests.get(registers_url)
		registers_dic = xmltodict.parse(registers_response.content)
		registers = registers_dic["info_sgclima"]["registers"]
		registers_df = pd.DataFrame()
		for i, dic in enumerate(registers):
			for key, value in dic.items():
				registers_df.loc[i, str(key)] = value
		df_dic[alias] = registers_df
print("--- %s seconds ---" % (time.time() - start_time))

# Alternative procedure for parsing parameters' registers to dataframes (BS4)
start_time = time.time()
df_dic = {}
for par in parameters:
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
print("--- %s seconds ---" % (time.time() - start_time))

# Checking for NAs in the parsed df's
for i, dic in df_dic.items():
	print(dic.isnull().any().any())

# Copying all dataframes to csv's
for i, df in df_dic.items():
	df.to_csv(os.path.join(
		datafilenamepath,
		(loc_id + "_" + i + "_" + data_ini +"_to_" + data_fi + ".csv")),
		index = None,
		header=True)

#Changing data types and generating datetime column
for i, df in df_dic.items():
	df["valor"] = df["valor"].apply(float)
	df["datetime"] = df["data"] + df["hora"]
	df["datetime"] = pd.to_datetime(df["datetime"], format="%Y-%m-%d%H:%M:%S")
	df_dic[i]["ps_id"] = df["ps_id"].apply(str)
	df_dic[i] = df.drop(["data", "hora", "ps_id"], axis= 1)

#Renaming the "valor" column in the dataframes
for i, df in df_dic.items():
	df_dic[i] = df_dic[i].rename(columns = {"valor" : i})

df_names = list(df_dic.keys())

# Merging dataframes on "datetime"
# test = pd.merge(df_dic["Rooftop"], df_dic[df_names[22]], how= "outer", on= "datetime")
# test_2 = pd.merge(test, df_dic[df_names[4]], how= "outer", on= "datetime")

df_397 = df_dic[df_names[0]]
for i, dic in enumerate(df_dic.items()):
	if i > 0:
		df_397 = pd.merge(df_397, dic[1], how="outer", on="datetime")

# Sorting df on datetime
df_397 = df_397.sort_values("datetime")

df_397.to_csv(os.path.join(datafilenamepath, "397_df_2019-01_to_2019-09.csv"), index = None, header=True)

# Adding other time variables
# df_2020_01["hour"] = df_2020_01["datetime"].dt.hour
#
# df_2020_01.info()

# =========== Parsing infoloc ===================
infoloc_url = api_url + user + "/" + password + "/" + "infoloc"

# Sending request to the url
infoloc_response = requests.get(infoloc_url)

# Parsing ordered dictionary with xmltodic
infoloc_dic = xmltodict.parse(infoloc_response.content)

# For loop to construct df based on "keys"

# Navigating down in the dictionary to get to the registers level
infoloc = infoloc_dic["info_sgclima"]["infoloc"]

infoloc_df = pd.DataFrame()
for i, dic in enumerate(infoloc):
	for key, value in dic.items():
		infoloc_df.loc[i, str(key)] = value

# Checking NAs
infoloc_df.isnull().any().any()

# Drop "tipo_pr" because of NAs
infoloc_df = infoloc_df.drop("tipo_pr", axis= "columns")

# Write infoloc to csv
infoloc_df.to_csv(os.path.join(datafilenamepath, "infoloc.csv"), index = None, header=True)






#============= Old (archive) ========================
#================ Registers ==========================
ps_id = "13" # T.exterior

registers_url = api_url + user + "/" + password + "/" + "registers?id=" + loc_id + "&ps_id=" + ps_id + "&data_ini=" + data_ini + "&data_fi=" + data_fi

# Sending request to the url
registers_response = requests.get(registers_url)

# Writing df with Beautifulsoup
# The following method is only reliable if all dictionaries (of each register) have the same keys in the same order.

registers_soup = BeautifulSoup(registers_response.text, 'lxml')

registers_df = pd.DataFrame()

col_names = ["valor", "data", "hora", "ps_id"]

for i in col_names:
	registers_df[i] = [x.get_text() for x in registers_soup.find_all(i)]

#--------- Try something...
print(registers_soup.get_text())

#--------------

#--------- Creating dictionary from XML ----------
# This methos is slower but doesn't need the dictionaries to have the same structure (like in the previous method).

#Parsing ordered dictionary with xmltodic
registers_dic = xmltodict.parse(registers_response.content)

# Creating json file (but not neccesary for final approach)
# with open("registers.json", "w") as f:
# 	json.dump(registers_dic, f, indent = 2)

# Trying different things with sample dictionary
# with open("registers.json") as f:
# 	sample_dic = json.load(f)
#
# sample_dic["info_sgclima"]["registers"]

#--------- For loop to construct df based on "keys" ---------

# Navigating down in the dictionary to get to the registers level
registers = registers_dic["info_sgclima"]["registers"]

# len(registers_dic["info_sgclima"]["registers"])

# registers[0].items()

# Checking whether the keys of 2 dics are the same
# keys = set(registers[0]).intersection(set(registers[1]))
# print(keys)

registers_df = pd.DataFrame()
for i, dic in enumerate(registers):
	for key, value in dic.items():
		registers_df.loc[i, str(key)] = value

#Checking NAs
registers_df.isnull().any().any()


