import pandas as pd
import numpy as np
import datetime
import os
import matplotlib
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
import plotly.io as pio
import plotly.offline as py
import seaborn as sns
from sklearn.decomposition import PCA
from numpy import array, mean, cov
from numpy.linalg import eig
from sklearn.preprocessing import StandardScaler
import re
from collections import Counter
from json import loads, dumps

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
datafilenamepath = r"..\test\data"# File path where the extracted data will be copied

#=============== API: user data ================
apiUserData = pd.read_csv(os.path.join(os.getcwd(), "userdata.csv"), index_col= 0)
password = apiUserData.loc["password"].values[0]
user = apiUserData.loc["user"].values[0]
api_url = apiUserData.loc["api_url"].values[0]


def to_dict(input_ordered_dict):
    return loads(dumps(input_ordered_dict))

#============== Function: extract data ==============
def extract_data(cli_id, data_ini, data_fi):
    parameters_url = api_url + user + "/" + password + "/" + "parameters_cli?id=" + str(cli_id)
    # Sending request to the url
    parameters_response = requests.get(parameters_url)
    # Parsing ordered dictionary with xmltodic
    parameters_dic = xmltodict.parse(parameters_response.content)
    # Navigating down in the dictionary to get to the registers level
    parameters = parameters_dic["info_sgclima"]["parameter"]

    # Registers: parsing of registers for all parameters
    parameters_list = []
    start_time = time.time()
    df_dic, dic_ps = {}, {}
    
    #lista con todos los parametros de esa instalacion
    for n, par in enumerate(parameters):
        dic_ps[par["alias"]] = par["ps_id"] +"_"+par["loc_id"]
     
    #busqueda de los parametros que nos interesan
    print('la lista inicial es')
    print(dic_ps.keys())
    pattern = '((u|U|m|M)[0-9]|ver-inv|exterior|time|((impulsi(o|ó)n)|onsigna) (c|C)olector)'
    features= []
    for i in dic_ps.keys():
        if re.search(pattern, i) !=None:
            features.append(i)
            
    pattern1 = '(a|A)larma'
    pattern2 = '(C|c)ond'
    pattern3 = '(E|e)nfriadora'
    pattern4 = 'acidad|emanda|umedad|ondensaci'
    pattern5 = '(P|p)aro'

    eliminar = []

    for i in features:
        if  re.search(pattern1, i) !=None :
            eliminar.append(i)

    for i in features:
        if  re.search(pattern2, i) !=None :
            eliminar.append(i)

    for i in features:
        if  re.search(pattern3, i) !=None :
            eliminar.append(i)

    for i in features:
        if  re.search(pattern4, i) !=None :
            eliminar.append(i)
            
    for i in features:
        if  re.search(pattern5, i) !=None :
            eliminar.append(i)

    features= list(set(features) - set(eliminar))                              
            
    #variable que guardara la feature donde mirar las sumas de potencia       
    pattern1 = '(E|e)nfriadoras'
    pattern2 = '(P|p)ot'
    parametro_suma_potencias = ""
    for i in dic_ps.keys():
        if re.search(pattern1, i) != None and re.search(pattern2, i) != None:
            parametro_suma_potencias = i
            features.append(i)

    if parametro_suma_potencias == "":
        pattern1 = '(c|C)lima'
        for i in dic_ps.keys():
            if re.search(pattern1, i) != None and re.search(pattern2, i) != None:
                parametro_suma_potencias = i
                features.append(i)
                print('holahola')
                print(parametro_suma_potencias)
                print(features)
        
            
    pattern1 = '(impulsi(o|ó)n (c|C)olector)' 
    for i in dic_ps.keys():
        if re.search(pattern1, i) != None:
                t_objetivo = i
                features.append(i)
    
    features= list(set(features))
    print(features)
    #buscar todos lo parametros que me interesan 
    for n, par in enumerate(parameters):
        # if {par["alias"]}.intersection(para_to_drop) == set():
        ps_id = par["ps_id"]
        alias = par["alias"]
        if alias in features:
            loc_id_i = par["loc_id"]
            registers_url = api_url + user + "/" + password + "/" + "registers?id=" + loc_id_i + "&ps_id=" + ps_id + "&data_ini=" + data_ini + "&data_fi=" + data_fi
            registers_response = requests.get(registers_url)
            # registers_soup = BeautifulSoup(registers_response.text, 'lxml')
            registers_soup = BeautifulSoup(registers_response.text, 'html.parser')

            col_names = ["valor", "data", "hora", "ps_id"]
            registers_df = pd.DataFrame()
            for i in col_names:
                registers_df[i] = [x.get_text() for x in registers_soup.find_all(i)]
            print(alias)
            print(registers_df)
            if  not registers_df.empty:
                df_dic[alias] = registers_df
                #print(df_dic[alias])
            print(str(n + 1), "/", str(len(parameters)))


    #df_dic: diccionario con un dataframe por parametro con las 4 columas
    print("--- %s seconds ---" % (time.time() - start_time))
    
    # Changing data types and generating datetime column
    for i, df in df_dic.items():

        df["datetime"] = df["data"] + df["hora"]
        df["datetime"] = pd.to_datetime(df["datetime"], format="%Y-%m-%d%H:%M:%S")
        df_dic[i] = df.drop(["data", "hora", "ps_id"], axis=1)
        
    #df_dic: diccionario con un dataframe por parametro con dos columnas: valor y datetime
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

    data = df_merge

    """
    #leer los datos de infolocs con las potencias de paro y de marcha y buscar esta instal.
    parameters_url = api_url + user + "/" + password + "/" + "infoloc"
    # Sending request to the url
    parameters_response = requests.get(parameters_url)
    # Parsing ordered dictionary with xmltodic
    parameters_dic = xmltodict.parse(parameters_response.content)
    # Navigating down in the dictionary to get to the registers level
    infos = parameters_dic["info_sgclima"]["infoloc"]
    potencias = {}
    for info in infos:
        potencias[to_dict(info)['id']]={'pot_arranque' : to_dict(info)['pot_arranque'],'pot_parada' : to_dict(info)['pot_parada'] }
    print(potencias)
        
     
    dic_infoloc = {}
    for i in range(infoloc.shape[0]):
        dic_infoloc[infoloc['id'].iloc[i]] = {'pot_arranque':infoloc['pot_arranque'].iloc[i], 'pot_parada':infoloc['pot_parada'].iloc[i]}

    loc_id = int(datafilename[:3])
    if dic_infoloc.get(loc_id) == None:
        potencia_on, potencia_of = 20, 1
    else:
        potencia_on, potencia_of = dic_infoloc.get(loc_id)['pot_arranque'], dic_infoloc.get(loc_id)['pot_parada']
        
 
    if potencias.get(loc_id) == None:
        potencia_on, potencia_of = 20, 1
    else:
        potencia_on, potencia_of = potencias.get(loc_id)['pot_arranque'], potencias.get(loc_id)['pot_parada']
    """ 
    #nueva modalidad de lectura infolocs

    potencia_on, potencia_of = "" , ""
    parameters_url = api_url + user + "/" + password + "/" + "infocli?id=" + str(cli_id)
    # Sending request to the url
    parameters_response = requests.get(parameters_url)
    # Parsing ordered dictionary with xmltodic
    parameters_dic = xmltodict.parse(parameters_response.content)
    # Navigating down in the dictionary to get to the registers level
    infos = parameters_dic["info_sgclima"]["infocli"]
    
    """
    for n, par in enumerate(infos):
        print(par)
        if par["pot_arranque"]== 0:
            potencia_on = par["pot_arranque"]
        if par["pot_parada"]== 0:
            potencia_of = par["pot_parada"]
    """
    potencia_on, potencia_of = 50, 10  
    #buscar de todas las columnas de datos las que seran nuestras features

    #subtitute alias name for ps_id
    
    #for alias in list(data):
        #if alias!= 'datetime':
            #print(alias)
           # print(dic_ps[alias])
           # data.rename(columns={alias:dic_ps[alias]})
        
    # poner en orden dataframe e interpolar todos los NaN
    #data['datetime'] = pd.to_datetime(data['datetime'], format="%Y-%m-%d %H:%M:%S")
    #data["Ver - Inv Instalación"] = data["Ver - Inv Instalación"].apply(str)
    data = data.sort_values('datetime').reset_index().drop("index", axis = "columns")
    data = data.set_index('datetime')
    data.index = pd.to_datetime(data.index)
    for column in list(data):
        data[column] = pd.to_numeric(data[column])
    print('data 2')
    print(data)
    data = data.interpolate(method='time',limit_direction='forward', axis=0) # Line corrected
    data = data.replace("nan", np.nan).fillna(method='ffill').fillna(method='bfill')
    data.isnull().any().any()
    #data = data.asfreq(pd.Timedelta(minutes= 5))
    data = data.resample('5T').mean()
    data = data.interpolate(method='time', limit_direction='forward', axis=0)
    data = data.fillna(method='ffill')
    data = data.reset_index()
    

    print(data)
    return data, potencia_on, potencia_of, parametro_suma_potencias, t_objetivo, dic_ps

#=================================== Debug ==========================================
# df = data.copy()
# df.to_csv(os.path.join(os.getcwd(), "data", "df_195_2018-03-01_2018-05-31.csv"), index= False)
# df = pd.read_csv(os.path.join(os.getcwd(), "data", "df_195_2018-03-01_2018-05-31.csv"))
#
# var_potencia = "Potencia Enfriadoras"
# t_objetivo = "Tª impulsión colector"
# power_on = 50
# power_off = 10
# i = 0
# time = df_hora_arranque[i]
# df_copy = df_arranque.copy()
# arranque = subset_arranque(df, var_potencia, t_objetivo, power_on, power_off)

def subset_arranque(df, var_potencia, t_objetivo, power_on, power_off, dic_ps):
    #Array with the power on times
    df_hora_arranque = df[(df[var_potencia].shift(-1) >= power_on) & (df[var_potencia] < power_off)]["datetime"].values
    #For loop to append 1 hour of data after each power-on one dataset after the other
    df_arranques = pd.DataFrame()
    df_arranque= pd.DataFrame()
    for i, time in enumerate(df_hora_arranque):
        #One single arranque dataset:
        df_arranque = df.loc[(df.datetime >= time) & (df.datetime < (time + pd.Timedelta(hours=1))), :]
        df_arranque = df_arranque.reset_index()
        df_arranque = df_arranque.drop(["index"], axis= "columns")
        #Creating cummulative sum DT and cummulative mean Potencia
        df_arranque["DT"] = (df_arranque[t_objetivo].shift(-1) - df_arranque[t_objetivo]).fillna(0)
        df_arranque["cumsumDT"] = df_arranque.DT.cumsum()
        #Update: the cummulative mean power is not useful, since it won't be an input in the prediction.
        #Change criteria to select the duration of the power_on/off using now the actual power
        #df_arranque["cummeanPotencia"] = df_arranque[var_potencia].cumsum() / pd.Series(np.arange(1, len(df_arranque[var_potencia])+1), df_arranque[var_potencia].index)
        #df_arranque["cummeanPotencia"] = np.where((df_arranque[var_potencia] <= power_off).cumprod().astype("bool"), df_arranque["cummeanPotencia"], np.nan)
        df_arranque.loc[1:, var_potencia] = np.where((df_arranque[var_potencia][1:] >= power_on).cumprod().astype("bool"),
                                                 df_arranque[var_potencia][1:], np.nan)
        #Creating "DTnames", "Pmean_names" and "arranque" columns
        df_arranque["DTnames"] = pd.Series(["DT05", "DT10", "DT15", "DT20", "DT25", "DT30", "DT35", "DT40", "DT45",  "DT50", "DT55", "DT60" ])
        df_arranque["Pmean_names"] = pd.Series(["P05", "P10", "P15", "P20", "P25", "P30", "P35", "P40", "P45", "P50", "P55", "P60" ])
        df_arranque = df_arranque.assign(arranque= "arranque" + str(i+1)) 
        #Pivoting "DTnames" and "Pmean_names"
        columns_DTnames = df_arranque.pivot(columns= "DTnames", values= "cumsumDT", index= "arranque").reset_index()
        columns_Pmean_names = df_arranque.pivot(columns= "Pmean_names",  values= var_potencia, index= "arranque").reset_index()
        #----------------- Renaming columns to "psid_locid"--------------
        df_arranque.drop(["datetime", "Pmean_names", "DTnames", "cumsumDT", "DT", "arranque"], axis= 1, inplace= True)
        psid_potencia = dic_ps[var_potencia]
        for col in df_arranque.columns:
            df_arranque.rename(columns={col: dic_ps[col]}, inplace= True)
        df_arranque.rename(columns= {psid_potencia: "p" + psid_potencia}, inplace= True)
        #----------------------------------------------------------------
        df_arranque = df_arranque.assign(arranque= "arranque" + str(i+1))
        df_arranque = pd.merge(df_arranque.iloc[0:1, :], pd.merge(columns_Pmean_names, columns_DTnames, on= "arranque"), on= "arranque")
        #Appending one "arranque" row after the other
        df_arranques = df_arranques.append(df_arranque)
    df_arranques = df_arranques.reset_index(drop= True)
    df_arranques = df_arranques.drop(["arranque"], axis= "columns")
    # Dropping features with zero variance
    df_arranques = df_arranques.loc[:, df_arranques.describe().loc["std"] != 0]

    return df_arranques

# prueba_arranque = subset_arranque(df, var_potencia, t_objetivo, power_on, power_off)
# prueba_arranque_test = subset_arranque(df, var_potencia, t_objetivo, power_on, power_off, dic_ps)
#
# df_arranques = prueba_arranque.copy()

def subset_parada(df, var_potencia, t_objetivo, power_on, power_off, dic_ps):
    #Array with the power of times
    df["datetime"] = pd.to_datetime(df["datetime"])
    df_hora_parada = df[(df[var_potencia].shift(-1) < power_off) & (df[var_potencia] > power_on)]["datetime"].values

    print('holita')
    print(df_hora_parada)

    #For loop to append 1 hour of data after each power-on one dataset after the other
    df_paradas = pd.DataFrame()
    df_parada= pd.DataFrame()
    for i, time in enumerate(df_hora_parada):
        #One single parada dataset:
        df_parada = df.loc[(df.datetime >= time) & (df.datetime < (time + pd.Timedelta(hours=1))), :]
        df_parada = df_parada.reset_index()
        df_parada = df_parada.drop(["index"], axis= "columns")
        #Creating cummulative sum DT and cummulative mean Potencia
        df_parada["DT"] = (df_parada[t_objetivo].shift(-1) - df_parada[t_objetivo]).fillna(0)
        df_parada["cumsumDT"] = df_parada.DT.cumsum()
        # Update: the cummulative mean power is not useful, since it won't be an input in the prediction.
        # Change criteria to select the duration of the power_on/off using now the actual power
        #df_parada["cummeanPotencia"] = df_parada[var_potencia].cumsum() / pd.Series(np.arange(1, len(df_parada[var_potencia])+1), df_parada[var_potencia].index)
        #df_parada["cummeanPotencia"] = np.where((df_parada[var_potencia] <= power_off).cumprod().astype("bool"), df_parada["cummeanPotencia"], np.nan)
        df_parada.loc[1:, var_potencia] = np.where((df_parada[var_potencia][1:] < power_off).cumprod().astype("bool"),
                                               df_parada[var_potencia][1:], np.nan)
        #Creating "DTnames", "Pmean_names" and "parada" columns
        df_parada["DTnames"] = pd.Series(["DT05", "DT10", "DT15", "DT20", "DT25", "DT30", "DT35", "DT40", "DT45",  "DT50", "DT55", "DT60" ])
        df_parada["Pmean_names"] = pd.Series(["P05", "P10", "P15", "P20", "P25", "P30", "P35", "P40", "P45", "P50", "P55", "P60" ])
        df_parada = df_parada.assign(parada= "parada" + str(i+1)) 
        #Pivoting "DTnames" and "Pmean_names"
        columns_DTnames = df_parada.pivot(columns= "DTnames", values= "cumsumDT", index= "parada").reset_index()
        columns_Pmean_names = df_parada.pivot(columns= "Pmean_names",  values= var_potencia, index= "parada").reset_index()
        # ----------------- Renaming columns to "psid_locid"--------------
        df_parada.drop(["datetime", "Pmean_names", "DTnames", "cumsumDT", "DT", "parada"], axis=1, inplace=True)
        psid_potencia = dic_ps[var_potencia]
        for col in df_parada.columns:
            df_parada.rename(columns={col: dic_ps[col]}, inplace=True)
        df_parada.rename(columns={psid_potencia: "p" + psid_potencia}, inplace=True)
        # ----------------------------------------------------------------
        df_parada = df_parada.assign(parada= "parada" + str(i+1))
        df_parada = pd.merge(df_parada.iloc[0:1, :], pd.merge(columns_Pmean_names, columns_DTnames, on= "parada"), on= "parada")
        #Appending one "parada" row after the other
        df_paradas = df_paradas.append(df_parada)
    df_paradas = df_paradas.reset_index(drop= True)
    df_paradas = df_paradas.drop(["parada"], axis= "columns")
    # Dropping features with zero variance
    df_paradas = df_paradas.loc[:, df_paradas.describe().loc["std"] != 0]

    return df_paradas

# prueba_parada = subset_parada(df, var_potencia, t_objetivo, power_on, power_off)
# prueba_parada_test = subset_parada(df, var_potencia, t_objetivo, power_on, power_off, dic_ps)

def divide_stations(data):

    pattern1 = 'ver-inv' 
    ver_inv = 'no'
    for i in data.keys().tolist():
        if re.search(pattern1, i) != None:
                ver_inv = i
    if (ver_inv == 'no'):
        df_ver, df_inv  = data, 'false'
    else:
        df_ver, df_inv  = data[data['ver-inv']== 0], data[data['ver-inv']== 1]
        

    return df_ver, df_inv

# df_ver, df_inv= divide_stations(data)

#def main(csv, file_path):
if __name__ == '__main__':
    #extraer datos, le mandas un cli_id y unas fechas y te devuelve 4 dataframes (ver/inf on/of) y las potencias de power on/of
    loc_ids = [195]#[816, 825, 912, 716]
    data_ini = '2019-01-15'
    data_fi = '2019-12-15'
    for loc_id in loc_ids:
        df, power_on, power_off, var_potencia, t_objetivo, dic_ps = extract_data(loc_id, data_ini, data_fi)
        print('var_potencia')
        print(var_potencia)
        print('t_objetivo')
        print(t_objetivo)

        print('power_on')
        print(power_on)
        print('power_off')
        print(power_off)
        """
        if 'datetime' in list(df):
            del df['datetime']
            """
        #dividir verano e invierno
        df_ver, df_inv = divide_stations(df)
        #parada verano
        df_ver_off = subset_parada(df_ver, var_potencia, t_objetivo, power_on, power_off, dic_ps)
        df_ver_off.to_csv(os.path.join(
            datafilenamepath,
            ("ver_parada" + str(loc_id) + "_df" + "_" + str(data_ini) +"_to_" + str(data_fi) + ".csv")), index=False, header=True)

        #encendido verano
        df_ver_on = subset_arranque(df_ver, var_potencia, t_objetivo, power_on, power_off, dic_ps)
        df_ver_on.to_csv(os.path.join(
            datafilenamepath,
            ("ver_arranque" + str(loc_id) + "_df" + "_" + str(data_ini) +"_to_" + str(data_fi) + ".csv")), index=False, header=True)

        print('df_ver_off')
        print(df_ver_off)
        print('df_ver_on')
        print(df_ver_on)

        if (df_inv != 'false' and not df_inv.empty):
            #parada invierno
            df_inv_off =  subset_parada(df_inv, var_potencia, t_objetivo, power_on, power_off, dic_ps)
            df_inv_off.to_csv(os.path.join(
            datafilenamepath,
            ("inv_parada" + str(loc_id) + "_df" + "_" + str(data_ini) +"_to_" + str(data_fi) + ".csv")), index=False, header=True)
            #encendido invierno
            df_inv_on = subset_arranque(df_inv, var_potencia, t_objetivo, power_on, power_off, dic_ps)
            df_inv_on.to_csv(os.path.join(
            datafilenamepath,
            ("inv_arranque" + str(loc_id) + "_df" + "_" + str(data_ini) +"_to_" + str(data_fi) + ".csv")), index=False, header=True)



