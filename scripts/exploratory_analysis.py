import pandas as pd
import numpy as np
import json
import xmltodict
import datetime
import os
import matplotlib
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
import plotly.io as pio
import plotly.offline as py
# ------------------------------------------------------
datafilenamepath = "C:\\Users\\edidd\\Documents\\Ubiqum\\Data Analytics Course\\Project_Heike2\\Data"

df_2020_01 = pd.read_csv(os.path.join(datafilenamepath, "df_2020_01.csv"))

#Checking for NAs
df_2020_01.isnull().any().any()

#Droping rows with NAs
df_2020_01 = df_2020_01.dropna()

#Exploring the variables
description = df_2020_01.describe()

#Looking at the distribution of data
df_names
nbins = int((df_2020_01["Paro - marcha"].max() - df_2020_01["Tª interior"].min())/.2)
plt.hist(df_2020_01["Potencia Rooftop"], bins = 10)
plt.show()

#Scatter plots
plt.scatter(x= df_2020_01["Potencia Rooftop"], y= df_2020_01["Tª interior"])

plt.title("Potencia Rooftop")
plt.xlabel("Potencia Rooftop")
plt.ylabel("Tª interior")
plt.show()

plt.show()

# Feature selection I
df_2020_01 = df_2020_01.drop(["Paro - marcha", "Alarma", "Algoritmos", "Rooftop", "Energía Contador", "CO2",
                              "Potencia General", "Presión Alta C1", "Presión Baja C1", "Presión Alta C2",
                              "Presión Baja C2", "Presión Alta C3", "Presión Alta C4", "Presión Baja C3",
                              "Presión Baja C4", "Tª SET verano", "Tª SET invierno"], axis = 1)

# Handling time variable
df_2020_01.info()
df_2020_01["datetime"] = pd.to_datetime(df_2020_01["datetime"])
test = df_2020_01.loc[df_2020_01.datetime.dt.day == 15, :]
test.info()

#Plotting
time = matplotlib.dates.date2num(test["datetime"])
plt.plot_date(x= time, y= test["Potencia Rooftop"])
plt.plot_date(x= time, y= test["Tª interior"])
plt.title("Potencia Rooftop vs time")
plt.xlabel("time")
plt.ylabel("Potencia Rooftop")
plt.show()

#Trying plotly
pio.renderers.default = 'browser'
fig = go.Figure()
fig.add_trace(go.Scatter(x= test["datetime"],
                         y= test["Potencia Rooftop"],
                         mode = "lines",
                         name= "Potencia Rooftop"))
fig.add_trace(go.Scatter(x= test["datetime"],
                         y= test["Tª interior"],
                         mode = "lines",
                         name= "T interior"))
fig.add_trace(go.Scatter(x= test["datetime"],
                         y= test["Tª exterior"],
                         mode = "lines",
                         name= "T exterior"))

pio.show(fig)
pio.renderers

df_2020_01.dtypes.index