# Predicting thermal inertia of HVAC installations
**Project in the field of IoT for building energy systems in cooperation with Indoorclima**

The smart management of HVAC installations leads to energy savings of 5% to 20%
A thermal inertia algorithm should indicate when to power on/off the HVAC system to reach desired temperature at the desired time.
Inertia model developed in the past running with limitations: not considered external temperature and can't be trained on more than 2-3 months of data.
<br>
<br>
The objectives of this project are:

- Create a model for the prediction of thermal inertia during power on and power off.
- Improve error metrics by means of feature selection/engineering as compared with the model currently used
- Create a model which can be trained on 1 year data without negative impacting error metrics

The report can be seen [here.](https://github.com/edidduplan/Project_Ernst_Indoorclima/blob/master/Presentations/Project%20Ernst.pdf)<br>
The modelling script [here.](https://nbviewer.jupyter.org/github/edidduplan/Project_Ernst_Indoorclima/blob/master/Jupyter%20Project/Modelling%20revisited.ipynb)
<br>
<br>
**Current development**  
<br>
Objectives:

- Scale-up to all locations and integrate into production (Azure DevOps)
- Use forecasts of outer temperature to further improve the model
- Use dummy variables to consider the occupation level of buildings (big stores)
<br>

First iteration of current development can be seen [here.](https://github.com/edidduplan/Project_Ernst_Indoorclima/tree/master/New%20Project)<br>

