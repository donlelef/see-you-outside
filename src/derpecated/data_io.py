import pandas as pd
import numpy as np
import sys
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt

data = pd.read_csv("https://raw.githubusercontent.com/daenuprobst/covid19-cases-switzerland/master/covid19_hospitalized_switzerland_openzh.csv")
    

def load_data():
    temp = pd.read_csv("https://raw.githubusercontent.com/daenuprobst/covid19-cases-switzerland/master/covid19_cases_switzerland_openzh.csv")
    data = []
    data.append(temp.filter(items=["CH"]).to_numpy().squeeze()) # total cases
    temp = pd.read_csv("https://raw.githubusercontent.com/daenuprobst/covid19-cases-switzerland/master/covid19_hospitalized_switzerland_openzh.csv")
    data.append(temp.filter(items=["CH"]).to_numpy().squeeze()) # hospitalized
    temp = pd.read_csv("https://raw.githubusercontent.com/daenuprobst/covid19-cases-switzerland/master/covid19_fatalities_switzerland_openzh.csv")
    data.append(temp.filter(items=["CH"]).to_numpy().squeeze()) # death
     

    return np.asarray(data)

