import pandas as pd
import numpy
import sys
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt

data = pd.read_csv("https://raw.githubusercontent.com/daenuprobst/covid19-cases-switzerland/master/covid19_hospitalized_switzerland_openzh.csv")
    

def load_data():
    data = pd.read_csv("https://raw.githubusercontent.com/daenuprobst/covid19-cases-switzerland/master/covid19_cases_switzerland_openzh.csv")
    total_case = data.filter(items='CH')

    return data

